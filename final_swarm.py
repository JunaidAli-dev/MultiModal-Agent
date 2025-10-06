import os
import operator
import json
import base64
from typing import TypedDict, Annotated, List
import sqlite3

# Core LangChain and LangGraph components
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, BaseMessage
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Pre-defined tools
from langchain_experimental.tools import PythonREPLTool
from langchain_tavily import TavilySearch
try:
    from information_model import create_information_agent
except ImportError:
    def create_information_agent():
        print("WARNING: 'information_model' not found. Using a dummy tool.")
        class DummyInfoTool:
            def invoke(self, query): return f"This is a dummy response for the query: '{query}'"
        return DummyInfoTool()

# --- 1. Define All Tools for the Swarm ---
class InformationTool(BaseModel):
    """Call this to retrieve information from the local financial document."""
    query: str = Field(..., description="The specific question to ask the information agent.")
class MathTool(BaseModel):
    """Call this to perform a calculation. The query must be a valid Python expression."""
    expression: str = Field(..., description="The Python expression to evaluate.")
class WebSearchTool(BaseModel):
    """Call this to search the web for up-to-date information."""
    query: str = Field(..., description="The search query for the web search agent.")
class ChartAnalysisTool(BaseModel):
    """Call this to analyze an image of a chart, figure, or table from a local file."""
    image_path: str = Field(..., description="The local file path to the image.")
    question: str = Field(..., description="The specific question to ask about the image's content.")
# NOTE: SummarizationTool is removed as a tool, as the LLM can summarize natively.

# --- 2. Enhanced Agent State with Planning ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    trace: Annotated[List, operator.add]
    plan: List[str]  # The step-by-step plan
    past_steps: Annotated[List, operator.add] # Results of completed steps

# --- 3. Persistence with SqliteSaver ---
conn = sqlite3.connect("swarm_memory.sqlite", check_same_thread=False)
memory = SqliteSaver(conn=conn)

# --- 4. Initialize Tools and LLMs ---
information_agent_tool = create_information_agent()
python_repl_tool = PythonREPLTool()
web_search_tool = TavilySearch(max_results=2)

# We use a powerful model for planning and reasoning
reasoning_llm = ChatGoogleGenerativeAI(model="models/gemini-pro-latest", temperature=0)
# And a cost-effective model for vision tasks
vision_llm = ChatGoogleGenerativeAI(model="models/gemini-2.5-flash", temperature=0)

available_tools = [InformationTool, MathTool, WebSearchTool, ChartAnalysisTool]
reasoning_llm_with_tools = reasoning_llm.bind_tools(available_tools)

# --- 5. Planner, Controller (Executor), and Tool Nodes ---

# In final_swarm.py

def planner_node(state: AgentState):
    print("---PLANNER: Devising a step-by-step plan---")
    user_query = state["messages"][-1].content
    
    # This is a much more robust "few-shot" prompt
    planner_prompt = f"""You are an expert planning agent. Your job is to create a clear, numbered, step-by-step plan to answer a user's complex query.

**Key Instructions:**
1.  **Decompose:** Break the query down into smaller, logical steps.
2.  **Information First:** For any calculation, you MUST add a preceding step to find and extract the necessary numbers first.
3.  **Tool-Oriented:** Each step must be a specific action that maps directly to one of the available tools.
4.  **No Conversation:** If no tools are required for the query, respond with an empty plan.

**Example:**
USER QUERY: "What was the total revenue, and what was the profit margin if the cost was $100?"
YOUR PLAN:
1. Find the total revenue from the document.
2. Calculate the profit margin using the revenue figure and the provided cost of $100.

**Now, create a plan for the following query:**

USER QUERY: "{user_query}"

AVAILABLE TOOLS: InformationTool (local financial doc), MathTool (calculations), WebSearchTool (current events), ChartAnalysisTool (image analysis).

Respond with ONLY the numbered list of steps. If no tools are required, respond with an empty plan.
"""
    response = reasoning_llm.invoke(planner_prompt)
    plan = [line.strip().split(". ", 1)[1] for line in response.content.split("\n") if ". " in line]
    return {"plan": plan}

# The improved controller_node function

def controller_node(state: AgentState):
    # Case 1: The plan is complete, and we have results to synthesize. (This logic is perfect)
    if not state.get("plan") and state.get("past_steps"):
        print("---CONTROLLER: Plan complete. Synthesizing final answer.---")
        synthesis_prompt = f"""Synthesize a final, comprehensive answer for the user based on their initial query and the results of the completed plan steps.
    Your persona is a sophisticated financial AI assistant. Be helpful, clear, and detailed in your financial analysis.

    INITIAL QUERY: {state['messages'][0].content}

    COMPLETED STEPS & RESULTS:
    {state['past_steps']}

    Provide the final answer in a clear, human-readable format.
    """
        response = reasoning_llm.invoke(synthesis_prompt)
        return {"messages": [AIMessage(content=response.content)]}

    # Case 2: The plan is empty from the start. (This logic is also perfect)
    elif not state.get("plan"):
        print("---CONTROLLER: No plan needed. Answering directly.---")
        direct_answer_prompt = f"""You are a helpful and friendly AI assistant. Please provide a direct, conversational response to the user's query.

    User Query: "{state['messages'][0].content}"
    """
        response = reasoning_llm.invoke(direct_answer_prompt)
        return {"messages": [AIMessage(content=response.content)]}

    # Case 3: The plan is not empty, execute the next step. (With an improved prompt)
    else:
        print("---CONTROLLER: Executing next step---")
        next_step = state["plan"][0]
        
        # --- IMPROVED EXECUTOR PROMPT ---
        executor_prompt = f"""You are an execution agent. Your sole job is to choose the single best tool to complete the next step of a plan.
    Do not attempt to answer the question yourself. Only select a tool.

    The user's overall goal is: "{state['messages'][0].content}"

    The next step you must execute is: "{next_step}"

    Here are the available tools: {", ".join([tool.__name__ for tool in available_tools])}

    Based on the next step, which tool should you call?
    """
        # --- END IMPROVEMENT ---
        
        response = reasoning_llm_with_tools.invoke(executor_prompt)
        return {"messages": [response]}

def execute_tool_node(state: AgentState):
    tool_call = state["messages"][-1].tool_calls[0]
    tool_name = tool_call["name"]
    tool_input_dict = tool_call["args"]
    
    current_step = state["plan"][0]
    print(f"---NODE: Executing '{tool_name}' for step: '{current_step}'---")
    
    try:
        if tool_name == "InformationTool":
            response_str = str(information_agent_tool.invoke(tool_input_dict["query"]))
        elif tool_name == "MathTool":
            response_str = str(python_repl_tool.invoke(tool_input_dict["expression"]))
        elif tool_name == "WebSearchTool":
            response_str = str(web_search_tool.invoke(tool_input_dict["query"]))
        elif tool_name == "ChartAnalysisTool":
            with open(tool_input_dict['image_path'], "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            msg = vision_llm.invoke([HumanMessage(content=[{"type": "text", "text": tool_input_dict['question']}, {"type": "image_url", "image_url": {"mime_type": "image/jpeg", "data": image_data}}])])
            response_str = msg.content
        else:
            response_str = "Error: Tool not found."
    except Exception as e:
        print(f"---ERROR in '{tool_name}': {e}---")
        response_str = f"Error executing tool '{tool_name}': {e}. The planner may need to try a different approach."

    # Update state: record step result, remove completed step from plan
    updated_plan = state["plan"][1:]
    
    trace_entry = {
        "agent": f"{tool_name} Agent",
        "tool": tool_name,
        "input": tool_input_dict,
        "output": response_str,
        "handoff-to": "controller"
    }

    return {
        "past_steps": [(current_step, response_str)],
        "plan": updated_plan,
        "messages": [ToolMessage(content=response_str, tool_call_id=tool_call['id'])],
        "trace": [trace_entry]
    }

# --- 6. Define the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("planner", planner_node)
workflow.add_node("controller", controller_node)
workflow.add_node("execute_tool", execute_tool_node)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "controller")
workflow.add_edge("execute_tool", "controller")

def route_logic(state: AgentState):
    # If there are tool calls, execute them
    if state["messages"][-1].tool_calls:
        return "execute_tool"
    # Otherwise, the plan is complete, so end
    return END

workflow.add_conditional_edges("controller", route_logic)
app = workflow.compile(checkpointer=memory)
print("✅ Definitive Planner-Executor Swarm compiled.")

# --- 7. Run the Interactive Swarm ---
if __name__ == "__main__":
    # Each user or session gets a unique thread_id
    config = {"configurable": {"thread_id": "user-session-1"}} 
    
    while True:
        query = input("Please enter your question (or 'reset', 'exit'): ")
        if query.lower() == "exit": break
        if query.lower() == "reset":
            # To reset, we can just assign a new thread_id
            import uuid
            config["configurable"]["thread_id"] = str(uuid.uuid4())
            print(f"--- Session has been reset. New session ID: {config['configurable']['thread_id']} ---")
            continue
        if not query.strip(): continue

        initial_state = {"messages": [HumanMessage(content=query)]}
        print(f"\n--- Starting Swarm with Query: '{query}' ---")
        
        final_state = None
        # The stream now gives us the complete state at each step
        for event in app.stream(initial_state, config=config, stream_mode="values"):
            final_state = event

        final_answer = final_state['messages'][-1].content
        trace = final_state.get('trace', [])

        log_output = {
            "query": query,
            "trace": trace,
            "final_answer": final_answer
        }
   
        print("\n--- Structured Log ---")
        print(json.dumps(log_output, indent=2, default=str)) # Use default=str to handle non-serializable objects
             
        print("\n--- Final Answer ---")
        print(final_answer)
        
        print("-" * 20)