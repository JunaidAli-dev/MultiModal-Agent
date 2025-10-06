# 📈 Financial Analysis AI Agent Swarm

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A sophisticated multi-agent AI system designed for in-depth, multimodal financial analysis of documents. This project leverages a **Planner-Executor** architecture built with LangChain and LangGraph to handle complex, multi-step queries that require reasoning over both text and visual elements like tables.

---

## 🚀 Core Features

* **🧠 Strategic Planning:** Decomposes complex user questions into a logical, step-by-step plan before execution, ensuring thorough analysis.
* **👁️ Multimodal Reasoning:** Capable of analyzing and integrating information from text, tables, and charts (as images) within a document.
* **💾 Persistent Memory:** Utilizes a SQLite database (`swarm_memory.sqlite`) to maintain conversation history, allowing the agent to remember context across sessions.
* **🛠️ Robust Tool Usage:** Equipped with a suite of tools for document querying (RAG), mathematical calculations, and web searches.
* **⚙️ Advanced Error Handling:** The agent can gracefully handle tool failures and its prompts are designed to re-plan or report issues clearly.
* **📊 Detailed Explainability:** Generates a structured JSON log for every query, detailing each step of the plan, the tool used, its input/output, and the final answer for complete transparency.

---

## 🏗️ Architectural Approach

The system is designed with a clear separation of concerns, divided into three distinct stages: Data Processing, Information Retrieval, and Agentic Reasoning.

### Stage 1: Data Processing & Multimodal Vector Store Creation (`run_task1.py`)

This initial stage transforms a raw PDF financial report into a structured, searchable "brain" for the AI. The core innovation is the creation of a **multimodal vector store** that respects the visual layout of data.

1.  **PDF Rendering:** The process begins by rendering each page of the PDF into a high-resolution image using `PyMuPDF` (`fitz`). This creates a visual canvas for element extraction.
2.  **Intelligent Partitioning:** The `unstructured.io` library is used with its `hi_res` strategy to parse the PDF. It acts as a computer vision model, identifying the *type* (e.g., text, table) and *location* (bounding box coordinates) of each content block.
3.  **Visual Element Extraction:** For each element identified as a table, the script uses the bounding box coordinates to crop a "screenshot" of that specific table from the pre-rendered page image. This technique is superior to text extraction as it perfectly preserves the visual structure of rows and columns, which is critical for financial data.
4.  **Data Structuring:** All processed content is standardized into LangChain `Document` objects.
    * **For Text:** The `page_content` is the text itself.
    * **For Tables:** The `page_content` is a simple placeholder description (e.g., *"Image of a table from page 5"*). The crucial data—the cropped table image—is encoded into a `base64` string and stored in the document's `metadata`, along with a `type: "image"` flag.
5.  **Vectorization & Indexing:** Google's `embedding-001` model converts the `page_content` of every document into a numerical vector. These vectors, along with their corresponding documents (including the metadata with images), are indexed into a `FAISS` vector store. This creates a highly efficient, searchable database where queries can find relevant information based on semantic meaning.

### Stage 2: The Information Retrieval Agent (`information_model.py`)

This file defines the `InformationTool`, the agent's specialized interface to the vector store. It's a complete **Retrieval-Augmented Generation (RAG)** chain designed for multimodal context.

1.  **Loading & Retrieval:** The agent loads the `FAISS` index and the same embedding model used to create it. It uses a **Maximum Marginal Relevance (MMR)** retriever, which is smarter than a simple similarity search because it fetches documents that are both relevant to the query and diverse from each other, providing a richer context.
2.  **Multimodal Context Assembly:** A custom function, `format_retrieved_documents`, is the core of this module. When documents are retrieved:
    * It concatenates all text content into a single block.
    * It inspects the `metadata` of each document. If a document is of `type: "image"`, it unpacks the `base64` image data.
    * It assembles a final payload for the LLM that includes both the collected text and the actual images, ready for a multimodal vision model.
3.  **Generation:** The assembled context is passed to a multimodal LLM (like `gemini-2.5-flash-lite`). The model is prompted to answer the user's question based *only* on this provided context, ensuring its answers are grounded in the document's data.

### Stage 3: The Agentic Reasoning Swarm (`final_swarm.py`)

This is the master orchestrator, built using **LangGraph**. It creates a state machine that can reason, plan, and execute complex tasks using the available tools.

1.  **The Planner (`planner_node`):** Upon receiving a complex user query, this agent runs first. It uses a powerful reasoning LLM (`gemini-pro-latest`) with a "few-shot" prompt (an included example) to break the user's goal into a numbered checklist of specific, tool-oriented actions. This plan is the agent's roadmap. For simple conversational queries, it correctly creates an empty plan.
2.  **The Controller/Executor (`controller_node`):** This is the central decision-maker. It has three distinct behaviors based on the current state:
    * **Conversational Reply:** If the plan is empty from the start (no tools needed), it provides a direct, friendly answer.
    * **Execute Next Step:** If the plan is active, it reads the next step and chooses the best tool (`InformationTool`, `MathTool`, etc.) to accomplish it.
    * **Synthesize Final Answer:** Once all plan steps are complete, it reviews the collected results (`past_steps`) and generates a single, comprehensive final answer.
3.  **The Tool Worker (`execute_tool_node`):** This node is the "hands" of the operation. When the controller chooses a tool, this node executes it, handles errors, and reports the result back to the state, ticking off the completed step from the plan.
4.  **Persistent Memory:** LangGraph's `SqliteSaver` is used as a checkpointer. The entire state of the conversation (messages, plan, past steps) is automatically saved to `swarm_memory.sqlite` after every step. This provides robust, long-term memory for the agent.

---

## 📂 File Structure

```
.
├── .env                 # Local file for API keys (ignored by Git)
├── .gitignore           # Specifies files for Git to ignore
├── final_swarm.py       # Main application: The Agent Swarm
├── information_model.py # Defines the RAG chain for the InformationTool
├── report.pdf           # The source financial document
├── requirements.txt     # List of Python dependencies
├── run_task1.py         # Script to create the FAISS vector store
└── README.md            # This file
```

---

## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

2.  **Create and Activate a Virtual Environment**
    ```bash
    # Create the environment
    python3 -m venv venv
    # Activate it (macOS/Linux)
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Your API Key**
    * Create a file named `.env` in the project root.
    * Add your Google AI Studio API key to this file:
        ```
        GOOGLE_API_KEY="YOUR_API_KEY_HERE"
        ```

---

## 🏃 Running the Application

The application runs in two stages.

#### Stage 1: Create the Vector Database

First, you need to process the `report.pdf` into a FAISS index. Run this command in your terminal:

```bash
python3 run_task1.py
```
This will create a `faiss_index_multimodal` folder in your directory. You only need to do this once unless your `report.pdf` changes.

#### Stage 2: Run the Agent Swarm

Now you can start the interactive agent.

```bash
# IMPORTANT: Set this environment variable if you are on macOS to prevent a common crash
export KMP_DUPLICATE_LIB_OK=TRUE

# Run the main agent
python3 final_swarm.py
```

The agent will load, and you can start asking it questions. A `swarm_memory.sqlite` file will be created to store your conversation.

**Example Questions:**
* **Simple:** `Hello, what can you do?`
* **Specific:** `What is the total debt listed in the document?`
* **Complex:** `First, find the total assets and total liabilities from the report. Then, calculate the debt-to-asset ratio.`

---

## 🛠️ Tools and Technologies

* **Orchestration:** LangChain & LangGraph
* **LLMs:** Google Generative AI (Gemini Pro, Gemini Flash)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **PDF Parsing:** `unstructured.io` & `PyMuPDF`
* **Embeddings:** Google `embedding-001`
* **Tools:** Tavily Search, Python REPL

---

## 📄 License

This project is licensed under the MIT License.