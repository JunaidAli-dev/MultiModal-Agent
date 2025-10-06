import os
import base64
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field # Ensure pydantic is imported correctly

# --- Configuration ---
FAISS_INDEX_PATH = "faiss_index_multimodal"
EMBEDDING_MODEL = "models/embedding-001"
LLM_VISION_MODEL = "models/gemini-2.5-flash-lite" 

def format_retrieved_documents(docs):
    """Prepares the context for the LLM, identifying if images need to be included."""
    context_parts = []
    text_context = "You are an expert financial analyst. Answer the user's question based on the following documents:\n\n"
    for doc in docs:
        text_context += f"---\nDocument Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page_number', 'N/A')}\n---\n"
        text_context += doc.page_content + "\n\n"
        if doc.metadata.get("type") == "image" and "base64_image" in doc.metadata:
            image_data = doc.metadata["base64_image"]
            context_parts.append({
                "type": "image_url",
                "image_url": {"mime_type": "image/jpeg", "data": image_data}
            })
    context_parts.insert(0, {"type": "text", "text": text_context})
    return context_parts

def create_information_agent():
    """Loads the vector store and creates a MULTIMODAL RAG chain."""
    print("Loading vector store for Information Agent...")
    
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={'k': 5})
    
    prompt_template = """
    Based on the context provided (which may include text and images), please answer the following question.
    If the context does not contain the answer, state that the information is not available in the provided documents.
    Question: {question}
    """
    
    llm_vision = ChatGoogleGenerativeAI(model=LLM_VISION_MODEL, temperature=0)

    # --- THIS IS THE CORRECTED FUNCTION ---
    def multimodal_rag_chain(question: str):
        # Step 1: Retrieve documents (this was already working)
        retrieved_docs = retriever.invoke(question)

        # Step 2: Format the documents for the LLM (the fix is here)
        # We now correctly assign the single return value to 'context_parts'
        context_parts = format_retrieved_documents(retrieved_docs)
        
        # Step 3: Combine the context with the prompt
        final_text_prompt = context_parts[0]['text'] + prompt_template.format(question=question)
        context_parts[0]['text'] = final_text_prompt

        # Step 4: Create the final message and invoke the LLM
        message = HumanMessage(content=context_parts)
        response = llm_vision.invoke([message])
        return response.content

    return RunnableLambda(multimodal_rag_chain)