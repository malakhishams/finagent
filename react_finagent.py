# IMPORT LIBRARIES

from langchain_community.document_loaders import TextLoader
from settings import GEMINI_API_KEY, GEMINI_MODEL
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from  langchain_community.vectorstores import Chroma
from langchain_core.tools.retriever import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# RAG PIPELINE: Load → Chunk → Embed → Store → Retrieve → Generate


# STEP1: Data ingestion and processing

loader = TextLoader("report.txt") # load the finanicial report text file
documents = loader.load() #load the doc into memory

# STEP2: Split the document into smaller chunks for better processing and embedding

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunked_docs = text_splitter.split_documents(documents)


#STEP3: Create embeddings for the chunked documents using a HuggingFace model and store them in a Chroma vector store

embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") # A lightweight model for generating embeddings, suitable for small datasets and quick retrieval.

vector_store = Chroma.from_documents(chunked_docs, embeddings_model)

def reset_vector_store():
    #Clears and re-initializes the Chroma vector store with fresh data
    global vector_store
    
    # Delete the existing collection
    vector_store.delete_collection()
    
    # Re-initialize with the same documents
    vector_store = Chroma.from_documents(chunked_docs, embeddings_model)
    print("Vector store reset and re-initialized successfully.")


# STEP4: Tool definition for retrieving chunks

retrieval_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(search_kwargs={"k":3}), # k=3 means the retriever will return the top 3 most relevant chunks from the vector store
    name="financial_report_retriever",
    description="Use this tool to retrieve relevant information from the financial report when needed."
)


# ReAct agent reasons in a loop: Thought → Action → Observation → Final Answer
# It autonomously decides when and how many times to call the retrieval tool

# STEP5: Configure ReAct agent

llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, api_key = GEMINI_API_KEY, temperature=0.1) # low temperature for more focused and deterministic responses, which is important for financial information retrieval.
FinAgent = create_react_agent(model=llm, tools=[retrieval_tool])


# STEP 6: Run complex multi-step query with ReAct reasoning trace

def probe_retrieval(query: str):
    print("\n==== SOURCE DOCUMENTS RETRIEVED ====")
    docs = vector_store.as_retriever(search_kwargs={"k": 3}).invoke(query)
    for i, doc in enumerate(docs):
        print(f"\n[Chunk {i+1}]:\n{doc.page_content}")
    print("========================================\n")

# Probe before running the agent
probe_retrieval("Q4 2024 profit margin and Q3 2024 total revenue")


query = """Using the financial report, answer both of these:
1. What was the Q4 2024 profit margin?
2. What was the Q3 2024 total revenue?
Then compare the two figures."""

print("\n" + "=="*30)
print("REACT AGENT REASONING TRACE")
print("=="*30 + "\n")


# langgraph agents use stream() instead of invoke() 
for step in FinAgent.stream(
    {"messages": [("human", query)]},
    stream_mode="values"
):
    message = step["messages"][-1]
      # Clean up Gemini's response format
    if hasattr(message, "content"):
        if isinstance(message.content, list):
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    print(f"AI: {block['text']}\n")
        else:
            message.pretty_print()
