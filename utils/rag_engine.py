import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# OpenAI settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL")
OPENAI_EMBEDDING_MODEL_DIMENSIONS = os.getenv("OPENAI_EMBEDDING_MODEL_DIMENSIONS")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")

# Pinecone settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT_REGION = os.getenv("PINECONE_ENVIRONMENT_REGION", "us-east-1")
PINECONE_CLOUD_PROVIDER = os.getenv("PINECONE_CLOUD_PROVIDER", "aws")
PINECONE_NAMESPACE = "default"

# --- Initialization ---
# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model=OPENAI_EMBEDDING_MODEL,
    dimensions=OPENAI_EMBEDDING_MODEL_DIMENSIONS,
    openai_api_key=OPENAI_API_KEY
)

# Create index if it doesn't exist
# Modern way to check for index existence
if PINECONE_INDEX_NAME not in [index.name for index in pc.list_indexes().indexes]:
    print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating a new one...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=OPENAI_EMBEDDING_MODEL_DIMENSIONS,
        metric="cosine",  # 'cosine' is a good default for text embeddings
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD_PROVIDER,
            region=PINECONE_ENVIRONMENT_REGION
        )
    )
    print("Index created successfully.")

# Global vectorstore object to hold the connection
vectorstore = None


# --- Core Functions ---

def load_text_into_vectorstore(text: str):
    """
    Splits text, embeds it, and upserts it into a Pinecone index.
    This function now uses a more efficient batch method from LangChain.
    """
    global vectorstore

    print("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])  # create_documents is a convenient method

    print(f"Loading {len(docs)} document chunks into Pinecone...")
    # from_documents handles embedding and upserting in one step
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        namespace=PINECONE_NAMESPACE
    )
    print("Documents loaded successfully.")


def ask_question(query: str) -> str:
    """
    Asks a question to the RAG system using the modern LCEL chain.
    """
    global vectorstore

    # If no documents have been loaded, connect to the existing index
    if not vectorstore:
        print("No in-memory vectorstore found. Connecting to existing Pinecone index...")
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
            namespace=PINECONE_NAMESPACE
        )
        # Small check to ensure the index is not empty
        try:
            vectorstore.as_retriever(search_kwargs={'k': 1}).invoke("test")
        except Exception:
            return "Vectorstore is not initialized or is empty. Please load a document first."

    print("Constructing RAG chain and asking question...")

    llm = ChatOpenAI(
        temperature=0,
        model=OPENAI_CHAT_MODEL,
        api_key=OPENAI_API_KEY
    )

    retriever = vectorstore.as_retriever()

    template = """
    Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(query)
    return result
