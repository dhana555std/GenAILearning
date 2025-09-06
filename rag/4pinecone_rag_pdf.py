import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
    Docx2txtLoader,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from utils.llm_utils import get_google_genai, get_ollama

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "resumes-rag"

# Initialize LLM and Embeddings
llm = get_google_genai()
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")  # 768 dimensions

# Paths
curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
books_dir = os.path.join(parent_dir, "resumes")


def create_pinecone_db():
    """Loads documents, chunks them, and stores/upserts into Pinecone."""
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Ensure index exists with the right dimension
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            dimension=768,  # Google embedding dimension
            metric="cosine"
        )

    desc = pc.describe_index(INDEX_NAME)
    print(f"Pinecone index dimension: {desc.dimension}")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"The folder {books_dir} does not exist.")

    # Pick only pdf/doc/docx files
    book_files = [f for f in os.listdir(books_dir) if f.lower().endswith((".pdf", ".docx", ".doc"))]

    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)

        # Load based on type
        if book_file.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                book_docs = loader.load()
                print(f"📄 Loaded {len(book_docs)} pages from {book_file} using PyPDFLoader")
                if not book_docs:
                    loader = UnstructuredPDFLoader(file_path)
                    book_docs = loader.load()
            except Exception as e:
                print(f"❌ Error loading {book_file}: {e}")
                continue
        elif book_file.lower().endswith((".doc", ".docx")):
            loader = Docx2txtLoader(file_path)
            book_docs = loader.load()
            print(f"📄 Loaded {len(book_docs)} sections from {book_file}")
        else:
            continue

        # Add metadata
        for i, doc in enumerate(book_docs):
            doc.metadata = {
                "location": "Bangaluru",
                "filename": book_file,
                "timestamp": get_timestamp()
            }
            documents.append(doc)

    # Split into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print(f"\n--- Total Chunks to Insert: {len(docs)} ---")

    # Store in Pinecone
    PineconeVectorStore.from_documents(docs, embeddings, index_name=INDEX_NAME)
    print("\n✅ Finished inserting documents into Pinecone ---")


def get_timestamp():
    utc_timestamp = datetime.now(timezone.utc).timestamp()
    dt_obj = datetime.fromtimestamp(utc_timestamp, tz=timezone.utc)
    return dt_obj.strftime("%Y-%m-%d %H:%M:%S UTC")


def filter_documents(docs, filters):
    """
    Filters documents dynamically based on metadata.

    Args:
        docs (list): List of LangChain Documents
        filters (dict): Dictionary of filtering conditions.
            Example:
            {
                "name": {"contains": "umesh"},
                "location": {"equals": "bangalore"},
                "role": {"startswith": "senior"},
                "email": {"endswith": "@gmail.com"}
            }

    Returns:
        list: Filtered documents
    """
    filtered_docs = []
    for doc in docs:
        for key, condition in filters.items():
            value = str(doc.metadata.get(key, "")).lower()

            if "contains" in condition:
                if condition["contains"].lower() not in value:
                    break
            elif "equals" in condition:
                if value != condition["equals"].lower():
                    break
            elif "startswith" in condition:
                if not value.startswith(condition["startswith"].lower()):
                    break
            elif "endswith" in condition:
                if not value.endswith(condition["endswith"].lower()):
                    break
            filtered_docs.append(doc)

    return filtered_docs


def get_query_response(query: str, criteria: dict = None):
    """Retrieve relevant docs from Pinecone and answer with Google GenAI LLM."""
    db = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.1},
    )

    relevant_docs = retriever.invoke(query)

    # Apply metadata filtering if filters provided
    if criteria:
        relevant_docs = filter_documents(relevant_docs, criteria)

    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs):
        print(f"📑 Document {i}:\n{doc.page_content[:300]}...\n")
        print(f"Metadata: {doc.metadata}\n")

    context = "\n".join([doc.page_content for doc in relevant_docs])

    template = """Answer the following question in ONE sentence only. If the context is not available say I don't know.

    Context:
    {context}

    Question: {question}

    Answer:"""

    prompt = PromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})

    print("\n--- Final Answer ---")
    print(answer)
    return answer


if __name__ == "__main__":
    create_pinecone_db()

    filters = {
        # "filename": {"contains": "Belge"},
        "location": {"equals": "Bangaluru"}
    }
    get_query_response("Who worked on Vue?", criteria=filters)
