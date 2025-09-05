import os
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

from utils.llm_utils import get_google_genai

# Load env vars
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "resumes-rag"

llm = get_google_genai()

# Google GenAI embeddings (4096-dim)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
books_dir = os.path.join(parent_dir, "resumes")


def create_pinecone_db():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            spec=ServerlessSpec(
                cloud="aws",  # or "gcp"
                region="us-east-1"
            ),
            dimension=768,  # embedding size for Google GenAI
            metric="cosine"
        )

    desc = pc.describe_index(INDEX_NAME)
    print(f"Pinecone index dimension: {desc.dimension}")

    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"The folder {books_dir} does not exist. Please check the path.")

    # Case-insensitive extension check
    book_files = [f for f in os.listdir(books_dir) if f.lower().endswith((".pdf", ".docx", ".doc"))]

    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)

        # --- Loader selection ---
        if book_file.lower().endswith(".pdf"):
            try:
                loader = PyPDFLoader(file_path)
                book_docs = loader.load()
                print(f"📄 Loaded {len(book_docs)} pages from {book_file} using PyPDFLoader")
                if not book_docs:
                    print(f"⚠️ No text in {book_file}, falling back to UnstructuredPDFLoader")
                    loader = UnstructuredPDFLoader(file_path)
                    book_docs = loader.load()
            except Exception as e:
                print(f"❌ Error loading {book_file} with PyPDFLoader: {e}")
                continue
        elif book_file.lower().endswith((".doc", ".docx")):
            loader = Docx2txtLoader(file_path)
            book_docs = loader.load()
            print(f"📄 Loaded {len(book_docs)} sections from {book_file}")
        else:
            continue

        # Add metadata
        for i, doc in enumerate(book_docs):
            doc.metadata = {"source": file_path, "book_file": book_file, "iteration": i}
            documents.append(doc)

    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    print(f"\n--- Total Chunks to Insert: {len(docs)} ---")

    # Store in Pinecone
    PineconeVectorStore.from_documents(docs, embeddings, index_name=INDEX_NAME)
    print("\n✅ Finished inserting documents into Pinecone ---")


def get_query_response(query: str):
    """Retrieve relevant docs from Pinecone and answer with Google GenAI LLM."""
    db = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.2},
    )

    relevant_docs = retriever.invoke(query)

    # ✅ Filter docs: only keep those whose metadata source contains "Sandhya"
    # filtered_docs = []
    # for doc in relevant_docs:
    #     source = doc.metadata.get("source", "").lower()
    #     if "umesh" in source:
    #         filtered_docs.append(doc)
    #
    # if not filtered_docs:
    #     print("\n⚠️ No documents matched filter (source contains 'Sandhya').")
    #     return "No relevant documents found for Sandhya."

    filtered_docs = relevant_docs

    print("\n--- Filtered Relevant Documents ---")
    for i, doc in enumerate(filtered_docs):
        print(f"📑 Document {i}:\n{doc.page_content[:300]}...\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

    context = "\n".join([doc.page_content for doc in filtered_docs])

    template = """Answer the following question in ONE sentence only.

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
    # create_pinecone_db()
    # get_query_response("Give me Sandya's phone and email")
    get_query_response("Provide list of Java developers")
