import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from utils.llm_utils import get_ollama

# Load env vars
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "books-rag"

llm = get_ollama()

embeddings = HuggingFaceEmbeddings(model_name="nomic-ai/nomic-embed-text-v1",
                                   model_kwargs={"trust_remote_code": True})

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)
books_dir = os.path.join(parent_dir, "books")


def create_pinecone_db():
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Ensure index exists
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"Creating Pinecone index: {INDEX_NAME}")
        pc.create_index(
            name=INDEX_NAME,
            spec=ServerlessSpec(
                cloud="aws",  # or "gcp"
                region="us-east-1"  # pick your preferred region
            ),
            dimension=768,
            metric="cosine"
        )

    if not os.path.exists(books_dir):
        raise FileNotFoundError(f"The folder {books_dir} does not exist. Please check the path.")

    book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

    documents = []
    for book_file in book_files:
        file_path = os.path.join(books_dir, book_file)
        loader = TextLoader(file_path)
        book_docs = loader.load()

        for i, doc in enumerate(book_docs):
            doc.metadata = {"source": file_path, "book_file": book_file, "iteration": i}
            documents.append(doc)

    # Split into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print(f"\n--- Total Chunks to Insert: {len(docs)} ---")

    # Store in Pinecone
    PineconeVectorStore.from_documents(docs, embeddings, index_name=INDEX_NAME)
    print("\n--- Finished inserting documents into Pinecone ---")


def get_query_response(query: str):
    """Retrieve relevant docs from Pinecone and answer with Ollama."""
    db = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": 0.3},
    )

    relevant_docs = retriever.invoke(query)

    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

    context = "\n".join([doc.page_content for doc in relevant_docs])

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
    create_pinecone_db()
    get_query_response("Who is the Author of Adventures of Sherlock Holmes?")
    get_query_response("Who is the Release date of frankenstein")
