import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from utils.llm_utils import get_ollama

llm = get_ollama()

curr_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(curr_dir)

books_dir = os.path.join(parent_dir, "books")
persistent_directory = os.path.join(parent_dir, "db", "chroma_db_with_metadata_hg")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_query_response(query: str, db: Chroma):
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

    # Step 3: Ask the LLM to give a one-liner answer
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


def create_chroma_db() -> Chroma:
    if not os.path.exists(persistent_directory):
        print("Persistent directory does not exist. Initializing vector store...")

        # Ensure the text book_file exists
        if not os.path.exists(books_dir):
            raise FileNotFoundError(
                f"The folder {books_dir} does not exist. Please check the path."
            )

        book_files = [f for f in os.listdir(books_dir) if f.endswith(".txt")]

        documents = []
        for book_file in book_files:
            file_path = os.path.join(books_dir, book_file)
            loader = TextLoader(file_path)
            book_docs = loader.load()

            for i, doc in enumerate(book_docs):
                doc.metadata = {"source": file_path, "book_file": book_file, "iteration": i}
                documents.append(doc)

        # Split the documents into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(documents)}")
        print(f"Sample chunk:\n{documents[0].page_content}\n")

        # Create the vector store and persist it automatically
        print("\n--- Creating vector store ---")
        db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
        print("\n--- Finished creating vector store ---")
        return db
    else:
        print("Vector store already exists. No need to initialize.")
        return Chroma(persist_directory=persistent_directory,
                      embedding_function=embeddings)


chroma_db = create_chroma_db()
get_query_response("Who is the Author of Adventures of Sherlock Holmes?", chroma_db)
get_query_response("Who is the Ulysses wife name?", chroma_db)
