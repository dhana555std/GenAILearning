from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import PyPDFLoader

from utils.llm_utils import get_google_genai


def main():
    llm = get_google_genai()

    # Load PDF as text
    loader = PyPDFLoader("../sample.pdf")
    pages = loader.load()
    file_text = "\n\n".join([page.page_content for page in pages])

    # Build prompt with extracted text
    question = "Tell me about Anti-Harassment based out of the document."
    message = HumanMessage(content=f"{question}\n\nDocument:\n{file_text}")

    response = llm.invoke([message])

    print("\n=== Result ===\n")
    print(response.content)


if __name__ == "__main__":
    main()
