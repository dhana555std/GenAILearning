from langchain_ollama import OllamaLLM


def main():
    llm = OllamaLLM(model="llama3.2:latest", top_k=6, top_p=0.9, temperature=0.7)

    prompt = """
    Write a program to perform automation test on a REST API with url "https://api.example.com/data". Use Playwright.
    """

    response = llm.invoke(prompt)
    print(f"Response:\n {response}")


if __name__ == "__main__":
    main()
