from langchain_ollama.llms import OllamaLLM


def main():
    llm = OllamaLLM(model="llama3.2:latest")
    prompt = "What is the capital of Andhra Pradesh?"
    response = llm.invoke(prompt)

    print("Response:\n")
    print(response)


if __name__ == "__main__":
    main()
