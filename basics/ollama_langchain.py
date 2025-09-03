from langchain_ollama.llms import OllamaLLM


def main():
    llm = OllamaLLM(model="tinyllama")

    poem = llm.invoke("Write a short and inspiring poem with the theme: God is great!.")

    print("Generated Poem:\n")
    print(poem)


if __name__:
    main()
