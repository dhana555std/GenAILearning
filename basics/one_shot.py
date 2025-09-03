from langchain_ollama.llms import OllamaLLM


def main():
    llm = OllamaLLM(model="llama3.2:latest")

    prompt = """
    Categorize the following into one of these categories: Work or Personal.
    Example:-
    Input: "Finish the quarterly report"
    Output: Work
    
    Now do for the following input:-
    Input: "Plan a trip with family"
    
    Guard rails:-
    Provided just the category name as output, either Work or Personal. Don't write anything else.
    Strictly Work or Personal.
    """

    poem = llm.invoke(prompt)

    print("Response:\n")
    print(poem)


if __name__ == "__main__":
    main()
