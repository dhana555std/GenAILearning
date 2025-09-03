from utils.llm_utils import get_llm, get_google_genai, get_ollama
from utils.response_utils import get_prompt_response


def main():
    llm = get_llm()
    res = llm.invoke("What is the capital of India?\n")
    print(res)

    response = get_prompt_response(res)
    print(f"Response:\n{response}")

    print("-----------------------------")

    llm = get_google_genai()
    res = llm.invoke("God is great!")
    print(res.content)

    print("-----------------------------")

    llm = get_ollama()
    res = llm.invoke("God is great!")
    print(res)


if __name__ == "__main__":
    main()