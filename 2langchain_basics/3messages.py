from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils.llm_utils import get_llm
from utils.response_utils import get_prompt_response


def main():
    messages = [
        SystemMessage("You are a Bank Manager."),
        HumanMessage("What is the home loan interest rate?"),
        AIMessage("It is 8.0%"),
        HumanMessage("How much is my interest on 88 lakh rupees. What will be the EMI for a duration of 20 years?"
                     "Just provide only EMI. Nothing else.")
    ]

    llm = get_llm()
    res = llm.invoke(messages)

    print(res)


if __name__ == "__main__":
    main()