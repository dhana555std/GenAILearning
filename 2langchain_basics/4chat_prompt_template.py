from utils.llm_utils import get_llm
from langchain_core.prompts import ChatPromptTemplate


def main():
    messages = [
        ("system", "You are a Bank Manager."),
        ("human", "What is the home loan interest rate?"),
        ("assistant", "It is {loan_rate}"),
        ("human", "How much is my emi for {principal} and duration of {duration} years."),
    ]

    chat_prompt = ChatPromptTemplate.from_messages(messages)

    # Format messages with values
    formatted_messages = chat_prompt.format_messages(
        loan_rate="8.5% per annum",
        principal="88 lakhs",
        duration="20"
    )

    llm = get_llm()
    res = llm.invoke(formatted_messages).content

    print(res)


if __name__ == "__main__":
    main()