from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda

from utils.llm_utils import get_google_genai


def main():
    llm = get_google_genai()
    parser = StrOutputParser()

    feedback_template = """Categorize into positive or negative the following {feedback}. Make it positive or negative.
     No other values. No description. No more comments."""

    prompt_feedback = PromptTemplate.from_template(template=feedback_template)
    feedback_chain = prompt_feedback | llm | parser

    prompt_positive = PromptTemplate.from_template("Write a one liner positive response on {feedback}.")
    prompt_negative = PromptTemplate.from_template("Write a one liner negative response on {feedback}.")

    positive_chain = prompt_positive | llm | parser
    negative_chain = prompt_negative | llm | parser

    result_chain = RunnableBranch(
        (lambda x: x.lower() == 'positive', positive_chain),
        (lambda x: x.lower() == 'negative', negative_chain),
        RunnableLambda(lambda x: "could not find sentiment")
    )

    chain = feedback_chain | result_chain
    res = chain.invoke({"feedback": "Food is worst!"})
    print(res)


if __name__ == "__main__":
    main()
