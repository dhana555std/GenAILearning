from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.llm_utils import get_google_genai, get_ollama
from langchain_core.runnables import RunnableLambda


def main():
    gemini = get_google_genai()
    ollama = get_ollama()

    capital_template = """What is the capital of {country}. Just mention the name.
    No extra words, sentences or descriptions"""

    visits_template = """Write an essay on {capital} in {count} bullet points.Use only bullet points.
        No pre or post text."""

    prompt_template_capital = PromptTemplate.from_template(template=capital_template)
    prompt_template_visits = PromptTemplate.from_template(template=visits_template)

    chain = (prompt_template_capital | ollama | StrOutputParser() |
             RunnableLambda(lambda capital: {"capital": capital, "count": 4})
             | prompt_template_visits | gemini | StrOutputParser())
    res = chain.invoke({"country": "Austria"})
    chain.get_graph().print_ascii()

    print(res)


if __name__ == "__main__":
    main()
