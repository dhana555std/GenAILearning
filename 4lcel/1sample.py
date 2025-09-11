from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from utils.llm_utils import get_ollama


def main():
    prompt_template_text = "Tell {count} jokes on {topic}"
    prompt_template = PromptTemplate.from_template(template=prompt_template_text)

    llm = get_ollama()

    chain = prompt_template | llm | StrOutputParser()
    res = chain.invoke({"count": 2, "topic": "judges"})
    print(res)


if __name__ == "__main__":
    main()
