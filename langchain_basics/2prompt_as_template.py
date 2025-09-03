from langchain_core.prompts import PromptTemplate
from utils.llm_utils import get_llm


def main():
    prompt_template_str = "Write {count} one liner joke(s) on {topic}."
    prompt_template = PromptTemplate.from_template(prompt_template_str)
    prompt = prompt_template.format(
        count=2,
        topic="lawyers"
    )

    llm = get_llm()
    res = llm.invoke(prompt)
    print(f"response is {res}")


if __name__ == "__main__":
    main()
