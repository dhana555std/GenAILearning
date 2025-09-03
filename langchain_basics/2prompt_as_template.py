from langchain_core.prompts import PromptTemplate

from utils.llm_utils import get_llm


def main():
    prompt_template_text = "Write {count} one liner joke(s) on {topic}."
    prompt_template = PromptTemplate(template=prompt_template_text,
                                     input_variables=['count', 'topic'])
    final_prompt = prompt_template.format(count=2, topic="Judges")

    res = get_llm().invoke(final_prompt).content
    print(res)


if __name__ == "__main__":
    main()
