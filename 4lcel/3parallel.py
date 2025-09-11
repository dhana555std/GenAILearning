from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.llm_utils import get_google_genai, get_ollama
from langchain_core.runnables import RunnableParallel


def main():
    gemini = get_google_genai()
    ollama = get_ollama()
    parser = StrOutputParser()

    notes_template = "Write a 10 pointer notes on the topic {topic}"
    quiz_template = "Create 5 MCQ on the {topic}. Also provide the answers in the end."
    merge_template = "Merge both {notes} and {quiz}"

    prompt_template_notes = PromptTemplate.from_template(template=notes_template)
    prompt_template_quiz = PromptTemplate.from_template(template=quiz_template)

    parallel_chain = RunnableParallel({
        "notes": prompt_template_notes | gemini | parser,
        "quiz": prompt_template_quiz | gemini | parser
    })

    prompt_template_merge = PromptTemplate.from_template(template=merge_template)
    merge_chain = prompt_template_merge | gemini | parser
    chain = parallel_chain | merge_chain
    res = chain.invoke({"topic": "Low code tools"})
    print(res)

    chain.get_graph().print_ascii()


if __name__ == "__main__":
    main()
