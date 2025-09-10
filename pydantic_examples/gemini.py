from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils.llm_utils import get_google_genai, get_ollama


# 1. Define the schema with Pydantic
class Person(BaseModel):
    name: str = Field(..., description="The person's full name")
    skills: list[str] = Field(..., description="List of skills the person has")
    experience: int = Field(..., description="Years of experience")


# 2. Create a parser
parser = PydanticOutputParser(pydantic_object=Person)

# 3. Build a prompt with formatting instructions
prompt = PromptTemplate(
    template="""
Extract information about the person from the text.

TEXT:
{resume_text}

{format_instructions}
""",
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 4. Connect LLM
llm = get_google_genai()

# 5. Build the chain
chain = prompt | llm | parser

# 6. Invoke
resume_text = """
John Doe is a Senior Software Engineer with 8 years of experience.
He is skilled in Python, Java, and Docker.
"""
result = chain.invoke({"resume_text": resume_text})

print("Parsed result:", result)
print("Name:", result.name)
print("Skills:", result.skills)
print("Experience:", result.experience)
