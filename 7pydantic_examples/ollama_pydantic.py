from langchain_ollama import ChatOllama
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils.llm_utils import get_ollama


# Define schema
class Person(BaseModel):
    name: str = Field(..., description="The person's full name")
    skills: list[str] = Field(..., description="List of skills the person has")
    experience: int = Field(..., description="Years of experience")


# Create parser
parser = PydanticOutputParser(pydantic_object=Person)

# Initialize Ollama LLM
llm = get_ollama()

# Prompt that forces structured JSON
prompt = f"""
Extract the following fields and return ONLY valid JSON object, without schema or extra fields. 
Don't give any extra text. Either pre or post:
{parser.get_format_instructions()}

Text: "Alice has 5 years of Python and LangChain experience."
"""

resp = llm.invoke(prompt)

# Parse output
person = parser.parse(resp)
print(person)
