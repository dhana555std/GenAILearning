from typing import List

from pydantic import BaseModel, Field

from utils.llm_utils import get_google_genai


class Address(BaseModel):
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country name")


class Employee(BaseModel):
    name: str = Field(..., description="The person's full name")
    experience: int = Field(..., description="Years of experience")
    skills: List[str] = Field(..., description="List of skills")
    address: Address = Field(..., description="Home address")


# Initialize Gemini model
llm = get_google_genai()

# Structured output with Employee schema
structured_llm = llm.with_structured_output(Employee)

result = structured_llm.invoke(
    "Alice is an AI Engineer from Bengaluru, India, lives on 123 Main St, "
    "knows Python and LangChain, and has 5 years of experience."
)

print(result)
print(result.address.city)   # Access nested field
