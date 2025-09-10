import json
from typing import List
from pydantic import BaseModel, Field
from utils.llm_utils import get_ollama
from langchain.prompts import ChatPromptTemplate


class Address(BaseModel):
    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City name")
    country: str = Field(..., description="Country name")


class Employee(BaseModel):
    name: str = Field(..., description="The person's full name")
    experience: int = Field(..., description="Years of experience")
    skills: List[str] = Field(..., description="List of skills")
    address: Address = Field(..., description="Home address")


llm = get_ollama()

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. 
Extract employee details and return ONLY valid JSON matching this schema:
Do not add any unneceassary text either pre or post. Just created the end result.

{{
  "name": "string",
  "experience": int,
  "skills": ["string"],
  "address": {{
    "street": "string",
    "city": "string",
    "country": "string"
  }}
}}

Input:
{input}
""")

# LLM
llm = get_ollama()

# Run
chain = prompt | llm
response = chain.invoke({"input": "Alice is an AI Engineer from Bengaluru, India, "
                                  "lives on 123 Main St, knows Python and LangChain, "
                                  "and has 5 years of experience."})

# Parse JSON safely
try:
    data = json.loads(response)
    employee = Employee(**data)
    print(employee)
    print(employee.address.city)
except Exception as e:
    print("❌ Parsing failed:", e)
    print("Raw output:", response.content)
