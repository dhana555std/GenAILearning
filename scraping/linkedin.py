import os

from pydantic import BaseModel, Field
from typing import List
from firecrawl import Firecrawl
from dotenv import load_dotenv


class Certification(BaseModel):
    name: str = Field(..., description="name of the certificate")
    issuedBy: str = Field(..., description="Board of issue")


class Language(BaseModel):
    name: str = Field(..., description="name of the language")
    issuedBy: str = Field(..., description="Proficiency level")


class Experience(BaseModel):
    name: str = Field(..., description="name of the company")
    designation: str = Field(..., description="designation during the tenure")
    duration: str = Field(..., description="druation/timeline during which period the candidate worked there. ")
    location: str = Field(..., description="work location during the tenure ")


class Profile(BaseModel):
    name: str = Field(..., description="Name of the person in the page.")
    about: str = Field(..., description="The about content.")
    tagline: str = Field(..., description="the tagline below name.")
    current_org:  str = Field(..., description="Current Organization")
    degree: str = Field(..., description="Highest Degree")
    experience: List[Experience] = Field(..., description="Experience details. Companies worked previously.")
    languages: List[Language] = Field(..., description="Languages.")
    certificates: List[Certification] = Field(..., description="Licenses & Certifications.")


load_dotenv()
firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

# Use JSON format with a Pydantic schema
doc = firecrawl.scrape(
    "https://in.linkedin.com/in/v-s-s-r-dhanapathi-marepalli-24716b92",
    formats=[{"type": "json", "schema": Profile}],
)
print(doc.json)
