import os

from pydantic import BaseModel, Field
from typing import List
from firecrawl import Firecrawl
from dotenv import load_dotenv


class Leader(BaseModel):
    name: str = Field(..., description="Leader name")
    designation: str = Field(..., description="Designation of the leader.")
    imageURL: str = Field(..., description="url of the Leader's image.")


class Leadership(BaseModel):
    top: List[Leader] = Field(..., description="Leaders details with name, designation and imageURL.")


load_dotenv()
firecrawl = Firecrawl(api_key=os.getenv("FIRECRAWL_API_KEY"))

# Use JSON format with a Pydantic schema
doc = firecrawl.scrape(
    "https://www.accionlabs.com/about-us#leadership",
    formats=[{"type": "json", "schema": Leadership}],
)
print(doc.json["top"])
