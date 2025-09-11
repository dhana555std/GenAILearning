"""
Reference
1. http://docs.langchain.com/oss/python/langchain-agents
2. hwchase17/react (https://smith.langchain.com/hub/hwchase17/react?organizationId=529179d8-5092-5e66-a0f8-1a11e45e8d25)

"""
from typing import Dict, Any

from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
import requests

from utils.llm_utils import get_ollama


@tool("dhanam_value_calculation", description="This method takes two inputs and generates dhanam value for it.")
def dhanam_value(a: str, b: str) -> float:
    """Generate Dhanam value."""
    try:
        num1 = int(a)
        num2 = int(b)

    except ValueError:
        raise ValueError(f"Invalid input {a}. Expected an integer.")
    return num1 * 32 * num2 / 7


@tool(
    "get_employees_details_by_page",
    description=(
        "Retrieve employees from the organization (simulated using reqres.in API). "
        "If no page is provided, returns the first page. "
        "If a page number is provided, restricts results to that page."
    )
)
def get_employee_details(page: str = None) -> Dict[str, Any]:
    """Retrieve employees from the organization (simulated using reqres.in API)."""
    # Normalize page input
    try:
        print(f"page is {page} and {type(page)}.")
        curr_page = page.split('"')[1]
        # curr_page = page.split("'page':")[1].split("}")[0].strip(" '")
    except (ValueError, TypeError):
        curr_page = 1  # fallback default

    print(f"curr_page is {curr_page}.")
    url = f"https://reqres.in/api/users?page={curr_page}"

    # Add API key header
    headers = {
        "x-api-key": "reqres-free-v1"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json().get("data", [])


if __name__ == "__main__":
    llm = get_ollama()
    prompt = hub.pull("hwchase17/react")

    tools = [dhanam_value, get_employee_details]

    agent = create_react_agent(llm=llm,
                               tools=tools,
                               prompt=prompt)

    agent_executor = AgentExecutor(agent=agent,
                                   tools=tools,
                                   verbose=True,
                                   handle_parsing_errors=True,
                                   max_iterations=3,
                                   early_stopping_method="generate"
                                   )

    res = agent_executor.invoke({"input": "Get employee details from page 2"})
    print(f"result = {res['output']}")
