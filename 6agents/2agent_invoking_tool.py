"""
Reference
1. http://docs.langchain.com/oss/python/langchain-agents
2. hwchase17/react (https://smith.langchain.com/hub/hwchase17/react?organizationId=529179d8-5092-5e66-a0f8-1a11e45e8d25)

"""
from langchain_core.tools import tool
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

from utils.llm_utils import get_llm


@tool("dhanam_value_calculation", description="This method takes an input and generates dhanam value for it.")
def dhanam_value(a: str) -> float:
    """Generate Dhanam value."""
    try:
        num = int(a)
    except ValueError:
        raise ValueError(f"Invalid input {a}. Expected an integer.")
    return num * 32 / 7


if __name__ == "__main__":
    llm = get_llm()
    prompt = hub.pull("hwchase17/react")
    tools = [dhanam_value]

    agent = create_react_agent(llm=llm,
                               tools=tools,
                               prompt=prompt)

    agent_executor = AgentExecutor(agent=agent,
                                   tools=tools,
                                   verbose=True,
                                   handle_parsing_errors=True)

    res = agent_executor.invoke({"input": "What is the Dhanam Value of 9"})
    print(f"result = {res['output']}")
