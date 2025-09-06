"""
Reference https://docs.langchain.com/oss/python/langchain-tools
"""
from langchain_core.tools import tool


@tool("dhanam_value_calculation", description="This method takes an input and generates dhanam"
                                              "value for it.")
def dhanam_value(a: int) -> float:
    """
    This method is used to generate Dhanam value.
    :param a: input for which Dhanam value to be calculated.
    :return: Dhanam value for the input.
    """
    return a * 32 / 7


print(dhanam_value.name)
print(dhanam_value.description)
print(dhanam_value.schema)
print(dhanam_value.args)
