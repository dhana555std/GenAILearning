from langchain_ollama import OllamaLLM

# Create Ollama LLM instance with temperature = 0
llm_low = OllamaLLM(model="llama3.2", temperature=0)

# Create Ollama LLM instance with temperature = 0.8
llm_mid = OllamaLLM(model="llama3.2", temperature=0.8)

# Create Ollama LLM instance with temperature = 1.2
llm_high = OllamaLLM(model="llama3.2", temperature=2.0)

prompt = "Write a very short, two-sentence story about a spaceship exploring a new planet."

# Generate responses
response_low = llm_low.invoke(prompt)
response_mid = llm_mid.invoke(prompt)
response_high = llm_high.invoke(prompt)


print("Low Temperature (0) Scientist:", response_low, "\n")
print("Low Temperature (0.8) Narrator:", response_mid, "\n")
print("High Temperature (2.0) Poet:", response_high)
