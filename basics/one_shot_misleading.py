from langchain_ollama.llms import OllamaLLM


def main():
    llm = OllamaLLM(model="llama3.2")

    prompt = """
    You are a financial risk analyst. Classify loan applications as Low Risk or High Risk. 

    Example:  
    Applicant: Age 45, Salary: ₹80,000/month, Credit Score: 750, Loan Amount: ₹10,00,000  
    Risk: Low Risk  
    
    Now classify the following applicant:  
    Applicant: Age 23, Salary: ₹90,000/month, Credit Score: 720, Loan Amount: ₹1,00,000  
    Risk:
    
    Guard rails:
    Provide just the risk category as output, either Low Risk or High Risk. Don't write anything else.
    Strictly Low Risk or High Risk.

    """

    poem = llm.invoke(prompt)

    print("Response:\n")
    print(poem)


if __name__ == "__main__":
    main()
