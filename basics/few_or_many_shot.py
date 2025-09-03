from langchain_ollama.llms import OllamaLLM


def main():
    llm = OllamaLLM(model="llama3.2")

    prompt = """You are an expert Financial Risk Analyst.  
Your task is to classify loan applications into one of three categories: Low Risk, Moderate Risk, or High Risk.  

The decision is based on:  
- Age of the applicant  
- Monthly salary  
- Loan amount requested  
- Credit score (CIBIL score)  

Guard Rules:  
1. Provide only the risk category as the output: Low Risk, Moderate Risk, or High Risk.  
2. Do not include any explanation or additional text.  
3. Follow the examples given below for consistency.  

Examples:  

Applicant: Age 45, Salary: ₹80,000/month, Credit Score: 750, Loan Amount: ₹10,00,000  
Risk: Low Risk  

Applicant: Age 25, Salary: ₹30,000/month, Credit Score: 600, Loan Amount: ₹7,00,000  
Risk: High Risk  

Applicant: Age 38, Salary: ₹1,50,000/month, Credit Score: 720, Loan Amount: ₹12,00,000  
Risk: Moderate Risk  

Applicant: Age 50, Salary: ₹60,000/month, Credit Score: 770, Loan Amount: ₹5,00,000  
Risk: Low Risk  

Applicant: Age 29, Salary: ₹40,000/month, Credit Score: 650, Loan Amount: ₹6,00,000  
Risk: High Risk  

Applicant: Age 42, Salary: ₹2,00,000/month, Credit Score: 800, Loan Amount: ₹15,00,000  
Risk: Moderate Risk  

Applicant: Age 33, Salary: ₹1,20,000/month, Credit Score: 690, Loan Amount: ₹5,00,000  
Risk: Low Risk  

Applicant: Age 27, Salary: ₹55,000/month, Credit Score: 640, Loan Amount: ₹9,00,000  
Risk: High Risk  

Applicant: Age 36, Salary: ₹90,000/month, Credit Score: 710, Loan Amount: ₹7,00,000  
Risk: Moderate Risk  

Applicant: Age 48, Salary: ₹1,00,000/month, Credit Score: 780, Loan Amount: ₹8,00,000  
Risk: Low Risk  

---  

Now classify the following applicant:  
Applicant: Age 40, Salary: ₹70,000/month, Credit Score: 700, Loan Amount: ₹39,00,000  
Risk:

"""

    poem = llm.invoke(prompt)

    print("Response:\n")
    print(poem)


if __name__ == "__main__":
    main()
