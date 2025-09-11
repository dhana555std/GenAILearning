from langchain_ollama.llms import OllamaLLM


def main():
    prompt = """You are an expert financial risk analyst.  
Your task is to classify loan applications into one of three categories: Low Risk, Moderate Risk, or High Risk.

Decision factors (in order of priority):
1. Credit score (CIBIL) — highest priority. 
2. Monthly salary — second priority.
3. Loan amount requested — third priority.
4. Age of the applicant — fourth priority.

Instruction:
Reason internally about the factors (with credit score given highest weight, then salary, then loan amount, then age). Use the provided examples as guidance. When classifying a new applicant, output **only** the final risk category on a single line: `Low Risk`, `Moderate Risk`, or `High Risk`. Do **not** output step-by-step chain-of-thought. The examples below include short, one-line rationales to illustrate the decision patterns—these rationales are for guidance only.

Guard Rules:
1. Output must be exactly one of: `Low Risk`, `Moderate Risk`, or `High Risk`.Also provide Rationale.
2. Do not include any explanation, commentary, or extra text in the output.
3. Follow the examples and their concise rationales for consistency.

Examples (with concise rationale for guidance):

Applicant: Age 45, Salary: ₹80,000/month, Credit Score: 750, Loan Amount: ₹10,00,000  
Risk: Low Risk  
Rationale: Strong CIBIL (750) and sufficient salary relative to loan → low risk.

Applicant: Age 25, Salary: ₹30,000/month, Credit Score: 600, Loan Amount: ₹7,00,000  
Risk: High Risk  
Rationale: Low CIBIL (600) and high loan relative to income → high risk.

Applicant: Age 38, Salary: ₹1,50,000/month, Credit Score: 720, Loan Amount: ₹12,00,000  
Risk: Moderate Risk  
Rationale: Good CIBIL (720) but large loan amount → moderate risk.

Applicant: Age 50, Salary: ₹60,000/month, Credit Score: 770, Loan Amount: ₹5,00,000  
Risk: Low Risk  
Rationale: High CIBIL (770) and reasonable loan relative to salary → low risk.

Applicant: Age 29, Salary: ₹40,000/month, Credit Score: 650, Loan Amount: ₹6,00,000  
Risk: High Risk  
Rationale: Borderline CIBIL (650) with sizable loan → high risk.

Applicant: Age 42, Salary: ₹2,00,000/month, Credit Score: 800, Loan Amount: ₹15,00,000  
Risk: Moderate Risk  
Rationale: Excellent CIBIL (800) but very large loan amount → moderate risk.

Applicant: Age 33, Salary: ₹1,20,000/month, Credit Score: 690, Loan Amount: ₹5,00,000  
Risk: Low Risk  
Rationale: Good salary and reasonable loan; CIBIL slightly below 700 but acceptable → low risk.

Applicant: Age 27, Salary: ₹55,000/month, Credit Score: 640, Loan Amount: ₹9,00,000  
Risk: High Risk  
Rationale: Low CIBIL (640) and high loan relative to income → high risk.

Applicant: Age 36, Salary: ₹90,000/month, Credit Score: 710, Loan Amount: ₹7,00,000  
Risk: Moderate Risk  
Rationale: CIBIL just above 700 but loan sizable → moderate risk.

Applicant: Age 48, Salary: ₹1,00,000/month, Credit Score: 780, Loan Amount: ₹8,00,000  
Risk: Low Risk  
Rationale: High CIBIL and strong salary relative to loan → low risk.

---

Now classify the following applicant (apply the decision priorities above; output only the final category):

Applicant: Age 40, Salary: ₹70,000/month, Credit Score: 700, Loan Amount: ₹72,00,0000  
Risk:
"""

    llm = OllamaLLM(model="llama3.2:latest")
    response = llm.invoke(prompt)

    print("Response:\n")
    print(response)


if __name__ == "__main__":
    main()
