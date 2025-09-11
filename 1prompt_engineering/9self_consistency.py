from langchain_ollama.llms import OllamaLLM


def main():
    llm = OllamaLLM(model="llama3.2:latest")

    prompt = """
    You are an expert financial advisor.  
Your task is to recommend the most suitable policy for an applicant based on:  
- Age of the applicant  
- Interest rate of the policy  
- Risk level of the policy  

**Decision Rules:**  
1. If the applicant’s age > 60 → avoid policies with High Risk, prefer safer policies.  
2. If the applicant’s age < 60 → prefer policies with higher interest rates, even if they are riskier.  
3. Final choice should balance age, risk, and interest, following the above rules.  
4. Use **self-consistency**: Generate multiple reasoning paths, then pick the most consistent and logical outcome as
   the final recommendation.  

---

### Example Case 1: Senior Applicant (Age 65)  

Applicant: Age 65  
Policy A: Interest 12%, Risk = High  
Policy B: Interest 8%, Risk = Moderate  
Policy C: Interest 6%, Risk = Low  

**Reasoning Path 1 (Risk-Focused):**  
- Age > 60 → avoid high risk. Eliminate Policy A.  
- Compare B (Moderate) vs C (Low).  
- C is safer, but B balances return and safety better.  

**Reasoning Path 2 (Balanced Approach):**  
- Eliminate A due to High Risk.  
- Policy B has moderate risk and decent returns.  
- Policy B is optimal.  

**Reasoning Path 3 (Return-Oriented):**  
- Even though age > 60, moderate risk is tolerable.  
- Policy B provides better return than C.  

**Final Consistent Outcome:**  
Policy **B** is recommended.  

---

### Example Case 2: Mid-Age Applicant (Age 40)  

Applicant: Age 40  
Policy A: Interest 12%, Risk = High  
Policy B: Interest 8%, Risk = Moderate  
Policy C: Interest 6%, Risk = Low  

**Reasoning Path 1 (Return-Focused):**  
- Age < 60, risk tolerance is higher.  
- Policy A (12%, High Risk) is most attractive.  

**Reasoning Path 2 (Balanced Approach):**  
- Risk is tolerable but avoid reckless choices.  
- Policy B is safer but slightly less rewarding.  

**Reasoning Path 3 (Aggressive):**  
- Mid-age, can tolerate High Risk for higher rewards.  
- Policy A is preferred.  

**Final Consistent Outcome:**  
Policy **A** is recommended.  

---

### Example Case 3: Young Applicant (Age 25)  

Applicant: Age 25  
Policy A: Interest 12%, Risk = High  
Policy B: Interest 8%, Risk = Moderate  
Policy C: Interest 6%, Risk = Low  

**Reasoning Path 1 (Aggressive):**  
- Very young → maximum time to recover from risks.  
- Policy A is most suitable.  

**Reasoning Path 2 (Return-Focused):**  
- Prioritize higher interest rates over safety.  
- Choose Policy A.  

**Reasoning Path 3 (Balanced):**  
- Even if risk is high, at 25 it is tolerable.  
- Policy A is optimal.  

**Final Consistent Outcome:**  
Policy **A** is recommended.  

---

### Now classify this new case:  

Applicant: Age 55  
Policy A: Interest 11%, Risk = High  
Policy B: Interest 9%, Risk = Moderate  
Policy C: Interest 7%, Risk = Low 

Provide Final Recommendation based on this.

Guard Rails:-
Provide a proper reasoning path for each attempt.

    """

    response = llm.invoke(prompt)

    print("Response:\n")
    print(response)


if __name__ == "__main__":
    main()
