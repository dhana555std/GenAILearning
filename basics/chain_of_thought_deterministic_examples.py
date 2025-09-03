from langchain_ollama.llms import OllamaLLM


def main():
    prompt = """You are an expert Loan Manager who decides whether the Loan amount is risky or not.
    
## Decision Factors (priority order).
1. **CIBIL Score** (highest priority)  
   - < 600 → High Risk
   - 600–699 → Moderate Risk
   - ≥ 700 → Low Risk

2. **Loan Amount**  
   - < ₹6,00,000 → Low Risk   
   - ₹6,00,000 – ₹9,99,999 → Moderate Risk  
   - ≥ ₹10,00,000 → High Risk 

3. **Age**  
   - > 60 → High Risk
   - 50–59 → Moderate Risk  
   - < 50 → Low Risk
Each example shows:
- Applicant (Age, Loan Amount, CIBIL)
- Aggregation: per-factor classification with brief, clear reasoning
- Final Risk: 
  * If there is at least one High Risk factor, overall is High Risk.
  * If no Highs but at least one Moderate, overall is Moderate.
  * Only if all Low, overall is Low.

1) Applicant: Age 30, Loan Amount: ₹5,00,000, CIBIL: 580  
Aggregation:
- CIBIL = High Risk — CIBIL 580 < 600   
- Loan Amount = Low Risk — ₹5,00,000 < ₹6,00,000 
- Age = Low Risk — age 30 (<50) 
Final Risk: High Risk — any High → overall High.

2) Applicant: Age 50, Loan Amount: ₹5,00,000, CIBIL: 580  
Aggregation:
- CIBIL = High Risk - 580 < 600 High Risk
- Loan Amount = Low Risk - ₹5,00,000 < ₹6,00,000 
- Age = Moderate Risk — age 50 falls in 50–59.  
Final Risk: High Risk — any High → overall High.

3) Applicant: Age 65, Loan Amount: ₹5,00,000, CIBIL: 580  
Aggregation:
- CIBIL = High Risk —  580 < 600 High Risk
- Loan Amount = Low Risk — ₹5,00,000 < ₹6,00,000   
- Age = High Risk — age 65 > 60  
Final Risk: High Risk — multiple High factors; overall High.

4) Applicant: Age 30, Loan Amount: ₹7,00,000, CIBIL: 580  
Aggregation:
- CIBIL = High Risk — 580 < 600 
- Loan Amount = Moderate Risk — ₹7,00,000 falls in ₹6L–₹9.99L 
- Age = Low Risk — age 30 < 50
Final Risk: High Risk — CIBIL High overrides others.

5) Applicant: Age 50, Loan Amount: ₹7,00,000, CIBIL: 580  
Aggregation:
- CIBIL = High Risk 580 < 600  
- Loan Amount = Moderate Risk — ₹7L in mid-range.
- Age = Moderate Risk — age 50 adds moderate concern.  
Final Risk: High Risk — presence of any High → overall High.

6) Applicant: Age 65, Loan Amount: ₹7,00,000, CIBIL: 580  
Aggregation:
- CIBIL = High Risk — 580 < 600  
- Loan Amount = Moderate Risk — ₹7L is mid-range loan. 
- Age = High Risk — >60 
Final Risk: High Risk — High factors present.

7) Applicant: Age 30, Loan Amount: ₹15,00,000, CIBIL: 580  
Aggregation:
- CIBIL = High Risk — 580 < 600
- Loan Amount = High Risk — ₹15L ≥ ₹10L 
- Age = Low Risk - age 30 < 50 
Final Risk: High Risk — multiple High factors.

8) Applicant: Age 50, Loan Amount: ₹15,00,000, CIBIL: 580  
Aggregation:
- CIBIL = High Risk — 580 < 600 
- Loan Amount = High Risk — ₹15L ≥ ₹10L 
- Age = Moderate Risk — age in 50 - 59 bracket.
Final Risk: High Risk — High dominates.

9) Applicant: Age 65, Loan Amount: ₹15,00,000, CIBIL: 580  
Aggregation:
- CIBIL = High Risk — 580 < 600  
- Loan Amount = High Risk — ₹15L ≥ ₹10L
- Age = High Risk — age > 60   
Final Risk: High Risk — unanimous High.

10) Applicant: Age 30, Loan Amount: ₹5,00,000, CIBIL: 650  
Aggregation:
- CIBIL = Moderate Risk — 650 in 600–699 band.  
- Loan Amount = Low Risk — ₹5,00,000 < ₹6,00,000  
- Age = Low Risk — age 30 < 50
Final Risk: Moderate Risk — no Highs, at least one Moderate → Moderate.

11) Applicant: Age 50, Loan Amount: ₹5,00,000, CIBIL: 650  
Aggregation:
- CIBIL = Moderate Risk — 650 in 600–699 range.
- Loan Amount = Low Risk — ₹5,00,000 < ₹6,00,000 
- Age = Moderate Risk — age 50 → 50-59 range.  
Final Risk: Moderate Risk — no Highs, Moderates present → Moderate.

12) Applicant: Age 65, Loan Amount: ₹5,00,000, CIBIL: 650  
Aggregation:
- CIBIL = Moderate Risk — 650 in 600–699 range.
- Loan Amount = Low Risk — 5L < 6L  
- Age = High Risk — age > 60 
Final Risk: High Risk — any High → High.

13) Applicant: Age 30, Loan Amount: ₹7,00,000, CIBIL: 650  
Aggregation:
- CIBIL = Moderate Risk — 650 in 600–699 range.  
- Loan Amount = Moderate Risk — ₹7,00,000 in ₹6L–₹9.99L band.
- Age = Low Risk — age 30 < 50  
Final Risk: Moderate Risk — no Highs; Moderates → Moderate.

14) Applicant: Age 50, Loan Amount: ₹7,00,000, CIBIL: 650  
Aggregation:
- CIBIL = Moderate Risk — 650 in 600–699 range.
- Loan Amount = Moderate Risk — ₹7L < ₹10L
- Age = Moderate Risk — 50 in 50–59 range.  
Final Risk: Moderate Risk — multiple Moderates but no Highs → Moderate.

15) Applicant: Age 65, Loan Amount: ₹7,00,000, CIBIL: 650  
Aggregation:
- CIBIL = Moderate Risk — 650 in 600–699 range.
- Loan Amount = Moderate Risk — ₹7L < ₹10L  
- Age = High Risk — age >60 increases risk significantly.  
Final Risk: High Risk — High dominates as there is one High risk.

16) Applicant: Age 30, Loan Amount: ₹15,00,000, CIBIL: 650  
Aggregation:
- CIBIL = Moderate Risk — 650 in 600–699 range.  
- Loan Amount = High Risk — loan ≥ ₹10L  
- Age = Low Risk — 30 < 50.  
Final Risk: High Risk — loan High forces overall High.

17) Applicant: Age 50, Loan Amount: ₹15,00,000, CIBIL: 650  
Aggregation:
- CIBIL = Moderate Risk — 650 in 600–699 range.  
- Loan Amount = High Risk — 15L ≥ 10L 
- Age = Moderate Risk — 50 < 60   
Final Risk: High Risk — presence of High (loan) → High.

18) Applicant: Age 65, Loan Amount: ₹15,00,000, CIBIL: 650  
Aggregation:
- CIBIL = Moderate Risk — 650 in 600–699 range.  
- Loan Amount = High Risk — 15L ≥ 10L 
- Age = High Risk — age(60) >60  
Final Risk: High Risk — multiple Highs and Moderates → High.

19) Applicant: Age 30, Loan Amount: ₹5,00,000, CIBIL: 720  
Aggregation:
- CIBIL = Low Risk — 720 ≥ 700 
- Loan Amount = Low Risk — ₹5,00,000 < ₹6,00,000 
- Age = Low Risk — age < 50  
Final Risk: Low Risk — all Low → Low.

20) Applicant: Age 50, Loan Amount: ₹5,00,000, CIBIL: 720  
Aggregation:
- CIBIL = Low Risk — 720 ≥ 700 
- Loan Amount = Low Risk — ₹5,00,000 < ₹6,00,000
- Age = Moderate Risk — age in 50–59 range.  
Final Risk: Moderate Risk — no Highs, at least one Moderate → Moderate.

21) Applicant: Age 65, Loan Amount: ₹5,00,000, CIBIL: 720  
Aggregation:
- CIBIL = Low Risk — 720 ≥ 700  
- Loan Amount = Low Risk — 5L < 6L 
- Age = High Risk — age > 60  
Final Risk: High Risk — Age High → overall High.

22) Applicant: Age 30, Loan Amount: ₹7,00,000, CIBIL: 720  
Aggregation:
- CIBIL = Low Risk — 720 ≥ 700  
- Loan Amount = Moderate Risk — ₹7,00,000 in ₹6L–₹9.99L range. 
- Age = Low Risk — age 30 < 50 
Final Risk: Moderate Risk — moderate loan causes overall Moderate (no Highs).

23) Applicant: Age 50, Loan Amount: ₹7,00,000, CIBIL: 720  
Aggregation:
- CIBIL = Low Risk — 720 ≥ 700
- Loan Amount = Moderate Risk — ₹7L in 6L–9.99L range.  
- Age = Moderate Risk — age 50 in 50–59 range. 
Final Risk: Moderate Risk — Moderates present; no Highs → Moderate.

24) Applicant: Age 65, Loan Amount: ₹7,00,000, CIBIL: 720  
Aggregation:
- CIBIL = Low Risk — 720 ≥ 700  
- Loan Amount = Moderate Risk - ₹7L in 6L–9.99L range.  
- Age = High Risk — age > 60  
Final Risk: High Risk —  High dominates.

25) Applicant: Age 30, Loan Amount: ₹15,00,000, CIBIL: 720  
Aggregation:
- CIBIL = Low Risk — 720 ≥ 700
- Loan Amount = High Risk — ₹15L ≥ ₹10L 
- Age = Low Risk — age 30 < 50 
Final Risk: High Risk — loan High → overall High.

26) Applicant: Age 50, Loan Amount: ₹15,00,000, CIBIL: 720  
Aggregation:
- CIBIL = Low Risk — 720 ≥ 700  
- Loan Amount = High Risk — ₹15L ≥ ₹10L  
- Age = Moderate Risk — age 50 in 50–59 range. 
Final Risk: High Risk — High (loan) takes precedence.

27) Applicant: Age 65, Loan Amount: ₹15,00,000, CIBIL: 720  
Aggregation:
- CIBIL = Low Risk — 720 ≥ 700 
- Loan Amount = High Risk — ₹15L ≥ ₹10L  
- Age = High Risk — age > 60 
Final Risk: High Risk — High factors present (loan, age) → High.

Now classify the following applicant (apply the decision priorities above; output only the final category):
Applicant: Age 20, Salary: ₹70,000/month, Credit Score: 700, Loan Amount: 500000

"""

    llm = OllamaLLM(model="llama3.2:3b")
    response = llm.invoke(prompt)

    print("Response:\n")
    print(response)


if __name__ == "__main__":
    main()
