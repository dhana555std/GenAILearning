from sklearn.model_selection import train_test_split
import pandas as pd

# Example dataset
data = pd.DataFrame({
    "Age": [25, 42, 33, 55, 29, 40, 60, 22, 35, 45],
    "Salary": [30000, 90000, 60000, 40000, 120000, 80000, 35000, 25000, 70000, 95000],
    "LoanAmount": [500000, 1000000, 700000, 800000, 1200000, 600000, 900000, 300000, 750000, 1100000],
    "CreditScore": [650, 750, 700, 580, 720, 690, 600, 500, 710, 770],
    "Approved": ["No", "Yes", "Yes", "No", "Yes", "Yes", "No", "No", "Yes", "Yes"]
})

# Split dataset with exactly 2 records in test set
train, test = train_test_split(data, test_size=2, random_state=None)

print("Training set:\n", train)
print("\nTest set:\n", test)
