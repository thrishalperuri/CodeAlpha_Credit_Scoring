import os
import numpy as np
import pandas as pd

def generate_credit_dataset(n_samples=1500, random_state=42):
    """
    Generates a realistic credit scoring dataset with common financial features.
    """
    np.random.seed(random_state)
    
    # Feature 1: Age (18 to 75)
    age = np.random.randint(20, 70, size=n_samples)
    
    # Feature 2: Annual Income ($15,000 to $180,000)
    annual_income = np.random.exponential(scale=35000, size=n_samples) + 18000
    annual_income = np.clip(annual_income, 15000, 200000).round(-2)
    
    # Feature 3: Employment Years (0 to 40)
    employment_years = np.clip((age - 18) * np.random.uniform(0.1, 0.85, size=n_samples), 0, 40).round(1)
    
    # Feature 4: Existing Total Debt
    debt = annual_income * np.random.uniform(0.05, 0.95, size=n_samples)
    debt = debt.round(-2)
    
    # Feature 5: Debt to Income (DTI) Ratio
    dti_ratio = (debt / annual_income).round(3)
    
    # Feature 6: Number of Open Credit Lines
    open_credit_lines = np.random.poisson(lam=5, size=n_samples) + 1
    
    # Feature 7: Past Delinquencies (0 to 6)
    delinquencies = np.random.choice([0, 1, 2, 3, 4], size=n_samples, p=[0.72, 0.16, 0.07, 0.03, 0.02])
    
    # Feature 8: Loan Amount Requested
    loan_amount = np.random.uniform(2000, 45000, size=n_samples).round(-2)
    
    # Feature 9: Loan Purpose
    loan_purposes = np.random.choice(
        ['Debt Consolidation', 'Home Improvement', 'Education', 'Business', 'Personal', 'Auto'],
        size=n_samples,
        p=[0.35, 0.20, 0.15, 0.12, 0.10, 0.08]
    )
    
    # Feature 10: Credit Score (FICO range 300 to 850)
    base_score = 660 + (annual_income / 10000) * 12 - (delinquencies * 45) - (dti_ratio * 90) + (employment_years * 2.5)
    credit_score = np.clip(base_score + np.random.normal(0, 30, size=n_samples), 350, 850).astype(int)
    
    # Target Variable: Creditworthy (1 = Good Credit Risk / Approved, 0 = High Default Risk / Denied)
    risk_score = (
        (credit_score / 850) * 0.45 +
        ((annual_income - loan_amount) / 100000) * 0.25 -
        (dti_ratio * 0.20) -
        (delinquencies * 0.15)
    )
    # Binary classification with logistic probability threshold
    prob = 1 / (1 + np.exp(-4 * (risk_score - np.median(risk_score))))
    creditworthy = (np.random.uniform(0, 1, size=n_samples) < prob).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Annual_Income': annual_income,
        'Employment_Years': employment_years,
        'Total_Debt': debt,
        'DTI_Ratio': dti_ratio,
        'Open_Credit_Lines': open_credit_lines,
        'Delinquent_Accounts': delinquencies,
        'Credit_Score': credit_score,
        'Loan_Amount': loan_amount,
        'Loan_Purpose': loan_purposes,
        'Creditworthy': creditworthy
    })
    
    return df

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(data_dir, 'credit_data.csv')
    df = generate_credit_dataset()
    df.to_csv(output_path, index=False)
    print(f"Generated realistic credit dataset with {len(df)} rows saved to: {output_path}")
