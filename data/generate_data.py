import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_emi_data(num_records=2500):
    print(f"Generating {num_records} synthetic credit records...")
    
    # 1. Age: 21 to 65 years
    age = np.random.randint(21, 66, size=num_records)
    
    # 2. Monthly Income: log-normal-like mixture to look realistic
    # Typically, most people are in the lower-middle range, fewer in high range
    income_base = np.random.exponential(scale=50000, size=num_records) + 15000
    monthly_income = np.clip(income_base, 15000, 250000).astype(int)
    
    # 3. Monthly Expenses: 25% to 70% of income + some base expense, but must be less than income
    expense_ratio = np.random.uniform(0.25, 0.70, size=num_records)
    monthly_expenses = (monthly_income * expense_ratio + np.random.randint(2000, 8000, size=num_records)).astype(int)
    # Ensure expenses are less than income
    monthly_expenses = np.minimum(monthly_expenses, monthly_income - 3000)
    monthly_expenses = np.maximum(monthly_expenses, 5000) # Min expense of 5000
    
    # 4. Credit Score: 300 to 850 (centered around 650 with normal distribution)
    credit_score = np.random.normal(loc=650, scale=120, size=num_records).astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    
    # 5. Employment Type: Salaried (60%), Self-Employed (30%), Unemployed (10%)
    emp_choices = ['Salaried', 'Self-Employed', 'Unemployed']
    employment_type = np.random.choice(emp_choices, size=num_records, p=[0.60, 0.30, 0.10])
    
    # Unemployed people have very low income (usually government support or passive), let's adjust their income
    for i in range(num_records):
        if employment_type[i] == 'Unemployed':
            monthly_income[i] = np.random.randint(5000, 20000)
            monthly_expenses[i] = np.minimum(monthly_income[i] - 1000, np.random.randint(4000, 15000))
            monthly_expenses[i] = np.maximum(monthly_expenses[i], 3000)
            
    # 6. Existing Loan Amount & Existing EMI
    # 45% of people have no existing loan, others have loans scaled to their income
    has_loan = np.random.choice([0, 1], size=num_records, p=[0.45, 0.55])
    existing_loan_amount = np.zeros(num_records, dtype=int)
    existing_emi = np.zeros(num_records, dtype=int)
    
    for i in range(num_records):
        if has_loan[i] == 1 and employment_type[i] != 'Unemployed':
            # Loan amount is typically 1 to 10 times their monthly income
            existing_loan_amount[i] = int(monthly_income[i] * np.random.uniform(1.0, 8.0))
            # Round loan amount to nearest 10,000
            existing_loan_amount[i] = int(np.round(existing_loan_amount[i] / 10000) * 10000)
            # EMI is loan amount spread over 1 to 5 years (12 to 60 months) with 10-15% interest
            # Simple approximation: EMI is 2% to 3.5% of the loan amount
            existing_emi[i] = int(existing_loan_amount[i] * np.random.uniform(0.02, 0.035))
            # Round EMI to nearest 100
            existing_emi[i] = int(np.round(existing_emi[i] / 100) * 100)
            
            # Ensure existing EMI doesn't exceed income - expenses
            max_possible_emi = max(0, monthly_income[i] - monthly_expenses[i] - 2000)
            if existing_emi[i] > max_possible_emi:
                existing_emi[i] = max_possible_emi
                if existing_emi[i] == 0:
                    existing_loan_amount[i] = 0
                else:
                    existing_loan_amount[i] = int(existing_emi[i] / 0.025)
    
    # 7. Number of Dependents: 0 to 4
    dependents = np.random.choice([0, 1, 2, 3, 4], size=num_records, p=[0.3, 0.3, 0.25, 0.1, 0.05])
    
    # ----------------------------------------------------
    # Target 1: EMI_Eligibility (Classification Target)
    # ----------------------------------------------------
    # Calculate Debt-to-Income (DTI) Ratio
    # If income is 0 (or very low), DTI becomes huge
    dti = np.zeros(num_records)
    for i in range(num_records):
        if monthly_income[i] > 0:
            dti[i] = (monthly_expenses[i] + existing_emi[i]) / monthly_income[i]
        else:
            dti[i] = 2.0
            
    # Calculate a credit score factor (0 to 1)
    credit_factor = (credit_score - 300) / 550
    
    # Rule-based logit with some noise
    # We want classification models to learn standard credit risk behaviors
    eligibility_score = np.zeros(num_records)
    for i in range(num_records):
        # Base logit
        logit = -1.5
        
        # Credit Score effect (strongest predictor)
        logit += 6.5 * credit_factor[i]
        if credit_score[i] < 550:
            logit -= 3.0  # Big penalty for subprime credit score
        elif credit_score[i] >= 750:
            logit += 2.0  # Bonus for excellent credit score
            
        # Debt-to-Income effect
        logit -= 7.0 * dti[i]
        if dti[i] > 0.55:
            logit -= 2.0  # Big penalty for high debt obligations
            
        # Employment Type effect
        if employment_type[i] == 'Salaried':
            logit += 1.5
        elif employment_type[i] == 'Self-Employed':
            logit += 0.5
        else: # Unemployed
            logit -= 5.0
            
        # Dependents effect
        logit -= 0.3 * dependents[i]
        
        # Age effect (slight penalty for very young or very old)
        if age[i] < 23 or age[i] > 60:
            logit -= 0.5
            
        # Sigmoid probability
        prob = 1.0 / (1.0 + np.exp(-logit))
        eligibility_score[i] = prob
        
    # Generate binary target with noise (random flip 4% of the time)
    emi_eligibility = (eligibility_score >= 0.5).astype(int)
    noise_mask = np.random.rand(num_records) < 0.04
    emi_eligibility[noise_mask] = 1 - emi_eligibility[noise_mask]
    
    # Overwrite eligibility for Unemployed: they should be ineligible 99% of the time
    for i in range(num_records):
        if employment_type[i] == 'Unemployed' and np.random.rand() < 0.99:
            emi_eligibility[i] = 0
            
    # ----------------------------------------------------
    # Target 2: Max_Affordable_EMI (Regression Target)
    # ----------------------------------------------------
    max_affordable_emi = np.zeros(num_records, dtype=int)
    for i in range(num_records):
        if emi_eligibility[i] == 0:
            # If not eligible, their max affordable EMI is set to 0 or very small
            # (but let's keep some capacity for cases where they just missed eligibility, e.g. low score but high income,
            # or let's make it exactly 0 for Unemployed, and highly constrained for others).
            # To make it easier for regression model, let's calculate capacity based on finances,
            # but cap it low if credit score is bad.
            foir_cap = 0.10 + 0.15 * credit_factor[i]  # Max 10% to 25% FOIR for ineligible
        else:
            foir_cap = 0.30 + 0.25 * credit_factor[i]  # Max 30% to 55% FOIR for eligible
            
        # Capacity calculation: FOIR * Income - Existing EMI
        capacity = (monthly_income[i] * foir_cap) - existing_emi[i]
        
        # Adjust for dependents (each dependent reduces surplus income by 1000)
        capacity -= dependents[i] * 1000
        
        # Age penalty (slightly less capacity as you approach retirement)
        if age[i] > 55:
            capacity *= (1.0 - 0.01 * (age[i] - 55))
            
        # Unemployed capacity is always 0
        if employment_type[i] == 'Unemployed':
            capacity = 0
            
        # Ensure capacity is non-negative
        capacity = max(0, capacity)
        
        # Add some Gaussian noise (scaled to income)
        if capacity > 0:
            noise = np.random.normal(loc=0, scale=min(1500, capacity * 0.1))
            capacity += noise
            
        max_affordable_emi[i] = int(np.round(max(0, capacity) / 100) * 100)
        
    # Create DataFrame
    df = pd.DataFrame({
        'Age': age,
        'Monthly_Income': monthly_income,
        'Monthly_Expenses': monthly_expenses,
        'Credit_Score': credit_score,
        'Existing_Loan_Amount': existing_loan_amount,
        'Existing_EMI': existing_emi,
        'Employment_Type': employment_type,
        'Dependents': dependents,
        'EMI_Eligibility': emi_eligibility,
        'Max_Affordable_EMI': max_affordable_emi
    })
    
    return df

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    df = generate_emi_data(2500)
    
    output_path = os.path.join('data', 'emi_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset saved successfully to: {output_path}")
    print(df.head())
    print("\nSummary of target variables:")
    print(df[['EMI_Eligibility', 'Max_Affordable_EMI']].describe())
    print(f"Eligibility Rate: {df['EMI_Eligibility'].mean() * 100:.2f}%")
