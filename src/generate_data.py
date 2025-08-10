# src/generate_data.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

RNG = np.random.default_rng(42)

def generate_synthetic(n=20000):
    """
    Generate synthetic hospital admission data for readmission prediction.
    Returns a DataFrame with features and a binary target 'readmit_30d'
    """
    age = RNG.integers(18, 95, size=n)
    sex = RNG.choice(['M', 'F'], size=n, p=[0.48, 0.52])
    length_of_stay = np.maximum(1, (RNG.poisson(5, size=n)))  # days
    num_prev_adm = RNG.poisson(1.2, size=n)
    comorbidity_score = np.clip(RNG.normal(2.5, 1.8, size=n), 0, 10)
    labs_mean = np.clip(RNG.normal(100, 20, size=n), 20, 300)  # e.g. composite lab score
    med_changes = RNG.integers(0, 6, size=n)  # number of medication changes at discharge
    discharge_disposition = RNG.choice(['home', 'other_facility', 'home_with_care'],
                                       size=n, p=[0.75, 0.15, 0.10])
    insurance = RNG.choice(['public', 'private', 'selfpay'], size=n, p=[0.5, 0.4, 0.1])

    # base risk score (logit)
    logit = (
        0.03 * (age - 50) +
        0.25 * (comorbidity_score) +
        0.15 * num_prev_adm +
        0.12 * np.log1p(length_of_stay) +
        0.02 * (labs_mean - 100) +
        0.18 * med_changes
    )
    # discharge disposition increases risk when not home
    logit += np.where(discharge_disposition == 'other_facility', 0.6, 0.0)
    logit += np.where(discharge_disposition == 'home_with_care', 0.3, 0.0)
    # insurance effect
    logit += np.where(insurance == 'public', 0.15, 0.0)
    logit += np.where(insurance == 'selfpay', -0.2, 0.0)
    # sex little effect
    logit += np.where(sex == 'F', -0.03, 0.0)

    # convert logit to probability via logistic
    prob = 1 / (1 + np.exp(-logit))

    # add noise
    prob = np.clip(prob * RNG.normal(1.0, 0.12, size=n), 0, 1)

    readmit_30d = RNG.binomial(1, prob)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'length_of_stay': length_of_stay,
        'num_prev_adm': num_prev_adm,
        'comorbidity_score': comorbidity_score,
        'labs_mean': labs_mean,
        'med_changes': med_changes,
        'discharge_disposition': discharge_disposition,
        'insurance': insurance,
        'readmit_30d': readmit_30d
    })
    return df

def save_data(path='data/readmission_synthetic.csv', n=20000, test_size=0.2):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = generate_synthetic(n)
    df.to_csv(path, index=False)
    print(f"Saved synthetic data to: {path}")

    # also write train/test split for convenience
    train, test = train_test_split(df, test_size=test_size, random_state=42, stratify=df['readmit_30d'])
    train.to_csv('data/train.csv', index=False)
    test.to_csv('data/test.csv', index=False)
    print("Saved data/train.csv and data/test.csv")

if __name__ == "__main__":
    save_data()
