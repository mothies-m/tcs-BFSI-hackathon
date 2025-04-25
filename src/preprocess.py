import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def preprocess_dataset(dataset_path):
    df = pd.read_csv(dataset_path + "/german_credit_data.csv")

    df = df.drop(columns=["Unnamed: 0"])
    df = df.fillna("Not Available")

    df["Risk"] = df["Risk"].replace({"bad": 0, "good": 1})

    # One-hot encode selected categorical features directly
    # (keep them as strings for this step)
    df = pd.get_dummies(df, columns=["Sex", "Checking account"])

    # Drop unneeded or redundant columns
    df = df.drop(["Housing", "Purpose", "Saving accounts"], axis=1)

    # Apply log transform
    for col in ["Duration", "Credit amount", "Age"]:
        df[col] = df[col].apply(lambda x: np.log(x) if x > 0 else 0)

    # Separate features and target
    X = df.drop("Risk", axis=1)
    y = df["Risk"]
    df.rename(
        columns={
            "Sex_0": "Sex_male",
            "Sex_1": "Sex_female",
            "Checking account_0": "Checking account_little",
            "Checking account_1": "Checking account_moderate",
            "Checking account_2": "Checking account_Not Available",
            "Checking account_3": "Checking account_rich",
        },
        inplace=True,
        errors="ignore",
    )
    new_order = [
        "Age",
        "Job",
        "Credit amount",
        "Duration",
        "Sex_female",
        "Sex_male",
        "Checking account_Not Available",
        "Checking account_little",
        "Checking account_moderate",
        "Checking account_rich",
        "Risk",
    ]

    # Reorder the columns
    df = df[new_order]

    # Oversampling using SMOTE
    oversample = SMOTE()
    X_smote, y_smote = oversample.fit_resample(X, y)

    print("Preprocessing Complete")
    return X_smote, y_smote
