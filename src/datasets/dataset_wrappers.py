import pandas as pd
import numpy as np
from sklearn.datasets import make_moons, make_circles


def german(path: str = "data/german.csv") -> dict:
    """
    Load the german dataset.

    Parameters:
        - path: the path to the german dataset (str)
    """
    raw_df = pd.read_csv(path)
    categorical_columns = [
        "checking_status",
        "credit_history",
        "purpose",
        "savings_status",
        "employment",
        "personal_status",
        "other_parties",
        "property_magnitude",
        "other_payment_plans",
        "housing",
        "job",
        "own_telephone",
        "foreign_worker",
    ]
    continuous_columns = [
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
    ]
    target_column = "class"

    monotonic_increase_columns = []
    monotonic_decrease_columns = []
    freeze_columns = ["foreign_worker"]
    feature_ranges = {
        "duration": [int(raw_df["duration"].min()), int(raw_df["duration"].max())],
        "credit_amount": [
            int(raw_df["credit_amount"].min()),
            int(raw_df["credit_amount"].max()),
        ],
        "installment_commitment": [
            int(raw_df["installment_commitment"].min()),
            int(raw_df["installment_commitment"].max()),
        ],
        "residence_since": [
            int(raw_df["residence_since"].min()),
            int(raw_df["residence_since"].max()),
        ],
        "existing_credits": [
            int(raw_df["existing_credits"].min()),
            int(raw_df["existing_credits"].max()),
        ],
        "num_dependents": [
            int(raw_df["num_dependents"].min()),
            int(raw_df["num_dependents"].max()),
        ],
        "age": [18, int(raw_df["age"].max())],
    }

    data = {
        "raw_df": raw_df,
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "target_column": target_column,
        "monotonic_increase_columns": monotonic_increase_columns,
        "monotonic_decrease_columns": monotonic_decrease_columns,
        "freeze_columns": freeze_columns,
        "feature_ranges": feature_ranges,
    }

    return data


def german_binary(path: str = "data/german.csv") -> dict:
    """
    Load the german dataset.

    binarize the categorical columns
    """
    raw_df = pd.read_csv(path)

    categorical_columns = [
        "checking_status",
        "credit_history",
        "purpose",
        "savings_status",
        "employment",
        "personal_status",
        "other_parties",
        "property_magnitude",
        "other_payment_plans",
        "housing",
        "job",
        "own_telephone",
        "foreign_worker",
    ]

    continuous_columns = [
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
    ]

    target_column = "class"

    monotonic_increase_columns = []
    monotonic_decrease_columns = []

    freeze_columns = ["foreign_worker"]

    feature_ranges = {
        "duration": [int(raw_df["duration"].min()), int(raw_df["duration"].max())],
        "credit_amount": [
            int(raw_df["credit_amount"].min()),
            int(raw_df["credit_amount"].max()),
        ],
        "installment_commitment": [
            int(raw_df["installment_commitment"].min()),
            int(raw_df["installment_commitment"].max()),
        ],
        "residence_since": [
            int(raw_df["residence_since"].min()),
            int(raw_df["residence_since"].max()),
        ],
        "existing_credits": [
            int(raw_df["existing_credits"].min()),
            int(raw_df["existing_credits"].max()),
        ],
        "num_dependents": [
            int(raw_df["num_dependents"].min()),
            int(raw_df["num_dependents"].max()),
        ],
        "age": [18, int(raw_df["age"].max())],
    }

    binarized = {
        "has_checking": raw_df["checking_status"].apply(
            lambda x: 1 if x != "no checking" else 0
        ),
        "good_credit_history": raw_df["credit_history"].apply(
            lambda x: 1
            if x in ["critical/other existing credit", "delayed previously"]
            else 0
        ),
        "no_or_low_savings": raw_df["savings_status"].apply(
            lambda x: 1 if x in ["no known savings", "<100"] else 0
        ),
        "unemployed": raw_df["employment"].apply(
            lambda x: 1 if x == "unemployed" else 0
        ),
        "single": raw_df["personal_status"].apply(
            lambda x: 1 if x == "male single" else 0
        ),
        "other_parties": raw_df["other_parties"].apply(
            lambda x: 1 if x != "none" else 0
        ),
        "has_real_estate": raw_df["property_magnitude"].apply(
            lambda x: 1 if x == "real estate" else 0
        ),
        "other_payment_plans": raw_df["other_payment_plans"].apply(
            lambda x: 1 if x != "none" else 0
        ),
        "house_owner": raw_df["housing"].apply(lambda x: 1 if x == "own" else 0),
        "good_job_prospects": raw_df["job"].apply(
            lambda x: 1 if x in ["skilled", "high qualif/self emp/mgmt"] else 0
        ),
        "has_telephone": raw_df["own_telephone"].apply(
            lambda x: 1 if x == "yes" else 0
        ),
        "is_foreign_worker": raw_df["foreign_worker"].apply(
            lambda x: 1 if x == "yes" else 0
        ),
    }

    raw_df = raw_df.drop(columns=categorical_columns)
    raw_df = pd.concat([raw_df, pd.DataFrame(binarized)], axis=1)

    raw_df = raw_df.dropna(how="any", axis=0)

    categorical_columns = list(binarized.keys())

    print(raw_df.info())

    data = {
        "raw_df": raw_df,
        "categorical_columns": [],
        "continuous_columns": continuous_columns,
        "target_column": target_column,
        "monotonic_increase_columns": monotonic_increase_columns,
        "monotonic_decrease_columns": monotonic_decrease_columns,
        "freeze_columns": freeze_columns,
        "feature_ranges": feature_ranges,
        "binary_columns": list(binarized.keys()),
    }

    return data


def fico(path: str = "data/fico.csv") -> dict:
    """
    Load the fico dataset.

    Parameters:
        - path: the path to the fico dataset (str)
    """

    raw_df = pd.read_csv(path)
    categorical_columns = []
    continuous_columns = raw_df.columns.tolist()
    continuous_columns.remove("RiskPerformance")
    target_column = "RiskPerformance"

    freeze_columns = ["ExternalRiskEstimate"]
    feature_ranges = {
        "PercentTradesNeverDelq": [0, 100],
        "PercentInstallTrades": [0, 100],
        "PercentTradesWBalance": [0, 100],
    }
    monotonic_increase_columns = []
    monotonic_decrease_columns = []

    # In fico dataset negative values mean that the value is missing
    mask_negative = ~np.any(raw_df[continuous_columns] < 0, axis=1)
    raw_df = raw_df[mask_negative]

    data = {
        "raw_df": raw_df,
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "target_column": target_column,
        "monotonic_increase_columns": monotonic_increase_columns,
        "monotonic_decrease_columns": monotonic_decrease_columns,
        "freeze_columns": freeze_columns,
        "feature_ranges": feature_ranges,
    }

    return data


def wine_quality(path: str = "data/wine_quality.csv") -> dict:
    # fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol,quality

    raw_df = pd.read_csv(path)
    categorical_columns = []
    continuous_columns = raw_df.columns.tolist()
    continuous_columns.remove("quality")
    target_column = "quality"

    freeze_columns = []
    feature_ranges = {
        "fixed_acidity": [raw_df["fixed_acidity"].min(), raw_df["fixed_acidity"].max()],
        "volatile_acidity": [
            raw_df["volatile_acidity"].min(),
            raw_df["volatile_acidity"].max(),
        ],
        "citric_acid": [raw_df["citric_acid"].min(), raw_df["citric_acid"].max()],
        "residual_sugar": [
            raw_df["residual_sugar"].min(),
            raw_df["residual_sugar"].max(),
        ],
        "chlorides": [raw_df["chlorides"].min(), raw_df["chlorides"].max()],
        "free_sulfur_dioxide": [
            raw_df["free_sulfur_dioxide"].min(),
            raw_df["free_sulfur_dioxide"].max(),
        ],
        "total_sulfur_dioxide": [
            raw_df["total_sulfur_dioxide"].min(),
            raw_df["total_sulfur_dioxide"].max(),
        ],
        "density": [raw_df["density"].min(), raw_df["density"].max()],
        "pH": [raw_df["pH"].min(), raw_df["pH"].max()],
        "sulphates": [raw_df["sulphates"].min(), raw_df["sulphates"].max()],
        "alcohol": [raw_df["alcohol"].min(), raw_df["alcohol"].max()],
    }
    monotonic_increase_columns = []
    monotonic_decrease_columns = []

    data = {
        "raw_df": raw_df,
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "target_column": target_column,
        "monotonic_increase_columns": monotonic_increase_columns,
        "monotonic_decrease_columns": monotonic_decrease_columns,
        "freeze_columns": freeze_columns,
        "feature_ranges": feature_ranges,
    }

    return data


def breast_cancer(path: str = "data/breast_cancer.csv") -> dict:
    # radius1,texture1,perimeter1,area1,smoothness1,compactness1,concavity1,concave_points1,symmetry1,fractal_dimension1,radius2,texture2,perimeter2,area2,smoothness2,compactness2,concavity2,concave_points2,symmetry2,fractal_dimension2,radius3,texture3,perimeter3,area3,smoothness3,compactness3,concavity3,concave_points3,symmetry3,fractal_dimension3,diagnosis
    # 17.99,10.38,122.8,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189,M

    raw_df = pd.read_csv(path)
    categorical_columns = []
    continuous_columns = raw_df.columns.tolist()
    continuous_columns.remove("diagnosis")
    target_column = "diagnosis"

    freeze_columns = []
    feature_ranges = {k: [raw_df[k].min(), raw_df[k].max()] for k in continuous_columns}
    monotonic_increase_columns = []
    monotonic_decrease_columns = []

    data = {
        "raw_df": raw_df,
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "target_column": target_column,
        "monotonic_increase_columns": monotonic_increase_columns,
        "monotonic_decrease_columns": monotonic_decrease_columns,
        "freeze_columns": freeze_columns,
        "feature_ranges": feature_ranges,
    }

    return data


def diabetes(path: str = "data/diabetes.csv") -> dict:
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
    raw_df = pd.read_csv(path)
    categorical_columns = []
    continuous_columns = raw_df.columns.tolist()
    continuous_columns.remove("Outcome")
    target_column = "Outcome"

    freeze_columns = []
    feature_ranges = {k: [raw_df[k].min(), raw_df[k].max()] for k in continuous_columns}
    monotonic_increase_columns = []
    monotonic_decrease_columns = []

    data = {
        "raw_df": raw_df,
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "target_column": target_column,
        "monotonic_increase_columns": monotonic_increase_columns,
        "monotonic_decrease_columns": monotonic_decrease_columns,
        "freeze_columns": freeze_columns,
        "feature_ranges": feature_ranges,
    }

    return data


def car_eval(path: str = "data/car_eval.csv") -> dict:
    df = pd.read_csv(path)

    cat_to_num = {
        "buying": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
        "maint": {"low": 0, "med": 1, "high": 2, "vhigh": 3},
        "doors": {"2": 0, "3": 1, "4": 2, "5more": 3},
        "persons": {"2": 0, "4": 1, "more": 2},
        "lug_boot": {"small": 0, "med": 1, "big": 2},
        "safety": {"low": 0, "med": 1, "high": 2},
        "acceptability": {"unacc": 0, "acc": 1, "good": 1, "vgood": 1},
    }

    transformed = df.copy()
    for col, mapping in cat_to_num.items():
        transformed[col] = transformed[col].map(mapping)

    transformed = transformed.rename(columns={"acceptability": "target"})

    continuous_columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
    categorical_columns = []

    target_column = "target"
    freeze_columns = []

    feature_ranges = {}
    monotonic_increase_columns = []

    monotonic_decrease_columns = []

    data = {
        "raw_df": transformed,
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "target_column": target_column,
        "monotonic_increase_columns": monotonic_increase_columns,
        "monotonic_decrease_columns": monotonic_decrease_columns,
        "freeze_columns": freeze_columns,
        "feature_ranges": feature_ranges,
    }

    return data


def rice(path: str = "data/rice.csv") -> dict:
    raw_df = pd.read_csv(path)

    categorical_columns = []
    continuous_columns = raw_df.columns.tolist()
    continuous_columns.remove("class")

    target_column = "class"

    freeze_columns = []
    feature_ranges = {k: [raw_df[k].min(), raw_df[k].max()] for k in continuous_columns}
    monotonic_increase_columns = []

    monotonic_decrease_columns = []

    data = {
        "raw_df": raw_df,
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "target_column": target_column,
        "monotonic_increase_columns": monotonic_increase_columns,
        "monotonic_decrease_columns": monotonic_decrease_columns,
        "freeze_columns": freeze_columns,
        "feature_ranges": feature_ranges,
    }

    return data


def compas(path: str = "data/compas.csv") -> dict:
    """
    Load the compas dataset.

    Parameters:
        - path: the path to the compas dataset (str)
    """

    raw_df = pd.read_csv(
        "https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv"
    )
    raw_df = raw_df[raw_df["type_of_assessment"] == "Risk of Recidivism"]

    features = [
        "sex",
        "age",
        "race",
        "juv_fel_count",
        "decile_score",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_days_from_compas",
        "c_charge_degree",
        "two_year_recid",
    ]

    categorical_columns = ["sex", "race", "c_charge_degree"]
    continuous_columns = [
        "age",
        "juv_fel_count",
        "decile_score",
        "juv_misd_count",
        "juv_other_count",
        "priors_count",
        "c_days_from_compas",
    ]
    target_column = "two_year_recid"
    freeze_columns = ["age", "sex", "race", "c_charge_degree"]

    feature_ranges = {
        "age": [18, 100],
        "decile_score": [0, 10],
    }

    raw_df = raw_df[features]
    raw_df = raw_df.dropna(how="any", axis=0)

    data = {
        "raw_df": raw_df,
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "target_column": target_column,
        "freeze_columns": freeze_columns,
        "feature_ranges": feature_ranges,
    }

    return data


def moons(n_samples: int = 1000, noise: float = 0.2, random_state: int = 0) -> dict:
    """
    Create two moons with the same number of samples and noise.

    Parameters:
        - n_samples: the number of samples for each moon (int)
        - noise: the noise of the moons (float)
        - random_state: the random state (int)
    """
    data = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    data2 = make_moons(n_samples=n_samples, noise=noise, random_state=random_state + 1)

    X = np.concatenate([data[0], data2[0] / 1.5 + 1.6])
    y = np.concatenate([data[1], data2[1]])

    # Normalize the data to 0-1
    X = (X - X.min()) / (X.max() - X.min())

    raw_df = pd.DataFrame(X, columns=["x1", "x2"])
    raw_df["y"] = y

    categorical_columns = []
    continuous_columns = ["x1", "x2"]
    target_column = "y"
    freeze_columns = []
    feature_ranges = {"x1": [0, 1], "x2": [0, 1]}

    data = {
        "raw_df": raw_df,
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "target_column": target_column,
        "freeze_columns": freeze_columns,
        "feature_ranges": feature_ranges,
    }

    return data


def donuts(n_samples: int = 1000, noise: float = 0.1, random_state: int = 0) -> dict:
    """
    Create two donuts with the same number of samples and noise.

    Parameters:
        - n_samples: the number of samples for each donut (int)
        - noise: the noise of the donuts (float)
        - random_state: the random state (int)
    """
    data = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
    data2 = make_circles(
        n_samples=n_samples, noise=noise, random_state=random_state + 1, factor=0.5
    )

    X = np.concatenate([data[0], data2[0] / 1.5 + 1.6])
    y = np.concatenate([data[1], data2[1]])

    # Normalize the data to 0-1
    X = (X - X.min()) / (X.max() - X.min())

    raw_df = pd.DataFrame(X, columns=["x1", "x2"])
    raw_df["y"] = y

    categorical_columns = []
    continuous_columns = ["x1", "x2"]
    target_column = "y"
    freeze_columns = []
    feature_ranges = {"x1": [0, 1], "x2": [0, 1]}

    data = {
        "raw_df": raw_df,
        "categorical_columns": categorical_columns,
        "continuous_columns": continuous_columns,
        "target_column": target_column,
        "freeze_columns": freeze_columns,
        "feature_ranges": feature_ranges,
    }

    return data
