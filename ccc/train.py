# train.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

DISEASE_LABELS = {
    "asthma": 0,
    "cold": 1,
    "covid": 2,
    "dengue": 3,
    "fever": 4,
    "flu": 5
}

ALL_COLUMNS = [
    # Asthma attributes
    "shortness_of_breath","wheezing","chest_tightness","cough","fatigue","difficulty_breathing",
    "rapid_breathing","chest_pain","sputum_production","nasal_congestion","running_nose","headache",
    "dizziness","anxiety","reduced_exercise_tolerance",
    
    # Cold attributes
    "sneezing","sore_throat","watery_eyes","mild_fever","earache","post_nasal_drip","hoarseness",
    "chills","body_ache","mild_cough",
    
    # COVID attributes
    "fever","loss_of_smell","loss_of_taste","muscle_pain","congestion","diarrhea","nausea","vomiting",
    
    # Dengue attributes
    "high_fever","severe_headache","pain_behind_eyes","joint_pain","rash","bleeding","abdominal_pain",
    "loss_of_appetite","restlessness",
    
    # Fever attributes
    "throught_pain","sweating","weakness","high_temperature","rapid_heartbeat",
]

ALL_COLUMNS = list(dict.fromkeys(ALL_COLUMNS))  # remove duplicates while preserving order


def load_and_prepare(path, disease_name):
    df = pd.read_csv(path)
    
    df = df[df["outcome"] == 1].copy()
    
    df["disease_label"] = DISEASE_LABELS[disease_name]
    
    for col in ALL_COLUMNS:
        if col not in df.columns:
            df[col] = 0
    
    df = df[ALL_COLUMNS + ["disease_label"]]
    
    for col in ALL_COLUMNS:
        df[col] = df[col].astype(int) 
    
    return df

asthma_df = load_and_prepare("asthma_dataset.csv", "asthma")
cold_df   = load_and_prepare("cold_dataset.csv",   "cold")
covid_df  = load_and_prepare("covid_dataset.csv",  "covid")
dengue_df = load_and_prepare("dengue_dataset.csv", "dengue")
fever_df  = load_and_prepare("fever_dataset.csv",  "fever")
flu_df    = load_and_prepare("flu_dataset.csv",    "flu")

all_data = pd.concat([asthma_df, cold_df, covid_df, dengue_df, fever_df, flu_df], ignore_index=True)

all_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)


X = all_data[ALL_COLUMNS]
y = all_data["disease_label"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)


preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Validation Accuracy:", acc)

joblib.dump(model, "disease_xgb_model.pkl")
print("Model saved to disease_xgb_model.pkl")
