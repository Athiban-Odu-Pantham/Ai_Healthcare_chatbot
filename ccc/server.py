# server.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  


xgb_model = joblib.load("disease_xgb_model.pkl")

ALL_COLUMNS = [
    "shortness_of_breath","wheezing","chest_tightness","cough","fatigue","difficulty_breathing",
    "rapid_breathing","chest_pain","sputum_production","nasal_congestion","running_nose","headache",
    "dizziness","anxiety","reduced_exercise_tolerance","sneezing","sore_throat","watery_eyes","mild_fever",
    "earache","post_nasal_drip","hoarseness","chills","body_ache","mild_cough","fever","loss_of_smell",
    "loss_of_taste","muscle_pain","congestion","diarrhea","nausea","vomiting","high_fever","severe_headache",
    "pain_behind_eyes","joint_pain","rash","bleeding","abdominal_pain","loss_of_appetite","restlessness",
    "throught_pain","sweating","weakness","high_temperature","rapid_heartbeat"
]

DISEASE_MAP = {
    0: "Asthma",
    1: "Cold",
    2: "COVID-19",
    3: "Dengue",
    4: "Fever",
    5: "Flu"
}

print("Loading DialoGPT model... (this may take a while first time)")
model_name = "microsoft/DialoGPT-medium"
chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
chat_model = AutoModelForCausalLM.from_pretrained(model_name)
chat_pipeline = pipeline("conversational", model=chat_model, tokenizer=chat_tokenizer)

from transformers import Conversation
conv_state = Conversation()

user_symptoms = {col: 0 for col in ALL_COLUMNS}  # all zero by default
collecting_symptoms = False
symptom_questions = []
current_symptom_index = 0

DISEASE_SYMPTOMS = {
    "cold": [
        "sneezing", "running_nose", "cough", "nasal_congestion", "sore_throat", "headache", "fatigue",
        "watery_eyes", "mild_fever", "earache", "post_nasal_drip", "hoarseness", "chills", "body_ache", "mild_cough"
    ],
    "fever": [
        "cough", "throught_pain", "headache", "fatigue", "loss_of_appetite", "sweating", "chills", "muscle_pain",
        "joint_pain", "nausea", "vomiting", "dizziness", "weakness", "high_temperature", "rapid_heartbeat"
    ],
    "asthma": [
        "shortness_of_breath", "wheezing", "chest_tightness", "cough", "fatigue", "difficulty_breathing",
        "rapid_breathing", "chest_pain", "sputum_production", "nasal_congestion", "running_nose", "headache",
        "dizziness", "anxiety", "reduced_exercise_tolerance"
    ],
    "covid": [
        "fever", "cough", "shortness_of_breath", "loss_of_smell", "loss_of_taste", "fatigue", "sore_throat",
        "headache", "muscle_pain", "congestion", "chills", "chest_pain", "diarrhea", "nausea", "vomiting"
    ],
    "dengue": [
        "high_fever", "severe_headache", "pain_behind_eyes", "joint_pain", "muscle_pain", "rash", "nausea", "vomiting",
        "bleeding", "abdominal_pain", "fatigue", "loss_of_appetite", "restlessness", "dizziness"
    ],
    "flu": [
        "fever", "chills", "body_ache", "sore_throat", "cough", "runny_nose", "fatigue", "headache", "muscle_pain",
        "congestion", "sneezing", "loss_of_appetite", "nausea", "vomiting", "dizziness"
    ]
}

possible_diseases = ["asthma", "cold", "covid", "dengue", "fever", "flu"]

@app.route("/")
def home():
    return "Welcome to the Healthcare Chatbot API. Use POST /chat to interact."

@app.route("/chat", methods=["POST"])
def chat():
    global collecting_symptoms, symptom_questions, current_symptom_index
    
    user_data = request.json
    user_message = user_data.get("message", "")
    
    conv_state.add_user_input(user_message)
    chat_pipeline(conv_state)
    bert_response = conv_state.generated_responses[-1] if conv_state.generated_responses else ""
    
    triggered_disease = None
    for d in possible_diseases:
        if d in user_message.lower():
            triggered_disease = d
            break
    
    if triggered_disease:
        collecting_symptoms = True
        symptom_list = DISEASE_SYMPTOMS[triggered_disease]
        symptom_questions = [
            {"symptom": s, "question": f"Do you have {s.replace('_',' ')}? (yes/no)"}
            for s in symptom_list
        ]
        current_symptom_index = 0
        bot_reply = (f"I see you mentioned {triggered_disease}. Let me ask you a few questions about your symptoms to help with a diagnosis.")
        return jsonify({"bot": bot_reply})
    
    if collecting_symptoms and current_symptom_index < len(symptom_questions):
        answer = user_message.strip().lower()
        if answer in ["yes", "no"]:
            symptom_key = symptom_questions[current_symptom_index]["symptom"]
            user_symptoms[symptom_key] = 1 if answer == "yes" else 0
            current_symptom_index += 1
            if current_symptom_index < len(symptom_questions):
                next_q = symptom_questions[current_symptom_index]["question"]
                return jsonify({"bot": next_q})
            else:
                collecting_symptoms = False
                return jsonify({"bot": "Thank you. Let me analyze your symptoms. Type anything (e.g., 'ok') to continue."})
        else:
            return jsonify({"bot": "Please answer 'yes' or 'no'."})
    
    if not collecting_symptoms and current_symptom_index > 0 and current_symptom_index == len(symptom_questions):
        x_input = np.array([user_symptoms[col] for col in ALL_COLUMNS]).reshape(1, -1)
        prediction = xgb_model.predict(x_input)[0]
        disease_name = DISEASE_MAP.get(int(prediction), "Unknown")
        recommendation = ""
        if disease_name == "Asthma":
            recommendation = "Use an inhaler if prescribed, avoid triggers, and consult a pulmonologist."
        elif disease_name == "Cold":
            recommendation = "Stay hydrated, rest well, and consider over-the-counter cold medications."
        elif disease_name == "COVID-19":
            recommendation = "Self-isolate, get tested, and follow local health guidelines."
        elif disease_name == "Dengue":
            recommendation = "Stay hydrated, monitor fever, and seek immediate medical attention if symptoms worsen."
        elif disease_name == "Fever":
            recommendation = "Take fever-reducing medication if needed, rest, and drink plenty of fluids."
        elif disease_name == "Flu":
            recommendation = "Consider antiviral medications, rest well, and stay hydrated."
        else:
            recommendation = "Consult a healthcare professional for an accurate diagnosis."
        
        symptom_questions.clear()
        current_symptom_index = 0
        final_reply = (f"Based on your answers, you may have **{disease_name}**.\nRecommendation: {recommendation}")
        return jsonify({"bot": final_reply})
    
    return jsonify({"bot": bert_response})

@app.route("/chat", methods=["GET"])
def chat_get():
    return "Please use a POST request with a JSON payload to interact with the chatbot."

if __name__ == "__main__":
    app.run(debug=True)
