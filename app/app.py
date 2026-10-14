
import streamlit as st
import pandas as pd
import joblib

# Configuration
st.set_page_config(page_title="Telco Churn", page_icon="📊")
st.title("Telco Customer Churn")
st.write("Estimer le risque de départ d'un client.")

# Charger les modèles
@st.cache_resource
def load_models():
    model = joblib.load("models/model.pkl")
    prep = joblib.load("models/clustering/preprocessor.pkl")
    kmeans = joblib.load("models/clustering/kmeans_churn_model.pkl")
    return model, prep, kmeans

try:
    model, prep, kmeans = load_models()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# Formulaire
with st.form("client_form"):

    st.subheader("Informations du client")

    gender = st.selectbox("Genre", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    Partner = st.selectbox("Partenaire", ["Yes", "No"])
    Dependents = st.selectbox("Personnes à charge", ["Yes", "No"])
    tenure = st.number_input("Ancienneté (mois)", 0, 100, 12)

    PhoneService = st.selectbox("Téléphone", ["Yes", "No"])
    MultipleLines = st.selectbox(
        "Lignes multiples", ["No", "Yes"]
    )
    InternetService = st.selectbox(
        "Internet", ["DSL", "Fiber optic", "No"]
    )
    Contract = st.selectbox(
        "Contrat", ["Month-to-month", "One year", "Two year"]
    )
    PaperlessBilling = st.selectbox(
        "Facturation électronique", ["Yes", "No"]
    )

    OnlineSecurity = st.selectbox(
        "Sécurité Internet", ["No", "Yes"]
    )
    OnlineBackup = st.selectbox(
        "Sauvegarde en ligne", ["No", "Yes"]
    )
    DeviceProtection = st.selectbox(
        "Protection des appareils", ["No", "Yes"]
    )
    TechSupport = st.selectbox(
        "Support technique", ["No", "Yes"]
    )
    StreamingTV = st.selectbox(
        "Streaming TV", ["No", "Yes"]
    )
    StreamingMovies = st.selectbox(
        "Streaming Movies", ["No", "Yes"]
    )

    PaymentMethod = st.selectbox(
        "Mode de paiement",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    MonthlyCharges = st.number_input("Frais mensuels", min_value=0.0, value=70.0)
    TotalCharges = st.number_input("Frais totaux", min_value=0.0, value=840.0)

    submitted = st.form_submit_button("Prédire le churn")

# Prédiction
if submitted:

    client_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    try:
        client_data = client_data[list(prep.feature_names_in_)]
        client_prep = prep.transform(client_data)

        cluster = kmeans.predict(client_prep)[0]

        client_final = pd.DataFrame(client_prep)
        client_final["cluster"] = cluster
        client_final.columns = client_final.columns.astype(str)

        prediction = model.predict(client_final)[0]
        probability = model.predict_proba(client_final)[0][1]

        st.subheader("Résultat")

        st.metric("Probabilité de churn", f"{probability:.1%}")

        if prediction == 1:
            st.error("Départ probable")
        else:
            st.success("Pas de départ prédit")

        st.progress(float(probability))

    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")
