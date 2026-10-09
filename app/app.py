
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Telco Churn",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Telco Customer Churn")
st.write("Estimez le risque de départ d'un client.")

# Charger les modèles
@st.cache_resource
def load_files():
    model = joblib.load("models/model.pkl")
    prep = joblib.load("models/clustering/preprocessor.pkl")
    kmeans = joblib.load("models/clustering/kmeans_churn_model.pkl"
    )
    return model, prep, kmeans


try:
    model, prep, kmeans = load_files()
except Exception as e:
    st.error(f"Impossible de charger les modèles : {e}")
    st.stop()


# Formulaire client
with st.form("client_form"):

    st.subheader("1. Informations générales")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Genre",
            ["Female", "Male"]
        )

        SeniorCitizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Non" if x == 0 else "Oui"
        )

        Partner = st.selectbox(
            "A un partenaire ?",
            ["Yes", "No"]
        )

        Dependents = st.selectbox(
            "A des personnes à charge ?",
            ["Yes", "No"]
        )

        tenure = st.number_input(
            "Ancienneté (mois)",
            min_value=0,
            max_value=100,
            value=12
        )

    with col2:
        PhoneService = st.selectbox(
            "Service téléphonique",
            ["Yes", "No"]
        )

        MultipleLines = st.selectbox(
            "Lignes multiples",
            ["No", "Yes", "No phone service"]
        )

        InternetService = st.selectbox(
            "Service Internet",
            ["DSL", "Fiber optic", "No"]
        )

        Contract = st.selectbox(
            "Type de contrat",
            ["Month-to-month", "One year", "Two year"]
        )

        PaperlessBilling = st.selectbox(
            "Facturation électronique",
            ["Yes", "No"]
        )

    st.subheader("2. Services Internet")

    col3, col4 = st.columns(2)

    with col3:
        OnlineSecurity = st.selectbox(
            "Sécurité Internet",
            ["No", "Yes", "No internet service"]
        )

        OnlineBackup = st.selectbox(
            "Sauvegarde en ligne",
            ["No", "Yes", "No internet service"]
        )

        DeviceProtection = st.selectbox(
            "Protection des appareils",
            ["No", "Yes", "No internet service"]
        )

    with col4:
        TechSupport = st.selectbox(
            "Support technique",
            ["No", "Yes", "No internet service"]
        )

        StreamingTV = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )

        StreamingMovies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

    st.subheader("3. Informations financières")

    col5, col6 = st.columns(2)

    with col5:
        PaymentMethod = st.selectbox(
            "Mode de paiement",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        MonthlyCharges = st.number_input(
            "Frais mensuels",
            min_value=0.0,
            value=70.0,
            step=1.0
        )

    with col6:
        TotalCharges = st.number_input(
            "Frais totaux",
            min_value=0.0,
            value=840.0,
            step=10.0
        )

    submitted = st.form_submit_button(
        "🔍 Prédire le churn",
        use_container_width=True
    )


# Prédiction
if submitted:

    # Créer les données avec les noms de colonnes originaux
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
        # Vérifier les colonnes attendues par le préprocesseur
        expected_columns = list(prep.feature_names_in_)
        client_data = client_data[expected_columns]

        # Appliquer le prétraitement
        client_prep = prep.transform(client_data)

        # Ajouter le cluster, comme pendant l'entraînement
        client_scaled = pd.DataFrame(client_prep)
        client_scaled["cluster"] = kmeans.predict(client_prep)

        # Préparer les colonnes pour le modèle entraîné
        client_scaled.columns = client_scaled.columns.astype(str)
        client_scaled = client_scaled.astype(float)

        # Faire la prédiction
        prediction = model.predict(client_scaled)[0]
        probability = model.predict_proba(client_scaled)[0, 1]

        # Afficher les résultats
        st.divider()
        st.subheader("Résultat de la prédiction")

        col_result1, col_result2 = st.columns(2)

        with col_result1:
            st.metric(
                "Probabilité estimée de churn",
                f"{probability:.1%}"
            )

        with col_result2:
            if prediction == 1:
                st.error("Départ probable")
            else:
                st.success("Pas de départ prédit")

        st.progress(float(probability))

        if prediction == 1:
            st.warning(
                "Le modèle estime que ce client présente "
                "un risque de départ. Cette estimation "
                "n'est pas une certitude."
            )
        else:
            st.info(
                "Le modèle ne prédit pas de départ pour "
                "ce client. Cela ne garantit pas qu'il restera."
            )

    except Exception as e:
        st.error(f"Erreur pendant la prédiction : {e}")
        st.caption(
            "Vérifiez les colonnes attendues, le prétraitement "
            "et la compatibilité du modèle enregistré."
        )