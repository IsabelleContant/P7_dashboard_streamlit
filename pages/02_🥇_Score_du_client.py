import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sklearn
import lightgbm as lgb
from lightgbm import LGBMClassifier
import shap
from streamlit_shap import st_shap
import pickle
from PIL import Image

############################
# Configuration de la page #
############################
st.set_page_config(
        page_title='Score du Client',
        page_icon = "🥇",
        layout="wide" )

# Définition de quelques styles css
st.markdown(""" 
            <style>
            body {font-family:'Roboto Condensed';}
            h1 {font-family:'Roboto Condensed';}
            h2 {font-family:'Roboto Condensed';}
            p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
            .css-18e3th9 {padding-top: 1rem; 
                          padding-right: 1rem; 
                          padding-bottom: 1rem; 
                          padding-left: 1rem;}
            .css-184tjsw p {font-family:'Roboto Condensed'; color:Gray; font-size:1rem;}
            </style> """, 
            unsafe_allow_html=True)

# Centrage de l'image du logo dans la sidebar
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    image = Image.open('logo projet fintech.png')
    st.sidebar.image(image, use_column_width="always")
with col3:
    st.sidebar.write("")

########################
# Lecture des fichiers #
########################
@st.cache #mise en cache de la fonction pour exécution unique
def lecture_X_test_original():
    X_test_original = pd.read_csv("Data/X_test_original.csv")
    X_test_original = X_test_original.rename(columns=str.lower)
    return X_test_original

@st.cache 
def lecture_X_test_clean():
    X_test_clean = pd.read_csv("Data/X_test_clean.csv")
    #st.dataframe(X_test_clean)
    return X_test_clean

@st.cache 
def lecture_description_variables():
    description_variables = pd.read_csv("Data/description_variable.csv", sep=";")
    return description_variables


if __name__ == "__main__":

    lecture_X_test_original()
    lecture_X_test_clean()
    lecture_description_variables()

    # Titre 1
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                1. Quel est le score de votre client ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

    ##########################################################
    # Création et affichage du sélecteur du numéro de client #
    ##########################################################
    liste_clients = list(lecture_X_test_original()['sk_id_curr'])
    col1, col2 = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
    with col1:
        ID_client = st.selectbox("*Veuillez sélectionner le numéro de votre client à l'aide du menu déroulant 👇*", 
                                (liste_clients))
        st.write("Vous avez sélectionné l'identifiant n° :", ID_client)
    with col2:
        st.write("")

    #################################################
    # Lecture du modèle de prédiction et des scores #
    #################################################
    model_LGBM = pickle.load(open("model_LGBM.pkl", "rb"))
    y_pred_lgbm = model_LGBM.predict(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))    # Prédiction de la classe 0 ou 1
    y_pred_lgbm_proba = model_LGBM.predict_proba(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1)) # Prédiction du % de risque

    # Récupération du score du client
    y_pred_lgbm_proba_df = pd.DataFrame(y_pred_lgbm_proba, columns=['proba_classe_0', 'proba_classe_1'])
    y_pred_lgbm_proba_df = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'],
                                    lecture_X_test_clean()['sk_id_curr']], axis=1)
    #st.dataframe(y_pred_lgbm_proba_df)
    score = y_pred_lgbm_proba_df[y_pred_lgbm_proba_df['sk_id_curr']==ID_client]
    score_value = round(score.proba_classe_1.iloc[0]*100, 2)

    # Récupération de la décision
    y_pred_lgbm_df = pd.DataFrame(y_pred_lgbm, columns=['prediction'])
    y_pred_lgbm_df = pd.concat([y_pred_lgbm_df, lecture_X_test_clean()['sk_id_curr']], axis=1)
    y_pred_lgbm_df['client'] = np.where(y_pred_lgbm_df.prediction == 1, "non solvable", "solvable")
    y_pred_lgbm_df['decision'] = np.where(y_pred_lgbm_df.prediction == 1, "refuser", "accorder")
    solvabilite = y_pred_lgbm_df.loc[y_pred_lgbm_df['sk_id_curr']==ID_client, "client"].values
    decision = y_pred_lgbm_df.loc[y_pred_lgbm_df['sk_id_curr']==ID_client, "decision"].values

    ##############################################################
    # Affichage du score et du graphique de gauge sur 2 colonnes #
    ##############################################################
    col1, col2 = st.columns(2)
    with col2:
        st.markdown(""" <br> <br> """, unsafe_allow_html=True)
        st.write(f"Le client dont l'identifiant est **{ID_client}** a obtenu le score de **{score_value:.1f}%**.")
        st.write(f"**Il y a donc un risque de {score_value:.1f}% que le client ait des difficultés de paiement.**")
        st.write(f"Le client est donc considéré par *'Prêt à dépenser'* comme **{solvabilite[0]}** \
                et décide de lui **{decision[0]}** le crédit. ")
    # Impression du graphique jauge
    with col1:
        fig = go.Figure(go.Indicator(
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        value = float(score_value),
                        mode = "gauge+number+delta",
                        title = {'text': "Score du client", 'font': {'size': 24}},
                        delta = {'reference': 35.2, 'increasing': {'color': "#3b203e"}},
                        gauge = {'axis': {'range': [None, 100],
                                'tickwidth': 3,
                                'tickcolor': 'darkblue'},
                                'bar': {'color': 'white', 'thickness' : 0.3},
                                'bgcolor': 'white',
                                'borderwidth': 1,
                                'bordercolor': 'gray',
                                'steps': [{'range': [0, 20], 'color': '#e8af92'},
                                        {'range': [20, 40], 'color': '#db6e59'},
                                        {'range': [40, 60], 'color': '#b43058'},
                                        {'range': [60, 80], 'color': '#772b58'},
                                        {'range': [80, 100], 'color': '#3b203e'}],
                                'threshold': {'line': {'color': 'white', 'width': 8},
                                            'thickness': 0.8,
                                            'value': 35.2 }}))

        fig.update_layout(paper_bgcolor='white',
                        height=400, width=500,
                        font={'color': '#772b58', 'family': 'Roboto Condensed'},
                        margin=dict(l=30, r=30, b=5, t=5))
        st.plotly_chart(fig, use_container_width=True)

    ################################
    # Explication de la prédiction #
    ################################
    # Titre 2
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                2. Comment le score de votre client est-il calculé ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

    # Calcul des valeurs Shap
    explainer_shap = shap.TreeExplainer(model_LGBM)
    shap_values = explainer_shap.shap_values(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))

    # récupération de l'index correspondant à l'identifiant du client
    idx = int(lecture_X_test_clean()[lecture_X_test_clean()['sk_id_curr']==ID_client].index[0])

    # Graphique force_plot
    st.write("Le graphique suivant appelé `force-plot` permet de voir où se place la prédiction (f(x)) par rapport à la `base value`.") 
    st.write("Nous observons également quelles sont les variables qui augmentent la probabilité du client d'être \
            en défaut de paiement (en rouge) et celles qui la diminuent (en bleu), ainsi que l’amplitude de cet impact.")
    st_shap(shap.force_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).iloc[idx,:], 
                            link='logit',
                            figsize=(20, 8),
                            ordering_keys=True,
                            text_rotation=0,
                            contribution_threshold=0.05))
    # Graphique decision_plot
    st.write("Le graphique ci-dessous appelé `decision_plot` est une autre manière de comprendre la prédiction.\
            Comme pour le graphique précédent, il met en évidence l’amplitude et la nature de l’impact de chaque variable \
            avec sa quantification ainsi que leur ordre d’importance. Mais surtout il permet d'observer \
            “la trajectoire” prise par la prédiction du client pour chacune des valeurs des variables affichées. ")
    st.write("Seules les 15 variables explicatives les plus importantes sont affichées par ordre décroissant.")
    st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).iloc[idx,:], 
                            feature_names=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns.to_list(),
                            feature_order='importance',
                            feature_display_range=slice(None, -16, -1), # affichage des 15 variables les + importantes
                            link='logit'))

    # Titre 3
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                3. Lexique des variables </h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

    st.write("La base de données globale contient un peu plus de 200 variables explicatives. Certaines d'entre elles étaient peu \
            renseignées ou peu voir non disciminantes et d'autres très corrélées (2 variables corrélées entre elles \
            apportent la même information : l'une d'elles est donc redondante).")
    st.write("Après leur analyse, 56 variables se sont avérées pertinentes pour prédire si le client aura ou non des difficultés de paiement.")

    pd.set_option('display.max_colwidth', None)
    st.dataframe(lecture_description_variables())
