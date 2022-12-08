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
        page_title='Interprétation Globale de la prédiction',
        page_icon = "📊",
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
            .css-1offfwp li {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
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

###########################
# Calcul des valeurs SHAP #
###########################
@st.cache 
def calcul_valeurs_shap():
    model_LGBM = pickle.load(open("model_LGBM.pkl", "rb"))
    explainer = shap.TreeExplainer(model_LGBM)
    shap_values = explainer.shap_values(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))
    return shap_values


if __name__ == "__main__":
    
    lecture_X_test_original()
    lecture_X_test_clean()
    lecture_description_variables()
    calcul_valeurs_shap()

    # Titre 1
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                1. Quelles sont les variables globalement les plus importantes pour comprendre la prédiction ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

    st.write("L’importance des variables est calculée en moyennant la valeur absolue des valeurs de Shap. \
            Les caractéristiques sont classées de l'effet le plus élevé au plus faible sur la prédiction. \
            Le calcul prend en compte la valeur SHAP absolue, donc peu importe si la fonctionnalité affecte \
            la prédiction de manière positive ou négative.")

    st.write("Pour résumer, les valeurs de Shapley calculent l’importance d’une variable en comparant ce qu’un modèle prédit \
            avec et sans cette variable. Cependant, étant donné que l’ordre dans lequel un modèle voit les variables peut affecter \
            ses prédictions, cela se fait dans tous les ordres possibles, afin que les fonctionnalités soient comparées équitablement. \
            Cette approche est inspirée de la théorie des jeux.")

    st.write("*__Le diagramme d'importance des variables__* répertorie les variables les plus significatives par ordre décroissant.\
            Les *__variables en haut__* contribuent davantage au modèle que celles en bas et ont donc un *__pouvoir prédictif élevé__*.")

    fig = plt.figure()
    plt.title("Interprétation Globale :\n Diagramme d'Importance des Variables", 
            fontname='Roboto Condensed',
            fontsize=20, 
            fontstyle='italic')
    st_shap(shap.summary_plot(calcul_valeurs_shap()[1], 
                            feature_names=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns,
                            plot_size=(12, 16),
                            color='#9ebeb8',
                            plot_type="bar",
                            max_display=56,
                            show = False))
    plt.show()

    # Titre 2
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                2. Quel est l'Impact de chaque caractéristique sur la prédiction ?</h1>
                """, 
                unsafe_allow_html=True)
    st.write("")

    st.write("Le diagramme des valeurs SHAP ci-dessous indique également comment chaque caractéristique impacte la prédiction. \
            Les valeurs de Shap sont représentées pour chaque variable dans leur ordre d’importance. \
            Chaque point représente une valeur de Shap (pour un client).")
    st.write("Les points fuchsia représentent des valeurs élevées de la variable et les points verts des valeurs basses de la variable.")

    fig = plt.figure()
    plt.title("Interprétation Globale :\n Impact de chaque caractéristique sur la prédiction\n", 
            fontname='Roboto Condensed',
            fontsize=20, 
            fontstyle='italic')
    st_shap(shap.summary_plot(calcul_valeurs_shap()[1], 
                            features=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1),
                            feature_names=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns,
                            plot_size=(12, 16),
                            cmap='PiYG_r',
                            plot_type="dot",
                            max_display=56,
                            show = False))
    plt.show()

    st.write("14 variables ont un impact significatif sur la prédiction (Moyenne des valeurs absolues des valeurs de Shap >= 0.1). \
            La première est sans contexte le score normalisé à partir d'une source de données externes.")
    st.markdown("""
    1. Plus la valeur du 'Score normalisé à partir d'une source de données externe' est faible (points de couleur vert), 
       et plus la valeur Shap est élevée et donc plus le modèle prédit que le client aura des difficultés de paiement.<br>
    2. Plus la dernière demande de crédit du client, avant la demande actuelle, enregistrée au bureau des crédits, est récente 
       (points de couleur vert), plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
    3. Plus le montant payé par le client par rapport au montant attendu est faible (points de couleur vert), 
       plus la valeur Shap est élevée et donc plus le modèle pédit que le client aura des difficultés de paiement.<br>
    4. Si le client est un homme, la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
    5. Plus la durée mensuelle du contrat pécédent du client est élevé (points de couleur fuchsia), 
       plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
    6. Plus le nombre de contrats pécédents refusés pour le client est élevé (points de couleur fuchsia), 
       plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
    7. Plus le client est jeune (points de couleur vert), plus la valeur Shap est élevée et
       donc plus le modèle prédit qu'il aura des difficultés de paiement.<br>
    8. Lorsque le client n'est pas allé dans l'enseignement supérieur (points vert), 
       la valeur Shap est élevée et donc plus le modèle pédit que le client aura des difficultés de paiement.<br>
    9. Nombre de crédits soldés du client enregistrés au bureau du crédit : *impact indéfini* <br>
    10. Plus le nombre de versements réalisés par la client est faible (points de couleur vert), 
        plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
    11. Plus l'ancienneté du client dans son entreprise est faible (points de couleur vert), 
        plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
    12. Plus le nombre de Cartes de Crédit du client enregistrées au bureau du crédit est élevé (points de couleur fuchsia),
        plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
    13. Plus le montant de la demande de prêt actuelle du client est élevé (points de couleur fuchsia), 
        plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
    14. Plus le montant de la demande de prêt précédente du client est faible (points de couleur vert), 
        plus la valeur Shap est élevée et donc plus le modèle prédit qu'il aura des difficultés de paiement.
            """, 
            unsafe_allow_html=True)
    
    # Titre 2
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                2. Graphique de dépendance</h1>
                """, 
                unsafe_allow_html=True)
    st.write("Nous pouvons obtenir un aperçu plus approfondi de l'effet de chaque fonctionnalité \
              sur l'ensemble de données avec un graphique de dépendance.")
    st.write("Le dependence plot permet d’analyser les variables deux par deux en suggérant une possiblité d’observation des interactions.\
              Le scatter plot représente une dépendence entre une variable (en x) et les shapley values (en y) \
              colorée par la variable la plus corrélées.")

    ################################################################################
    # Création et affichage du sélecteur des variables et des graphs de dépendance #
    ################################################################################
    liste_variables = lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns.to_list()

    col1, col2, = st.columns(2) # division de la largeur de la page en 2 pour diminuer la taille du menu déroulant
    with col1:
        ID_var = st.selectbox("*Veuillez sélectionner une variable à l'aide du menu déroulant 👇*", 
                                (liste_variables))
        st.write("Vous avez sélectionné la variable :", ID_var)

fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_subplot(121)
shap.dependence_plot(ID_var, 
                    calcul_valeurs_shap()[1], 
                    lecture_X_test_clean().drop(labels="sk_id_curr", axis=1), 
                    interaction_index=None,
                    alpha = 0.5,
                    x_jitter = 0.5,
                    title= "Graphique de Dépendance",
                    ax=ax1,
                    show = False)
ax2 = fig.add_subplot(122)
shap.dependence_plot(ID_var, 
                    calcul_valeurs_shap()[1], 
                    lecture_X_test_clean().drop(labels="sk_id_curr", axis=1), 
                    interaction_index='auto',
                    alpha = 0.5,
                    x_jitter = 0.5,
                    title= "Graphique de Dépendance et Intéraction",
                    ax=ax2,
                    show = False)
fig.tight_layout()
st.pyplot(fig)
























