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
from xplotter.insights import *

############################
# Configuration de la page #
############################
st.set_page_config(
        page_title='Profil du Client',
        page_icon = "😎",
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


if __name__ == "__main__":

    lecture_X_test_original()
    lecture_X_test_clean()
    lecture_description_variables()

    # Titre 1
    st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                1. Quel est le profil de votre client ?</h1>
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
    score_value = score.proba_classe_1.iloc[0]
    
    st.write(f"Le client dont l'identifiant est **{ID_client}** a obtenu le score de **{score_value:.1%}**.")
    st.write(f"**Il y a donc un risque de {score_value:.1%} que le client ait des difficultés de paiement.**")
    
    #st.dataframe(lecture_X_test_original())
    
    ########################################################
    # Récupération et affichage des informations du client #
    ########################################################
    data_client=lecture_X_test_original()[lecture_X_test_original().sk_id_curr == ID_client]
    
    col1, col2 = st.columns(2)
    with col1:
        # Titre H2
        st.markdown("""
                    <h2 style="color:#418b85;text-align:center;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
                    Profil socio-économique</h2>
                    """, 
                    unsafe_allow_html=True)
        st.write("")
        st.write(f"Genre : **{data_client['code_gender'].values[0]}**")
        st.write(f"Tranche d'âge : **{data_client['age_client'].values[0]}**")
        st.write(f"Ancienneté de la piède d'identité : **{data_client['anciennete_cni'].values[0]}**")
        st.write(f"Situation familiale : **{data_client['name_family_status'].values[0]}**")
        st.write(f"Taille de la famille : **{data_client['taille_famille'].values[0]}**")
        st.write(f"Nombre d'enfants : **{data_client['nbr_enfants'].values[0]}**")
        st.write(f"Niveau d'éducation : **{data_client['name_education_type'].values[0]}**")
        st.write(f"Revenu Total Annuel : **{data_client['total_revenus'].values[0]} $**")
        st.write(f"Type d'emploi : **{data_client['name_income_type'].values[0]}**")
        st.write(f"Ancienneté dans son entreprise actuelle : **{data_client['anciennete_entreprise'].values[0]}**")
        st.write(f"Type d'habitation : **{data_client['name_housing_type'].values[0]}**")
        st.write(f"Densité de la Population de la région où vit le client : **{data_client['pop_region'].values[0]}**")
        st.write(f"Evaluations de *'Prêt à dépenser'* de la région où vit le client : \
                   **{data_client['region_rating_client'].values[0]}**")
    
    with col2:
        # Titre H2
        st.markdown("""
                    <h2 style="color:#418b85;text-align:center;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
                    Profil emprunteur</h2>
                    """, 
                    unsafe_allow_html=True)
        st.write("")
        st.write(f"Type de Crédit demandé par le client : **{data_client['name_contract_type'].values[0]}**")
        st.write(f"Montant du Crédit demandé par le client : **{data_client['montant_credit'].values[0]} $**")
        st.write(f"Durée de remboursement du crédit : **{data_client['duree_remboursement'].values[0]}**")
        st.write(f"Taux d'endettement : **{data_client['taux_endettement'].values[0]}**")
        st.write(f"Score normalisé du client à partir d'une source de données externe : \
                  **{data_client['ext_source_2'].values[0]:.1%}**")
        st.write(f"Nombre de demande de prêt réalisée par le client : \
                   **{data_client['nb_demande_pret_precedente'].values[0]:.0f}**")
        st.write(f"Montant des demandes de prêt précédentes du client : \
                  **{data_client['montant_demande_pret_precedente'].values[0]} $**")
        st.write(f"Montant payé vs Montant attendu en % : **{data_client['montant_paye_vs_du'].values[0]:.1f}%**")
        st.write(f"Durée mensuelle moyenne des crédits précédents : **{data_client['cnt_instalment'].values[0]:.1f} mois**")
        st.write(f"Nombre de Crédit à la Consommation précédent du client : \
                  **{data_client['prev_contrat_type_consumer_loans'].values[0]:.0f}**")
        st.write(f"Nombre de Crédit Revolving précédent du client : \
                  **{data_client['prev_contrat_type_revolving_loans'].values[0]:.0f}**")
        st.write(f"Nombre de Crédit précédent refusé : \
                  **{data_client['prev_contrat_statut_refused'].values[0]:.0f}**")
        st.write(f"Nombre de crédits cloturés enregistrés au bureau du crédit : \
                  **{data_client['bureau_credit_actif_closed'].values[0]:.0f}**")
        st.write(f"Nombre de crédits de type *'carte de crédit'* enregistrés au bureau du crédit : \
                  **{data_client['bureau_credit_type_credit_card'].values[0]:.0f}**")
        st.write(f"Nombre d'années écoulées depuis la décision précédente : \
                  **{data_client['nb_year_depuis_decision_precedente'].values[0]:.0f} ans**")


    ###############################################################
    # Comparaison du profil du client à son groupe d'appartenance #
    ###############################################################

    # Titre 1
    st.markdown("""
                <br>
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                2. Comparaison du profil du client à celui des clients dont la probabilité de défaut de paiement est proche</h1>
                """, 
                unsafe_allow_html=True)
    st.write("Pour la définition des groupes de clients, faites défiler la page vers le bas.")
    
    # Calcul des valeurs Shap
    explainer_shap = shap.TreeExplainer(model_LGBM)
    shap_values = explainer_shap.shap_values(lecture_X_test_clean().drop(labels="sk_id_curr", axis=1))
    shap_values_df = pd.DataFrame(data=shap_values[1], columns=lecture_X_test_clean().drop(labels="sk_id_curr", axis=1).columns)
    
    df_groupes = pd.concat([y_pred_lgbm_proba_df['proba_classe_1'], shap_values_df], axis=1)
    df_groupes['typologie_clients'] = pd.qcut(df_groupes.proba_classe_1,
                                              q=5,
                                              precision=1,
                                              labels=['20%_et_moins',
                                                      '21%_30%',
                                                      '31%_40%',
                                                      '41%_60%',
                                                      '61%_et_plus'])

    # Titre H2
    st.markdown("""
                <h2 style="color:#418b85;text-align:left;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
                Comparaison de “la trajectoire” prise par la prédiction du client à celles des groupes de Clients</h2>
                """, 
                unsafe_allow_html=True)
    st.write("")

    # Moyenne des variables par classe
    df_groupes_mean = df_groupes.groupby(['typologie_clients']).mean()
    df_groupes_mean = df_groupes_mean.rename_axis('typologie_clients').reset_index()
    df_groupes_mean["index"]=[1,2,3,4,5]
    df_groupes_mean.set_index('index', inplace = True)
    
    # récupération de l'index correspondant à l'identifiant du client
    idx = int(lecture_X_test_clean()[lecture_X_test_clean()['sk_id_curr']==ID_client].index[0])

    # dataframe avec shap values du client et des 5 groupes de clients
    comparaison_client_groupe = pd.concat([df_groupes[df_groupes.index == idx], 
                                            df_groupes_mean],
                                            axis = 0)
    comparaison_client_groupe['typologie_clients'] = np.where(comparaison_client_groupe.index == idx, 
                                                          lecture_X_test_clean().iloc[idx, 0],
                                                          comparaison_client_groupe['typologie_clients'])
    # transformation en array
    nmp = comparaison_client_groupe.drop(
                      labels=['typologie_clients', "proba_classe_1"], axis=1).to_numpy()

    fig = plt.figure(figsize=(8, 20))
    st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                                nmp, 
                                feature_names=comparaison_client_groupe.drop(
                                              labels=['typologie_clients', "proba_classe_1"], axis=1).columns.to_list(),
                                feature_order='importance',
                                highlight=0,
                                legend_labels=['Client', '20%_et_moins', '21%_30%', '31%_40%', '41%_60%', '61%_et_plus'],
                                plot_color='inferno_r',
                                legend_location='center right',
                                feature_display_range=slice(None, -57, -1),
                                link='logit'))

    # Titre H2
    st.markdown("""
                <h2 style="color:#418b85;text-align:left;font-size:1.8em;font-style:italic;font-weight:700;margin:0px;">
                Constitution de groupes de clients selon leur probabilité de défaut de paiement</h2>
                """, 
                unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns(2)
    with col1:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        plot_countplot(df=df_groupes, 
                    col='typologie_clients', 
                    order=False,
                    palette='rocket_r', ax=ax1, orient='v', size_labels=12)
        plt.title("Regroupement des Clients selon leur Probabilité de Défaut de Paiement\n",
                loc="center", fontsize=16, fontstyle='italic', fontname='Roboto Condensed')
        fig1.tight_layout()
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        plot_aggregation(df=df_groupes,
                    group_col='typologie_clients',
                    value_col='proba_classe_1',
                    aggreg='mean',
                    palette="rocket_r", ax=ax2, orient='v', size_labels=12)
        plt.title("Probabilité Moyenne de Défaut de Paiement par Groupe de Clients\n",
                loc="center", fontsize=16, fontstyle='italic', fontname='Roboto Condensed')
        fig2.tight_layout()
        st.pyplot(fig2)
