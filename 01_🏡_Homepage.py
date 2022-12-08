import streamlit as st
from PIL import Image

st.set_page_config(
        page_title='Loan application scoring dashboard',
        page_icon = "🏡",
        layout="wide"
    )

# Définition de quelques styles css
st.markdown(""" 
            <style>
            body {font-family:'Roboto Condensed';}
            h1 {font-family:'Roboto Condensed';}
            p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
            .css-18e3th9 {padding-top: 1rem; 
                          padding-right: 1rem; 
                          padding-bottom: 1rem; 
                          padding-left: 1rem;}
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

# Titre et sous_titre du projet
st.markdown("""
            <p style="color:#772b58;text-align:center;font-size:2.8em;font-style:italic;font-weight:700;font-family:'Roboto Condensed';margin:0px;">
            Votre Client aura-t-il des difficultés de paiement ?</p>
            """, 
            unsafe_allow_html=True)
st.markdown("""
            <p style="color:Gray;text-align:center;font-size:1.5em;font-style:italic;font-family:'Roboto Condensed';margin:0px;">
            Isabelle Contant - OpenClassrooms Projet n°7 - Data Scientist</p>
            """, 
            unsafe_allow_html=True)
    
# Description du projet
st.markdown("""
            <p style="color:Gray;font-family:'Roboto Condensed';">
            <br><br>
            Ce Dashboard est conçu pour les chargés de relation avec la clientèle.</p>
            """, unsafe_allow_html=True)

st.markdown("""
            <p style="color:Gray;font-family:'Roboto Condensed';">
            <strong>👈 En cliquant sur la page "Score du client"</strong>, vous pourrez découvrir le score 
            qu'il a obtenu pour sa demande de prêt.<br>
            Vous y trouverez aussi une explication intelligible de la manière dont il a été calculé.<br>
            <strong>Ce score représente le pourcentage de risque qu’il ait des difficultés à rembourser son crédit</strong>.<br>
            Ce score est calculé à l'aide d'un modèle de prédiction appliqué à un ensemble de 307 511 clients dont on sait déjà 
            si ils ont eu ou non des difficultés de paiement.<br>
            Cette procédure permet de confronter les résultats du modèle de prédiction à la réalité et donc de "valider l'efficacité 
            du modèle de prédiction".<br></p>
            """, unsafe_allow_html=True)

st.write("")

st.markdown("""
            <p style="color:Gray;font-family:'Roboto Condensed';">
            <strong>👈 Sur la page "Profil du client"</strong>, vous trouverez une comparaison des informations descriptives 
            de votre client à un groupe de clients similaires.</p>
            """, unsafe_allow_html=True)

st.write("")

st.markdown("""
            <p style="color:Gray;font-family:'Roboto Condensed';">
            <strong>👈 Enfin, sur la page "Analyse globale"</strong>, vous trouverez une interprétation globale du modèle de prédiction.</p>
            """, unsafe_allow_html=True)

    
