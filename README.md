# P7_dashboard_streamlit
Projet réalisé dans le cadre du parcours diplômant de Data Scientist d'OpenClassrooms (projet n°7)

La société financière *Prêt à dépenser* propose des crédits à la consommation pour des personnes ayant peu ou pas du tout d'historique de prêt.

## **Implémentation d'un modèle de scoring :** 

L’entreprise souhaite mettre en œuvre un outil de scoring crédit qui calcule la probabilité qu’un client rembourse son crédit, puis classifie la demande en crédit accordé ou refusé. Elle souhaite donc développer un algorithme de classification en s'appuyant sur des sources de données variées (données comportementales, données provenant d'autres institutions financières, etc.).

**L'ensemble de l'analyse et de la modélisation est disponible via des "Notebook Jupyter" dans le dossier "Notebook" de ce repository.**

Les données originales sont téléchargeables sur Kaggle à [cette adresse](https://www.kaggle.com/c/home-credit-default-risk/data)


## **Dasboard interactif réalisé avec Streamlit :**

De plus, les chargés de relation client ont fait remonter le fait que les clients sont de plus en plus demandeurs de transparence vis-à-vis des décisions d’octroi de crédit. Cette demande de transparence des clients va tout à fait dans le sens des valeurs que l’entreprise veut incarner. *Prêt à dépenser* décide donc de développer un dashboard interactif pour que les chargés de relation client puissent à la fois expliquer de façon la plus transparente possible les décisions d’octroi de crédit, mais également permettre à leurs clients de disposer de leurs informations personnelles et de les explorer facilement.

*__Spécifications du dashboard__* : Il contient les fonctionnalités suivantes :

- Permettre de visualiser le score et l’interprétation de ce score pour chaque client de façon intelligible pour une personne non experte en data science.
- Permettre de visualiser des informations descriptives relatives à un client (via un système de filtre).
- Permettre de comparer les informations descriptives relatives à un client à l’ensemble des clients ou à un groupe de clients similaires.

**Le dashboard réalisé avec Streamlit est accessible en [cliquant ici](https://isabellecontant-p7-dashboard-streamlit-01--homepage-ubegdo.streamlit.app/)**
