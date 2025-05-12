# IMDb Top 250 Scraper & Dashboard

Ce projet permet de scraper les informations détaillées des 250 meilleurs films IMDb, de les stocker dans une base MongoDB, puis de les explorer et visualiser via une interface web moderne (Streamlit).

## Fonctionnalités principales

- **Scraping IMDb** :
  - Récupère titre, année, durée, note, genres, résumé, réalisateur, scénaristes, casting principal, affiche, lien IMDb…
  - Gère les différentes structures HTML IMDb 
  - Gère le consentement cookies automatiquement.
  - Correction automatique des liens d'images pour un affichage parfait.

- **Stockage MongoDB** :
  - Les films sont stockés dans la base `imdb_scraper`, collection `top_250_movies`.

- **Dashboard interactif (Streamlit)** :
  - Recherche, filtres dynamiques, favoris, suggestions de films similaires.
  - Visualisations : histogrammes, boxplots, nuages de mots, heatmaps, analyses par genre, décennie, réalisateur, etc.
  - Export CSV/Excel des résultats filtrés.
  - Analyse de sentiment des synopsis.


## Structure du projet

imdbSraper/
├── app_streamlit.py         # Dashboard interactif Streamlit
├── main.py                 # Script principal pour lancer le scraping et l'insertion en base
├── requirements.txt        # Dépendances Python
├── database/
│   └── mongo.py            # Fonctions MongoDB (connexion, insertion, lecture)
├── scraper/
│   └── imdb_scraper.py     # Scraper Selenium IMDb
└── templates/
    └── index.html          # Template HTML Flask

## Installation

1. **Cloner le projet**

```bash
git clone <repo_url>
cd imdbSraper
```

2. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

3. **Lancer MongoDB**

Assurez-vous que MongoDB tourne sur `localhost:27017` (par défaut).

## Utilisation

### 1. Scraper et insérer les films en base

```bash
python main.py
```
- Scrape tous les films et insère les résultats dans MongoDB.
- Un fichier `imdb_debug.html` est généré pour debug si besoin.

### 2. Lancer le dashboard interactif (Streamlit)

```bash
streamlit run app_streamlit.py
```
- Accédez à l'interface sur http://localhost:8501

### 3. (Optionnel) Lancer le mini-front Flask

```bash
python front.py
```
- Accédez à l'interface sur http://localhost:5000

## Personnalisation

- **Nombre de films** : Par défaut, tout le Top 250 est scrappé. Pour tester plus vite, limitez le nombre de films dans le code.
- **Données supplémentaires** : Le scraper peut être adapté pour extraire d'autres champs IMDb.
- **Export** : Les données filtrées peuvent être exportées en CSV ou Excel depuis le dashboard.

## Dépendances principales
- selenium
- webdriver-manager
- pymongo
- flask
- streamlit
- pandas
- plotly, seaborn, matplotlib, wordcloud, textblob, scikit-learn

## Conseils & limitations
- Le scraping IMDb peut être lent (1 à 2 min pour 250 films) car chaque page film est visitée.
- Si vous êtes bloqué (captcha, page vide), essayez sans le mode headless ou changez d'IP.

