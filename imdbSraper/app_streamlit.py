import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from database.mongo import get_all_movies
from io import BytesIO
import base64
import re
import requests
from textblob import TextBlob

st.set_page_config(page_title="IMDb Top 250 Dashboard", layout="wide")

# Chargement des données depuis MongoDB
movies = get_all_movies()
df = pd.DataFrame(movies)

if df.empty:
    st.warning("Aucune donnée trouvée dans la base MongoDB.")
    st.stop()

def parse_duration(d):
    if pd.isnull(d):
        return None
    if isinstance(d, (int, float)):
        return int(d)
    match = re.match(r"(?:(\d+)h)?\s*(\d+)?m?", str(d).strip())
    if match:
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return None

def get_poster_url(title, year=None):
    api_key = "demo"  
    url = f"http://www.omdbapi.com/?t={title}&apikey={api_key}"
    if year:
        url += f"&y={year}"
    try:
        r = requests.get(url)
        data = r.json()
        if (data.get('Poster') and data['Poster'] != 'N/A'):
            return data['Poster']
    except Exception:
        pass
    return "https://via.placeholder.com/200x300?text=No+Poster"

def get_youtube_trailer_url(title, year=None):
    query = f"{title} trailer"
    if year:
        query += f" {year}"
    return f"https://www.youtube.com/results?search_query={requests.utils.quote(query)}"

def sentiment_synopsis(s):
    if not isinstance(s, str) or not s.strip():
        return 0
    return TextBlob(s).sentiment.polarity

def get_similar_movies(selected_film, df, n=5):
    if 'genres' not in selected_film or 'genres' not in df.columns:
        return pd.DataFrame()
    genres = set(str(selected_film['genres']).split(','))
    sim = df[df['title'] != selected_film['title']].copy()
    sim['common_genres'] = sim['genres'].apply(lambda g: len(genres & set(str(g).split(','))))
    return sim.sort_values(['common_genres','rating'], ascending=[False,False]).head(n)

def plot_wordcloud(text, title):
    wc = WordCloud(width=800, height=300, background_color='white').generate(' '.join(text))
    fig, ax = plt.subplots(figsize=(8,3))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    st.caption(title)

# Nettoyage des colonnes principales
if 'rating' in df.columns:
    df = df[df['rating'].notnull() & (df['rating'] != 'N/A')]
    df['rating'] = df['rating'].astype(float)
if 'duration' in df.columns:
    df['duration'] = df['duration'].apply(parse_duration)
    df = df[df['duration'].notnull()]
    df['duration'] = df['duration'].astype(int)
if 'year' in df.columns:
    df = df[df['year'].notnull() & (df['year'] != 'N/A')]
    df['year'] = df['year'].astype(int)

rename_dict = {
    'genre': 'genres',
    'resume': 'synopsis',
    'realisateur': 'director',
    'casting': 'actors',
    'link': 'imdb_link',
    'image': 'poster_url',
}
df.rename(columns=rename_dict, inplace=True)

# --- Favoris ---
if 'favoris' not in st.session_state:
    st.session_state['favoris'] = set()

st.title("🎬 IMDb Top 250 - Dashboard interactif ")

# Bande déroulante (marquee) avec tous les titres de films
titres = df['title'].dropna().unique()
marquee_html = f"""
<div style='background:#22223b;color:#f2e9e4;padding:8px 0;font-size:18px;white-space:nowrap;overflow:hidden;'>
  <marquee behavior='scroll' direction='left' scrollamount='7'>
    {' | '.join(titres)}
  </marquee>
</div>
"""
st.markdown(marquee_html, unsafe_allow_html=True)

# Sélection d'un film (dropdown)
st.subheader("Sélectionnez un film pour voir ses caractéristiques")
selected_title = st.selectbox("Choisissez un film", titres)
film = df[df['title']==selected_title].iloc[0]

# Affichage des caractéristiques du film
colA, colB = st.columns([1,2])
with colA:
    poster_url = film.get('poster_url') or film.get('image')
    if poster_url and poster_url.startswith('//'):
        poster_url = 'https:' + poster_url
    if poster_url and not poster_url.startswith('http'):
        poster_url = 'https://www.imdb.com' + poster_url
    match = re.search(r'(https?:[^\\s]+?\.(?:jpg|jpeg|png))', poster_url or '', re.IGNORECASE)
    if match:
        poster_url = match.group(1)
    st.image(poster_url, width=200, caption="Affiche du film")
    if st.button("⭐ Ajouter/Retirer des favoris"):
        if film['title'] in st.session_state['favoris']:
            st.session_state['favoris'].remove(film['title'])
        else:
            st.session_state['favoris'].add(film['title'])
    if film['title'] in st.session_state['favoris']:
        st.success("Ce film est dans vos favoris !")
with colB:
    st.markdown(f"**Titre :** {film['title']}")
    st.markdown(f"**Année :** {film['year']}")
    st.markdown(f"**Note :** {film['rating']}")
    if 'genres' in film:
        st.markdown(f"**Genres :** {film['genres']}")
    if 'director' in film:
        st.markdown(f"**Réalisateur :** {film['director']}")
    if 'duration' in film:
        st.markdown(f"**Durée :** {film['duration']} min")
    if 'synopsis' in film:
        st.markdown(f"**Synopsis :** {film['synopsis']}")
        sentiment = sentiment_synopsis(film['synopsis'])
        st.markdown(f"**Sentiment du synopsis :** {'😊' if sentiment>0.2 else '😐' if sentiment>-0.2 else '😞'} (score : {sentiment:.2f})")
    if 'imdb_link' in film:
        st.markdown(f"[Lien IMDb]({film['imdb_link']})")
    trailer_url = get_youtube_trailer_url(film['title'], film['year'])
    st.markdown(f"[Voir la bande-annonce sur YouTube]({trailer_url})")

# Suggestions de films similaires
st.markdown("**Suggestions de films similaires :**")
sim = get_similar_movies(film, df)
if not sim.empty:
    sim['year'] = sim['year'].astype(int) 
    st.dataframe(sim[['title','year','rating','genres']].head(5))
else:
    st.info("Aucune suggestion disponible.")

# Affichage des favoris
if st.session_state['favoris']:
    st.subheader("⭐ Vos films favoris")
    favs = df[df['title'].isin(st.session_state['favoris'])]
    st.dataframe(favs[['title','year','rating','genres','director','duration']] if 'genres' in favs.columns and 'director' in favs.columns else favs)

# --- PAGE ANALYSES ---
st.markdown("---")
st.header("📊 Analyses et visualisations ")

# Statistiques globales
col1, col2, col3 = st.columns(3)
col1.metric("Note moyenne", round(df['rating'].mean(), 2))
col2.metric("Durée moyenne (min)", int(df['duration'].mean()) if 'duration' in df else '-')
col3.metric("Année médiane", int(df['year'].median()) if 'year' in df else '-')

# Genres et réalisateurs les plus fréquents
st.subheader("Genres et réalisateurs les plus fréquents")
if 'genres' in df.columns:
    all_genres = df['genres'].dropna().str.split(',').explode().str.strip()
    top_genres = all_genres.value_counts().head(10)
    fig_genres = px.bar(top_genres, x=top_genres.index, y=top_genres.values, labels={'x':'Genre','y':'Nombre'})
    st.plotly_chart(fig_genres, use_container_width=True)
if 'director' in df.columns:
    top_directors = df['director'].value_counts().head(10)
    fig_directors = px.bar(top_directors, x=top_directors.index, y=top_directors.values, labels={'x':'Réalisateur','y':'Nombre'})
    st.plotly_chart(fig_directors, use_container_width=True)

# Visualisations interactives
st.subheader("Visualisations interactives")
fig_hist = px.histogram(df, x='rating', nbins=20, title='Répartition des notes')
st.plotly_chart(fig_hist, use_container_width=True)
if 'year' in df.columns:
    df['decade'] = (df['year']//10)*10
    fig_decade = px.bar(df['decade'].value_counts().sort_index(), labels={'value':'Nombre de films','index':'Décennie'}, title='Répartition par décennie')
    st.plotly_chart(fig_decade, use_container_width=True)

if 'genres' in df.columns and 'duration' in df.columns:
    genre_duration = df[['genres','duration']].dropna().copy()
    genre_duration = genre_duration.assign(genre=genre_duration['genres'].str.split(',')).explode('genre')
    genre_duration['genre'] = genre_duration['genre'].str.strip()
    mean_duration = genre_duration.groupby('genre')['duration'].mean().sort_values(ascending=False).head(10)
    fig_dur = px.bar(mean_duration, x=mean_duration.index, y=mean_duration.values, labels={'x':'Genre','y':'Durée moyenne'}, title='Durée moyenne par genre')
    st.plotly_chart(fig_dur, use_container_width=True)

# Analyses avancées
st.subheader("Analyses avancées")
if 'director' in df.columns:
    top5_dir = df.groupby('director')['rating'].mean().sort_values(ascending=False).head(5)
    st.write("**Top 5 réalisateurs par note moyenne**")
    st.bar_chart(top5_dir)
    st.dataframe(df[df['director'].isin(top5_dir.index)][['title','director','rating']].sort_values('rating',ascending=False))
if 'genres' in df.columns:
    st.write("**Films les plus populaires par genre**")
    for genre in all_genres.value_counts().head(3).index:
        st.write(f"*{genre}*")
        st.dataframe(df[df['genres'].str.contains(genre)][['title','rating','year','director']].sort_values('rating',ascending=False).head(5))
if 'genres' in df.columns:
    st.write("**Nuage de mots des genres**")
    plot_wordcloud(all_genres, "Genres les plus présents")
if 'director' in df.columns:
    st.subheader("Nuage de mots des réalisateurs")
    plot_wordcloud(df['director'].dropna().astype(str), "Réalisateurs les plus présents")
if 'actors' in df.columns:
    st.subheader("Nuage de mots des acteurs")
    actors = df['actors'].dropna().str.split(',').explode().str.strip()
    plot_wordcloud(actors, "Acteurs les plus présents")

# Heatmap de corrélation
st.subheader("Heatmap de corrélation")
if {'rating','duration','year'}.issubset(df.columns):
    corr = df[['rating','duration','year']].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# Comparateur de films
st.subheader("Comparateur de films")
if 'title' in df.columns:
    films = df['title'].unique()
    film1 = st.selectbox("Film 1", films, key='film1')
    film2 = st.selectbox("Film 2", films, key='film2')
    if film1 and film2:
        comp = df[df['title'].isin([film1, film2])][['title','year','rating','genres','director','duration'] if 'genres' in df.columns and 'director' in df.columns else ['title','year','rating','duration']]
        st.dataframe(comp)

# Analyse qualitative des synopsis
st.subheader("Analyse qualitative des synopsis")
if 'synopsis' in df.columns:
    df['synopsis_length'] = df['synopsis'].str.len()
    st.write(f"Longueur moyenne des synopsis : {int(df['synopsis_length'].mean())} caractères")
    longest = df.loc[df['synopsis_length'].idxmax()]
    st.write("**Synopsis le plus long :**")
    st.write(f"*{longest['title']}* ({longest['year']}) - {longest['synopsis_length']} caractères")
    st.info(longest['synopsis'])


if df.empty:
    st.warning('Le DataFrame est vide à ce stade.')
else:
    missing_cols = [col for col in ['genres','year','rating','synopsis'] if col not in df.columns]
    if missing_cols:
        st.warning(f'Colonnes manquantes pour les analyses avancées : {missing_cols}')

# --- PAGE ANALYSES CINÉMATOGRAPHIQUES ---
st.markdown("---")

st.header("Analyses cinématographiques ")

# 1. Tendances cinématographiques par genre (évolution par décennie)
if 'genres' in df.columns and 'year' in df.columns:
    st.subheader("Tendances des genres par décennie")
    st.markdown(
        """
        _Ce graphique montre l'évolution de la popularité de chaque genre de film au fil des décennies. Il permet d'identifier les genres qui gagnent ou perdent en importance dans le temps._
        """
    )
    df['decade'] = (df['year']//10)*10
    genre_decade = df[['genres','decade']].dropna().copy()
    genre_decade = genre_decade.assign(genre=genre_decade['genres'].str.split(',')).explode('genre')
    genre_decade['genre'] = genre_decade['genre'].str.strip()
    pop_genre_decade = genre_decade.groupby(['decade','genre']).size().reset_index(name='count')
    fig = px.line(pop_genre_decade, x='decade', y='count', color='genre', title='Popularité des genres par décennie')
    st.plotly_chart(fig, use_container_width=True)

# 2. Performances des films par genre (note moyenne et volume)
if 'genres' in df.columns and 'rating' in df.columns:
    st.subheader("Performances des genres (note moyenne et volume)")
    st.markdown(
        """
        _Ce graphique compare la note moyenne et le nombre de films pour chaque genre. Il permet de repérer les genres les mieux notés et ceux qui sont les plus représentés dans le Top 250._
        """
    )
    genre_ratings = df[['genres','rating']].dropna().copy()
    genre_ratings = genre_ratings.assign(genre=genre_ratings['genres'].str.split(',')).explode('genre')
    genre_ratings['genre'] = genre_ratings['genre'].str.strip()
    perf = genre_ratings.groupby('genre').agg({'rating':'mean', 'genre':'count'}).rename(columns={'rating':'Note moyenne','genre':'Nombre de films'}).sort_values('Note moyenne',ascending=False)
    st.dataframe(perf)
 
# 4. Carrières des acteurs et réalisateurs
if 'actors' in df.columns or 'director' in df.columns:
    st.subheader("Étude de carrière : acteur ou réalisateur")
    st.markdown(
        """
        _Sélectionnez un acteur ou un réalisateur pour visualiser l'évolution de sa carrière dans le Top 250 : nombre de films et évolution des notes au fil des années._
        """
    )
    choix = st.radio("Choisissez :", ["Acteur/Actrice", "Réalisateur"], horizontal=True)
    if choix == "Acteur/Actrice" and 'actors' in df.columns:
        all_actors = df['actors'].dropna().str.split(',').explode().str.strip().unique()
        acteur = st.selectbox("Sélectionnez un acteur/actrice", sorted(all_actors), key='acteur_carreer')
        films_acteur = df[df['actors'].str.contains(acteur, na=False)]
        st.write(f"Nombre de films dans le Top 250 : {len(films_acteur)}")
        st.dataframe(films_acteur[['title','year','rating','genres','director']])
        if len(films_acteur)>1:
            fig = px.line(films_acteur.sort_values('year'), x='year', y='rating', title=f'Évolution des notes pour {acteur}')
            st.plotly_chart(fig, use_container_width=True)
    elif choix == "Réalisateur" and 'director' in df.columns:
        all_directors = df['director'].dropna().unique()
        real = st.selectbox("Sélectionnez un réalisateur", sorted(all_directors), key='real_carreer')
        films_real = df[df['director']==real]
        st.write(f"Nombre de films dans le Top 250 : {len(films_real)}")
        st.dataframe(films_real[['title','year','rating','genres']])
        if len(films_real)>1:
            fig = px.line(films_real.sort_values('year'), x='year', y='rating', title=f'Évolution des notes pour {real}')
            st.plotly_chart(fig, use_container_width=True)

# 5. Prédiction des tendances futures (régression sur le nombre de films par genre)
if 'genres' in df.columns and 'year' in df.columns:
    st.subheader("Prédiction : évolution future du nombre de films par genre")
    st.markdown(
        """
        _Ce graphique utilise une régression linéaire pour prédire le nombre de films d'un genre donné dans la prochaine décennie, en se basant sur l'évolution passée._
        """
    )
    import numpy as np
    from sklearn.linear_model import LinearRegression
    genre_decade = df[['genres','decade']].dropna().copy()
    genre_decade = genre_decade.assign(genre=genre_decade['genres'].str.split(',')).explode('genre')
    genre_decade['genre'] = genre_decade['genre'].str.strip()
    genre_select_pred = st.selectbox("Choisissez un genre à prédire", sorted(genre_decade['genre'].unique()), key='genre_pred')
    data_pred = genre_decade[genre_decade['genre']==genre_select_pred].groupby('decade').size().reset_index(name='count')
    if len(data_pred)>1:
        X = data_pred['decade'].values.reshape(-1,1)
        y = data_pred['count'].values
        model = LinearRegression().fit(X, y)
        next_decade = np.array([[data_pred['decade'].max()+10]])
        pred = model.predict(next_decade)[0]
        fig_pred = px.line(data_pred, x='decade', y='count', title=f"Évolution passée et prédiction ({genre_select_pred})")
        fig_pred.add_scatter(x=[next_decade[0][0]], y=[pred], mode='markers', marker=dict(color='red', size=12), name='Prédiction')
        st.plotly_chart(fig_pred, use_container_width=True)
        st.info(f"Projection pour la décennie {next_decade[0][0]} : {int(pred):d} films {genre_select_pred} (tendance linéaire)")

# --- TOP 10 GENRES : Boxplot des notes ---
if 'genres' in df.columns and 'rating' in df.columns:
    st.subheader("Boxplot des notes par genre (Top 10)")
    st.markdown(
        """
        _Ce graphique montre la distribution des notes pour les 10 genres les plus représentés dans le Top 250. Il permet de comparer la dispersion et la médiane des notes selon le genre._
        """
    )
    genre_ratings = df[['genres','rating']].dropna().copy()
    genre_ratings = genre_ratings.assign(genre=genre_ratings['genres'].str.split(',')).explode('genre')
    genre_ratings['genre'] = genre_ratings['genre'].str.strip()
    top10_genres = genre_ratings['genre'].value_counts().head(10).index.tolist()
    genre_ratings_top10 = genre_ratings[genre_ratings['genre'].isin(top10_genres)]
    fig_box_top10 = px.box(genre_ratings_top10, x='genre', y='rating', title='Boxplot des notes par genre (Top 10)')
    st.plotly_chart(fig_box_top10, use_container_width=True)

    # --- TOP 10 GENRES PAR NOTE MOYENNE ET VOLUME ---
    st.subheader("Top 10 genres par note moyenne et par volume")
    st.markdown(
        """
        _Ce tableau présente les 10 genres les mieux notés et les 10 genres les plus représentés (par nombre de films) dans le Top 250._
        """
    )
    perf = genre_ratings.groupby('genre').agg({'rating':'mean', 'genre':'count'}).rename(columns={'rating':'Note moyenne','genre':'Nombre de films'})
    top10_note = perf.sort_values('Note moyenne', ascending=False).head(10)
    top10_volume = perf.sort_values('Nombre de films', ascending=False).head(10)
    st.write("**Top 10 genres par note moyenne :**")
    st.dataframe(top10_note)
    st.write("**Top 10 genres par volume :**")
    st.dataframe(top10_volume)

# --- TOP 5 GENRES PAR SENTIMENT MOYEN ---
if 'genres' in df.columns and 'synopsis' in df.columns:
    st.subheader("Top 5 genres par score moyen de sentiment des synopsis")
    st.markdown(
        """
        _Ce tableau présente les 5 genres avec les synopsis les plus positifs et les 5 genres avec les synopsis les plus négatifs, selon le score moyen de sentiment._
        """
    )
    df['sentiment'] = df['synopsis'].apply(sentiment_synopsis)
    genre_sent = df[['genres','sentiment']].dropna().copy()
    genre_sent = genre_sent.assign(genre=genre_sent['genres'].str.split(',')).explode('genre')
    genre_sent['genre'] = genre_sent['genre'].str.strip()
    sent_mean = genre_sent.groupby('genre')['sentiment'].mean().sort_values(ascending=False)
    st.write("**Top 5 genres les plus positifs :**")
    st.dataframe(sent_mean.head(5))
    st.write("**Top 5 genres les plus négatifs :**")
    st.dataframe(sent_mean.tail(5))
    # Ajout du barplot pour les 5 genres les plus positifs et négatifs
    
    top5_pos = sent_mean.head(5)
    top5_neg = sent_mean.tail(5)
    sent_plot = pd.concat([top5_pos, top5_neg])
    st.subheader("Analyse de sentiment des synopsis par genre")
    st.markdown(
        """
        _Ce graphique présente le score moyen de sentiment des synopsis pour chaque genre. Un score positif indique des synopsis globalement positifs, un score négatif indique des synopsis plus sombres ou négatifs._
        """
    )
    fig_sent_ext = px.bar(sent_plot, x=sent_plot.index, y=sent_plot.values, color=sent_plot.values,
                         color_continuous_scale=[(0, "red"), (0.5, "lightgrey"), (1, "green")],
                         labels={'x':'Genre','y':'Sentiment moyen'},
                         title='Top 5 genres les plus positifs et négatifs (sentiment synopsis)')
    st.plotly_chart(fig_sent_ext, use_container_width=True)

# Filtres dynamiques
st.sidebar.header("Filtres dynamiques")
min_year, max_year = int(df['year'].min()), int(df['year'].max())
year_range = st.sidebar.slider("Année", min_year, max_year, (min_year, max_year))
min_rating, max_rating = float(df['rating'].min()), float(df['rating'].max())
rating_range = st.sidebar.slider("Note", min_rating, max_rating, (min_rating, max_rating))
genres_list = sorted(all_genres.unique()) if 'genres' in df.columns else []
genre_filter = st.sidebar.multiselect("Genre", genres_list)
title_search = st.sidebar.text_input("Recherche par titre")

filtered = df[(df['year']>=year_range[0]) & (df['year']<=year_range[1]) &
              (df['rating']>=rating_range[0]) & (df['rating']<=rating_range[1])]
if genre_filter:
    filtered = filtered[filtered['genres'].apply(lambda g: any(gen in g for gen in genre_filter) if isinstance(g, str) else False)]
if title_search:
    filtered = filtered[filtered['title'].str.contains(title_search, case=False, na=False)]

st.subheader("Tableau des films filtrés")
colonnes_possibles = ['title','year','rating','genres','director','duration']
colonnes_affichees = [col for col in colonnes_possibles if col in filtered.columns]
st.dataframe(filtered[colonnes_affichees].sort_values('rating',ascending=False))

# Export CSV
csv = filtered.to_csv(index=False).encode('utf-8')
st.download_button("Exporter les données filtrées en CSV", csv, "imdb_top250_filtre.csv", "text/csv")

# Export Excel
excel_buffer = BytesIO()
filtered.to_excel(excel_buffer, index=False)
st.download_button("Exporter en Excel", excel_buffer.getvalue(), "imdb_top250_filtre.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.sidebar.markdown("---")
st.sidebar.write("🌗 Pour le mode sombre/clair, utilisez les options de Streamlit dans le menu principal (en haut à droite)")

st.success("Dashboard présentant quelques analyses sur le Top 250 IMDb.")