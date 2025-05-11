from scraper.imdb_scraper import get_top_250_movies
from database.mongo import save_movies_to_db

def main():
    print("Scraping IMDb Top 250...")
    movies = get_top_250_movies()
    
    print(f"{len(movies)} films extraits. Insertion en base de données...")
    save_movies_to_db(movies)
    print("✅ Terminé !")

if __name__ == "__main__":
    main()
