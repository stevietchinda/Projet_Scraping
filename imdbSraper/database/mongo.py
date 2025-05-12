from pymongo import MongoClient

def save_movies_to_db(movies):
    client = MongoClient("mongodb://localhost:27017/")
    db = client["imdb_scraper"]
    collection = db["top_250_movies"]
    
    collection.delete_many({})  # Vide la collection avant l'insertion
    collection.insert_many(movies)
def get_all_movies():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["imdb_scraper"]
    collection = db["top_250_movies"]
    return list(collection.find())
