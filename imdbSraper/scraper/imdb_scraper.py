from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_top_250_movies():
    options = webdriver.ChromeOptions()
    # options.add_argument("--headless")  # Optionnel : pas de fenêtre
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    url = "https://www.imdb.com/chart/top"
    driver.get(url)

    # Tente d'accepter le consentement cookies si présent
    try:
        consent_btn = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button#onetrust-accept-btn-handler"))
        )
        consent_btn.click()
        print("Consentement cookies accepté.")
    except Exception:
        print("Pas de consentement cookies à accepter.")

    # Dump la page pour debug
    with open("imdb_debug.html", "w", encoding="utf-8") as f:
        f.write(driver.page_source)

    # Attente explicite du tableau (ancienne structure)
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "tbody.lister-list tr"))
        )
    except Exception as e:
        print("Le tableau n'a pas chargé (ancienne structure) :", e)

    movies = []
    rows = driver.find_elements(By.CSS_SELECTOR, "tbody.lister-list tr")
    print(f"DEBUG: {len(rows)} films trouvés (ancienne structure).")

    # Si rien trouvé, tente la nouvelle structure
    if len(rows) == 0:
        rows = driver.find_elements(By.CSS_SELECTOR, "li.ipc-metadata-list-summary-item")
        print(f"DEBUG: {len(rows)} films trouvés (nouvelle structure).")
        for row in rows:
            try:
                title = row.find_element(By.CSS_SELECTOR, "h3.ipc-title__text").text
                year = row.find_elements(By.CSS_SELECTOR, "span.cli-title-metadata-item")[0].text
                try:
                    duration = row.find_elements(By.CSS_SELECTOR, "span.cli-title-metadata-item")[1].text
                except Exception:
                    duration = None
                rating = row.find_element(By.CSS_SELECTOR, "span.ipc-rating-star--rating").text.replace(',', '.')
                link = row.find_element(By.CSS_SELECTOR, "a.ipc-title-link-wrapper").get_attribute("href")
                # Aller sur la page du film pour extraire les infos détaillées
                driver2 = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
                driver2.get(link)
                try:
                    image = driver2.find_element(By.CSS_SELECTOR, 'img.ipc-image').get_attribute('src')
                except Exception:
                    image = None
                # Genre (encore plus robuste)
                try:
                    genre = ', '.join([g.text for g in driver2.find_elements(By.CSS_SELECTOR, 'div.ipc-chip-list__scroller span.ipc-chip__text')])
                    if not genre:
                        genre = ', '.join([g.text for g in driver2.find_elements(By.CSS_SELECTOR, 'div[data-testid="genres"] a.ipc-chip__text')])
                    if not genre:
                        genre = ', '.join([g.text for g in driver2.find_elements(By.CSS_SELECTOR, 'a.ipc-chip__text')])
                except Exception:
                    genre = None
                # Résumé (plus robuste)
                try:
                    resume = driver2.find_element(By.CSS_SELECTOR, 'span[data-testid="plot-l"]').text
                except Exception:
                    try:
                        resume = driver2.find_element(By.CSS_SELECTOR, 'span[data-testid="plot-xl"]').text
                    except Exception:
                        resume = None
                # Réalisateur (plus robuste)
                try:
                    realisateur = ', '.join([
                        a.text for a in driver2.find_elements(By.CSS_SELECTOR, 'li[data-testid="title-pc-principal-credit"] span.ipc-metadata-list-item__label + div ul li a')
                        if 'dr' in a.get_attribute('href') or 'director' in a.get_attribute('href')
                    ])
                    if not realisateur:
                        # fallback: premier crédit
                        realisateur = driver2.find_elements(By.CSS_SELECTOR, 'li[data-testid="title-pc-principal-credit"]')[0].find_element(By.CSS_SELECTOR, 'a').text
                except Exception:
                    realisateur = None
                # Scénario (plus robuste)
                try:
                    scenario = ', '.join([
                        a.text for a in driver2.find_elements(By.CSS_SELECTOR, 'li[data-testid="title-pc-principal-credit"]')[1].find_elements(By.CSS_SELECTOR, 'a')
                    ])
                except Exception:
                    scenario = None
                try:
                    casting = ', '.join([c.text for c in driver2.find_elements(By.CSS_SELECTOR, 'a[data-testid^="title-cast-item__actor"]')][:5])
                except Exception:
                    casting = None
                driver2.quit()
            except Exception:
                title = year = rating = link = duration = image = genre = resume = realisateur = scenario = casting = None
            movies.append({
                "title": title,
                "year": year,
                "duration": duration,
                "rating": float(rating) if rating else None,
                "link": link,
                "image": image,
                "genre": genre,
                "resume": resume,
                "realisateur": realisateur,
                "scenario": scenario,
                "casting": casting
            })
    else:
        for row in rows:
            try:
                title_column = row.find_element(By.CSS_SELECTOR, "td.titleColumn")
                title = title_column.find_element(By.TAG_NAME, "a").text
                year = title_column.find_element(By.CLASS_NAME, "secondaryInfo").text.strip("()")
                # Durée non disponible dans la table Top 250 IMDb (ancienne structure)
                duration = None
                rating = row.find_element(By.CSS_SELECTOR, "td.ratingColumn.imdbRating strong").text.replace(',', '.')
                link = title_column.find_element(By.TAG_NAME, "a").get_attribute("href")
            except Exception:
                title = year = rating = link = duration = None
            movies.append({
                "title": title,
                "year": year,
                "duration": duration,
                "rating": float(rating) if rating else None,
                "link": link
            })
    driver.quit()
    return movies