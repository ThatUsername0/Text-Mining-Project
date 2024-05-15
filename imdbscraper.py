import requests
import os
import re
import pandas as pd
from bs4 import BeautifulSoup

def fetch_episode_description(search_query):
    search_url = f"https://www.imdb.com/find?q={search_query.replace(' ', '+')}&s=ep"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    
    response = requests.get(search_url, headers=headers)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('a', class_='ipc-metadata-list-summary-item__t')
    
    if results:
        # Click on the first result
        first_result = results[0]
        link = first_result['href'].split('?')[0]  # Remove query parameters
        if link:
            episode_url = f"https://www.imdb.com{link}plotsummary/"
            episode_page = requests.get(episode_url, headers=headers)
            episode_soup = BeautifulSoup(episode_page.text, 'html.parser')
            description_tag = episode_soup.find('div', class_='ipc-html-content-inner-div')
            if description_tag:
                return description_tag.text.strip()
    
    return f"Description not found"

def get_season_episode(filename):
    match = re.match(r'S(\d+)E(\d+)', filename)
    if match:
        season, episode = match.groups()
        return int(season), int(episode)
    return None

def descriptions_from_scripts(scripts_dir):
    # Create a list to store data
    data = []

    # List and sort the files
    files = [f for f in os.listdir(scripts_dir) if f.endswith(".txt")]
    files = [f for f in files if get_season_episode(f) is not None]
    files.sort(key=lambda f: get_season_episode(f))

    # Iterate through each text file in the directory in sorted order
    for filename in files:
        # Extract season and episode information
        script_path = os.path.join(scripts_dir, filename)
        with open(script_path, 'r', encoding='utf-8') as file:
            script = file.read()
        
        season_episode = filename.split()[0]  # e.g., 'S01E01'
        season = int(season_episode[1:3])
        episode = int(season_episode[4:6])
        
        # Construct search query for IMDb
        episode_title = ' '.join(filename.split()[1:]).replace('.txt', '')
        search_query = f"Friends {season_episode} The One {episode_title}"
        
        # Fetch episode description from IMDb
        description = fetch_episode_description(search_query)
        
        print(f"season: {season}, ep: {episode}\ndesc: {description}\n")
        # Append the data to the list
        data.append({
            'Season': season,
            'Episode': episode,
            'Script': script,
            'Description': description
        })

    # Return a DataFrame made from the data
    return pd.DataFrame(data)