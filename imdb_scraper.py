import requests
import os
import re
import pandas as pd
from bs4 import BeautifulSoup

def fetch_episode_description(search_query, series_name):
    """
    Fetches the episode description from IMDb for a given search query and series name.

    Args:
        search_query (str): The search query for the episode.
        series_name (str): The name of the TV series.

    Returns:
        str: The episode description or a "Description not found" message.
    """
    search_url = f"https://www.imdb.com/find?q={search_query.replace(' ', '+')}&s=ep"
    print(search_url)
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    
    # Send a request to the IMDb search page
    response = requests.get(search_url, headers=headers)
    response.raise_for_status()
    
    # Parse the HTML response
    soup = BeautifulSoup(response.text, 'html.parser')
    results = soup.find_all('li', class_='ipc-metadata-list-summary-item')

    for result in results:
        series_name_tag = result.find('a', class_='ipc-metadata-list-summary-item__li--link')
        if series_name_tag and series_name_tag.text.strip().lower() == series_name.lower():
            link = result.find('a', class_='ipc-metadata-list-summary-item__t')['href'].split('?')[0]
            if link:
                episode_url = f"https://www.imdb.com{link}plotsummary/"
                episode_page = requests.get(episode_url, headers=headers)
                episode_page.raise_for_status()
                
                episode_soup = BeautifulSoup(episode_page.text, 'html.parser')
                description_tag = episode_soup.find('div', class_='ipc-html-content-inner-div')
                if description_tag:
                    return description_tag.text.strip()
    
    return ""

def get_season_episode(filename):
    """
    Extracts the season and episode numbers from a filename.

    Args:
        filename (str): The filename to extract season and episode from.

    Returns:
        tuple: A tuple containing the season and episode numbers, or None if not matched.
    """
    match = re.match(r'S(\d+)E(\d+)', filename)
    if match:
        season, episode = match.groups()
        return int(season), int(episode)
    return None

def descriptions_from_scripts(scripts_dir):
    """
    Generates a DataFrame of episode descriptions from script files in a directory.

    Args:
        scripts_dir (str): The directory containing the script files.

    Returns:
        pd.DataFrame: A DataFrame containing season, episode, script, and description.
    """
    series_name = os.path.basename(scripts_dir)

    data = []
    files = [f for f in os.listdir(scripts_dir) if f.endswith(".txt")]
    files = [f for f in files if get_season_episode(f) is not None]
    files.sort(key=lambda f: get_season_episode(f))

    for filename in files:
        script_path = os.path.join(scripts_dir, filename)
        with open(script_path, 'r', encoding='utf-8') as file:
            script = file.read()
        
        season_episode = filename.split()[0]
        season = int(season_episode[1:3])
        episode = int(season_episode[4:6])
        
        episode_title = ' '.join(filename.split()[1:]).replace('.txt', '')
        
        # Special handling for the Friends series
        friends = "The One" if series_name.lower() == "friends" else ""
        search_query = f"{series_name} {season_episode} {friends} {episode_title}"
        
        # Fetch episode description from IMDb
        description = fetch_episode_description(search_query, series_name)
        
        print(f"season: {season}, ep: {episode}\ndesc: {description}\n")
        data.append({
            'Season': season,
            'Episode': episode,
            'Script': script,
            'Description': description
        })

    # Return the data as a DataFrame
    return pd.DataFrame(data)
