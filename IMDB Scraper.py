import requests
from bs4 import BeautifulSoup

def fetch_episode_descriptions(url):
    # Send a GET request to the URL
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all elements with the specified class
    descriptions = soup.find_all(class_='sc-7193fc79-2 kpMXpM')

    # Extract and return the text from each description
    return [desc.text for desc in descriptions]

# Example URL for an IMDb TV show page (Friends S1E1)
url = 'https://www.imdb.com/title/tt0903747/episodes?season=1'
episode_descriptions = fetch_episode_descriptions(url)

for i, desc in enumerate(episode_descriptions, 1):
    print(f"Episode {i}: {desc}")
