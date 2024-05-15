import os
import pandas as pd
from imdbscraper import descriptions_from_scripts

# Directory containing the Friends episodes scripts
scripts_dir = 'data/friends'

df = descriptions_from_scripts(scripts_dir)
print(df.head())