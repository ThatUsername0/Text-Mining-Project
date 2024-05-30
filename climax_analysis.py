import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imdb_scraper
import preprocessing
from tqdm import tqdm
from scipy.interpolate import make_interp_spline
from crop_script import find_climax_positions
from summarisation import summarise_script


# Directory containing the Friends episodes scripts
scripts_dir = 'data/friends'
series_name = os.path.basename(scripts_dir)
processed_data_file = f'data_processed/{series_name}_with_descriptions.csv'
preprocessed_data_file = f'data_processed/{series_name}_preprocessed.csv'

# Check if the processed data file exists
if os.path.exists(processed_data_file):
    # Load the processed data
    df = pd.read_csv(processed_data_file)
else:
    # Ensure the directory exists
    if not os.path.exists('data_processed'):
        os.makedirs('data_processed')
    # Generate descriptions and save the processed data
    df = imdb_scraper.descriptions_from_scripts(scripts_dir)
    df.to_csv(processed_data_file, index=False)

# Check if the preprocessed data file exists
if os.path.exists(preprocessed_data_file):
    # Load the preprocessed data
    preprocessed_data = pd.read_csv(preprocessed_data_file)
    preprocessed_data["Tokenized_Scenes"] = preprocessed_data["Tokenized_Scenes"].apply(eval)  # Convert string back to list
else:
    print("Preprocessing...")
    # Preprocess the data
    preprocessed_data, _, _ = preprocessing.preprocess(df)
    # Save the preprocessed data
    preprocessed_data.to_csv(preprocessed_data_file, index=False)
    print("Preprocessing done")

# Calculate the climax positions for each episode
climax_positions = find_climax_positions(preprocessed_data, start_from_percentage=40)
preprocessed_data["Climax_Position_Percentage"] = climax_positions

# Generate summaries with progress bar and print each summary
summaries = []
for idx, row in tqdm(preprocessed_data.iterrows(), total=preprocessed_data.shape[0], desc="Summarizing"):
    summary = summarise_script(row["Tokenized_Scenes"], row["Climax_Position_Percentage"])
    summaries.append(summary)
    print(f"Episode {idx + 1} Summary:\n{summary}\n")

preprocessed_data["Summary"] = summaries

# Save the data with summaries
summary_data_file = f'data_processed/{series_name}_summaries.csv'
preprocessed_data.to_csv(summary_data_file, index=False)

# Bin the climax positions
bins = np.linspace(0, 100, 20)  # 20 bins from 0% to 100%
preprocessed_data['Climax_Position_Bin'] = pd.cut(preprocessed_data["Climax_Position_Percentage"], bins)

# Calculate the frequency of each bin
climax_frequency = preprocessed_data['Climax_Position_Bin'].value_counts().sort_index()

# Convert bin intervals to midpoints for plotting
bin_midpoints = bins[:-1] + np.diff(bins) / 2
frequency_values = climax_frequency.values

# Smooth the frequency data
x_smooth = np.linspace(bin_midpoints.min(), bin_midpoints.max(), 300)
spl = make_interp_spline(bin_midpoints, frequency_values, k=3)
y_smooth = spl(x_smooth)

# Plotting the smoothed climax position frequencies
plt.figure(figsize=(12, 6))
plt.plot(x_smooth, y_smooth)
plt.xlabel('Climax Position (%)')
plt.ylabel('Frequency')
plt.title('Frequency of Climax Position Through Episodes')
plt.grid(True)
plt.show()