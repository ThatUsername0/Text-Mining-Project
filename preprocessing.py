import re
import pandas as pd
from transformers import T5Tokenizer

tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)

def preprocess_script(script):
    # Remove non-dialogue text like scene descriptions and stage directions
    script = re.sub(r'\(.*?\)', '', script)
    
    # Split the script into scenes based on regex pattern
    scenes = re.split(r'\[Scene:.*?\]', script)
    
    cleaned_scenes = []
    for scene in scenes:
        if scene.strip():
            # Remove names and colons before dialogue
            scene_lines = [re.sub(r'^[^:]+:\s*', '', line).strip() for line in scene.split('\n') if line.strip()]
            cleaned_scene = ' '.join(scene_lines)
            cleaned_scenes.append(cleaned_scene)
    
    return cleaned_scenes

def preprocess(data, train_model=False):
    data['Script'] = data['Script'].astype(str)
    data['Description'] = data['Description'].astype(str)
    
    # Apply the custom script preprocessing to partition scripts into scenes
    data['Scenes'] = data['Script'].apply(preprocess_script)

    if train_model:
        # Flatten the scenes into a single list and tokenize
        all_scenes = [scene for scenes in data['Scenes'] for scene in scenes]
        tokenized_scenes = tokenizer(all_scenes, padding='max_length', truncation=True, max_length=512)
        
        # Tokenize the descriptions
        tokenized_descriptions = tokenizer(list(data['Description']), padding='max_length', truncation=True, max_length=512)
        return data, tokenized_scenes, tokenized_descriptions
    
    return data