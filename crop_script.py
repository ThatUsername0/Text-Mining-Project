def friendsname(w):
    friends = [
        "Phoebe",
        "Chandler",
        "Ross",
        "Monica",
        "Rachel",
        "Joey",
        "Geller",
        "Pheebs",
        "Bing",
        "Tribbiani",
        "Buffay",
    ]
    for friend in friends:
        if w.startswith(friend):
            return True
    return False

def climax_scene(scenes, start_from_percentage=40):
    if len(scenes) <= 1:
        return [], -1  # Not enough scenes to process

    scenes_concentration = []
    start_index = int(len(scenes) * (start_from_percentage / 100))
    scenes = scenes[start_index:]  # Start from the specified percentage

    for scene in scenes:
        charactercount = 0
        total_words = len(scene)
        for w in scene:
            if friendsname(w) and len(w) >= 4:
                charactercount += 1
        concentration = (charactercount / total_words) * 100 if total_words > 0 else 0
        scenes_concentration.append(concentration)

    max_index = 0
    max_concentration = 0
    for idx, concentration in enumerate(scenes_concentration):
        if concentration > max_concentration:
            max_index = idx
            max_concentration = concentration

    return scenes[max_index], max_index + start_index

def find_climax_positions(data, start_from_percentage=40, verbose=True):
    climax_positions = []
    for episode in data["Tokenized_Scenes"]:
        if len(episode) <= 1:
            climax_positions.append(0)  # Not enough scenes to process
            continue

        word_count = sum(len(scene) for scene in episode)
        climax_scene_text, climax_pos = climax_scene(episode, start_from_percentage)
        if climax_pos == -1:
            climax_positions.append(0)
            continue

        climax_word_count = sum(len(scene) for scene in episode[:climax_pos + 1])
        climax_percentage = (climax_word_count / word_count) * 100
        climax_positions.append(climax_percentage)

        if verbose:
            # Print the scene with the highest concentration for testing
            print(f"\n\nHighest concentration scene for episode {len(climax_positions)}: {climax_scene_text}")

    return climax_positions