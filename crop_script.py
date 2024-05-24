def climax_scene(scenes):
    scenes_counts = []
    for scene in scenes:
        charactercount = 0
        for w in scene:
            if friendsname(w) and len(w) >= 4:
                charactercount += 1
        scenes_counts.append(charactercount)
    
    max_index = 0
    max_count = 0
    for idx,  count in enumerate(scenes_counts):
        if count > max_count:
            max_index = idx
            max_count = count
    return scenes[max_index], max_index

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

def find_climax_positions(data):
    climax_positions = []
    for episode in data["Tokenized_Scenes"]:
        word_count = sum(len(scene) for scene in episode)
        climax_scene_text, climax_pos = climax_scene(episode)
        climax_word_count = sum(len(scene) for scene in episode[:climax_pos+1])
        climax_percentage = (climax_word_count / word_count) * 100
        climax_positions.append(climax_percentage)
    return climax_positions
