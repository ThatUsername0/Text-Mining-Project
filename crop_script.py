import pandas as pd

def climax_scene(col):
    scenes_counts = []
    for scenes in col:
        for scene in scenes:
            charactercount = 0
            scene.split()
            for w in scene:
                if friendsname(w):
                    print('1')
                    charactercount +=1
            scenes_counts.append(charactercount)
    
    max_index = 0
    max_count = 0
    for idx,count in enumerate(scenes_counts):
        if count > max_count:
            max_index = idx
            max_count = count
    return scenes[max_index]

def friendsname(w):
    friends = ['Phoebe', 'Chandler', 'Ross', 'Monica', 'Rachel', 'Joey', 'Geller', 'Pheebs']
    for friend in friends:
        if w.startswith(friend):
            return 1