import os
import json

def get_data(mode):
    path = os.path.join(os.getcwd(), 'data', mode, 'meta_dataset.json')
    with open(path, 'r') as json_file:
        data = json.load(json_file)
    return data

