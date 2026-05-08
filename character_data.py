import pandas as pd
from pathlib import Path


def read_character():
    character_data = {}
    base_dir = Path(__file__).parent / 'fire_data'
    for file_path in base_dir.iterdir():
        name = str(file_path.relative_to(base_dir))[:-8]
        data = pd.read_feather(file_path)
        character_data[name] = data
    return character_data