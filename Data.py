import pandas as pd
from pathlib import Path


def read_data():
    char_data = {}
    base_dir = Path(__file__).parent / 'fire_data'
    for file_path in base_dir.iterdir():
        key = str(file_path.relative_to(base_dir))[:-8]
        value = pd.read_feather(file_path)
        char_data[key] = value

    date = char_data['index']
    del char_data['index']
    stock_code = char_data['columns']
    del char_data['columns']
    return stock_code, date, char_data