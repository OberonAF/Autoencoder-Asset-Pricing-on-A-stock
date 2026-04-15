import pandas as pd
from pathlib import Path


data_name = []
base_dir = Path(__file__).parent / 'fire_data'
for file_path in base_dir.iterdir():
    data_name.append(file_path.relative_to(base_dir))