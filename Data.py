import pandas as pd
import warnings
from pathlib import Path
from typing import Tuple, Dict


def read_data() -> Tuple[pd.Series, pd.Series, Dict]:
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


def risk_free_rate() -> pd.DataFrame:
    warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')
    list = []
    base_dir = Path(__file__).parent / 'risk_free_rate'
    for file_path in base_dir.iterdir():
        df = pd.read_excel(file_path)
        df_sort = df.sort_values(by='日期')
        value = pd.DataFrame(
            {'risk_free_rate': df_sort['1年'].values},
            index=df_sort['日期']
        )
        value.index.name = 'date'
        list.append(value)

    rf = pd.concat(list, axis=0)
    return (rf[~rf.index.duplicated(keep='last')] / 100).round(4)