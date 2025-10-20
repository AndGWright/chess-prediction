import re
import pandas as pd

def create_features(df):
    df = df[df['Event'].str.contains('tournament') & df['Result'].isin(['1-0','0-1'])].copy()

    df['game_length'] = df['TimeControl'].fillna('0+0').map(lambda x: int(x.split('+', 1)[0]))
    df['increment'] = df['TimeControl'].fillna('0+0').map(lambda x: int(x.split('+', 1)[1]))
    df['winner'] = (df['Result'] == '1-0').astype(int)
    df['rating_diff'] = df['WhiteElo'] - df['BlackElo']

    df.reset_index(drop=True, inplace=True)
    return df

def group_openings(string):
    string = string.split(':', 1)[0].split('|', 1)[0].strip()
    match = re.match(r'^(.*?(?:Defense|Attack|Game|Opening|Gambit|Countergambit)\b)', string)
    if match:
        return match.group(1)
    else:
        return string

def preprocess_openings(df):
    df['opening_group'] = df['Opening'].map(group_openings).str.replace("'", "")
    return df