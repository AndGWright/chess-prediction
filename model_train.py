import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re
from xgboost import XGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle


df=pd.read_parquet('data/train-00000-of-00064.parquet')
df_blitz=df[df['Event'].str.contains('tournament')]


df_blitz['game_length']=df_blitz['TimeControl'].map(lambda x: int(x.split('+', 1)[0]))
df_blitz['increment']=df_blitz['TimeControl'].map(lambda x: int(x.split('+', 1)[1]))
df_blitz['winner']=df_blitz.apply(lambda row: 'white' if row['Result']=='1-0' else 'black' if row['Result']=='0-1' else 'draw', axis=1)
df_blitz=df_blitz.loc[df_blitz['winner'] != 'draw']
df_blitz.reset_index(drop=True, inplace=True)
df_blitz['white_win']=df_blitz.apply(lambda row: 1 if row['winner']=='white' else 0,axis=1)
df_blitz['rating_diff']=df_blitz.apply(lambda row: row['WhiteElo']-row['BlackElo'],axis=1)

# remove all text after : or | characters (typically used to indicate variations)
def group_openings_1(string):
    if ':' in string:
        return string.split(':', 1)[0].strip()
    elif '|' in string:
        return string.split('|', 1)[0].strip()
    else:
        return string

# use regex to return only the string up to and including the following strings
def group_openings_2(string):
    pattern = r'^(.*?(?:Defense|Attack|Game|Opening|Gambit|Countergambit)\b)'
    match = re.match(pattern, string)
    if match:
        return match.group(1)
    else:
        return string
    

df_blitz['opening_group'] = df_blitz['Opening'].map(group_openings_1).map(group_openings_2)
df_blitz['opening_group'] = df_blitz['opening_group'].str.replace("'", "")



numeric_features = ['rating_diff','WhiteElo']
categorical_features = ['opening_group','increment','game_length']

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

model = XGBClassifier(max_depth= 12,
min_child_weight= 1,
gamma = 1,
subsample = 0.9,
colsample_bytree = 0.9)

pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    ("classifier", model)
])


X_train, X_test, y_train, y_test = train_test_split(
    df_blitz[numeric_features + categorical_features], 
    df_blitz["white_win"], 
    test_size=0.15, 
    random_state=42
)


pipeline.fit(X_train, y_train)


y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")

with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)