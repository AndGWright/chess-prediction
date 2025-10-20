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

df=df[df['Event'].str.contains('tournament')]
df['game_length']=df['TimeControl'].map(lambda x: int(x.split('+', 1)[0]))
df['increment']=df['TimeControl'].map(lambda x: int(x.split('+', 1)[1]))
df=df.loc[(df['Result'] == '1-0') | (df['Result'] == '0-1')]
df.reset_index(drop=True, inplace=True)
df['winner']=df.apply(lambda row: 1 if row['Result']=='1-0' else 0, axis=1)
df['rating_diff']=df.apply(lambda row: row['WhiteElo']-row['BlackElo'], axis=1)

# remove all text after : or | characters (typically used to indicate variations)
def group_openings(string):
    string = string.split(':', 1)[0].split('|', 1)[0].strip()
    match = re.match(r'^(.*?(?:Defense|Attack|Game|Opening|Gambit|Countergambit)\b)', string)
    if match:
        return match.group(1)
    else:
        return string
    

df['opening_group'] = df['Opening'].map(group_openings).str.replace("'", "")



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
    df[numeric_features + categorical_features], 
    df["winner"], 
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