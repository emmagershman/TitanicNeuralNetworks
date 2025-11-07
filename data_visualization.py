import pandas as pd
from sklearn.preprocessing import StandardScaler

test_df = pd.read_csv('data\\test.csv')
train_df = pd.read_csv('data\\train.csv')
COLUMNS = ['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_df = train_df[COLUMNS]


def map_numeric_columns(train_df, col_name):
    train_df[col_name] = train_df[col_name].fillna('nan')
    survival_probs = train_df.groupby(col_name)['Survived'].mean().sort_values()
    mapping = {x : i for i, x in enumerate(survival_probs.index)}
    train_df[col_name] = train_df[col_name].map(mapping)

for x in ['Sex', 'Embarked']:
    map_numeric_columns(train_df, x)

print(train_df)
# NUMERIC COLUMNS MAY HAVE NAN AS WELL
exit()

scaler = StandardScaler()
X = scaler.fit_transform(X)

print(train_df.head(10))

