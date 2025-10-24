import pandas as pd

test_df = pd.read_csv('data\\test.csv')
train_df = pd.read_csv('data\\train.csv')

print(train_df.dtypes)
print(train_df[['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']])

def map_numeric_columns(train_df, col_name):
    train_df[col_name] = train_df[col_name].fillna(0)
    unique_values = train_df[col_name].unique()
    mapping = {x : i for i, x in enumerate(unique_values)}
    train_df[col_name] = train_df[col_name].map(mapping)

for x in ['Sex', 'Ticket', 'Cabin', 'Embarked']:
    map_numeric_columns(train_df, x)

print(train_df.head(10))

