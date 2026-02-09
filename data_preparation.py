import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("infosys milestone1 csv.file.csv")

print(df.head())
print(df.isnull().sum())
print(df.columns)

df.dropna(inplace=True)
df = df[df['rating'] > 0]

user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df['user_encoded'] = user_encoder.fit_transform(df['user_id'])
df['product_encoded'] = product_encoder.fit_transform(df['product_id'])

X = df[['user_encoded', 'product_encoded']]
y = df['rating']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])
print("Unique users:",
       df['user_encoded'].nunique())
print("Unique products:",
      df['product_encoded'].nunique())

