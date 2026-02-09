import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

#Load dataset
df=pd.read_csv("infosys milestone1 csv.file.csv")

# Data cleaning
df.dropna(inplace=True)
df = df[df['rating'] > 0]

# Encode user and product IDs
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df['user_encoded'] = user_encoder.fit_transform(df['user_id'])
df['product_encoded'] = product_encoder.fit_transform(df['product_id'])

# Create user-item interaction matrix
user_item_matrix = df.pivot_table(
    index='user_encoded',
    columns='product_encoded',
    values='rating',
    fill_value=0
)

# Train-test split (users)
X_train, X_test = train_test_split(
    user_item_matrix,
    test_size=0.2,
    random_state=42
)

# Build KNN model (item-based collaborative filtering)
model = NearestNeighbors(
    metric='cosine',
    algorithm='brute',
    n_neighbors=3
)

#Train model
model.fit(X_train.T)

print("model training completed successfully")