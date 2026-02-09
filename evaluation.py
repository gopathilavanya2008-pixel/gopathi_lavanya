import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

#Load data
df=pd.read_csv("infosys milestone1 csv.file.csv")


#Clean data
df = df[df['rating'] > 0]

#Encode IDs
user_encoder = LabelEncoder()
product_encoder = LabelEncoder()

df['user_encoded'] = user_encoder.fit_transform(df['user_id'])
df['product_encoded'] = product_encoder.fit_transform(df['product_id'])

#Create user-item matrix
user_item_matrix = df.pivot_table(
    index='user_encoded',
    columns='product_encoded',
    values='rating',
    fill_value=0
)

#Train item-based KNN
model = NearestNeighbors(
    metric='cosine',
    algorithm='brute',
    n_neighbors=3
)
model.fit(user_item_matrix.T)

# Evaluation
actual = []
predicted = []

for user in user_item_matrix.index:
    for item in user_item_matrix.columns:
        actual_rating = user_item_matrix.loc[user, item]

        if actual_rating > 0:
            distances, indices = model.kneighbors(
                user_item_matrix.T.loc[[item]]
            )

            neighbor_items = indices.flatten()
            predicted_rating = user_item_matrix.iloc[user, neighbor_items].mean()

            actual.append(actual_rating)
            predicted.append(predicted_rating)
# Metric
mae = np.mean(np.abs(np.array(actual) - np.array(predicted)))

print("Mean Absolute Error (MAE):", round(mae, 2))
