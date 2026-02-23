from flask import Flask, jsonify, request
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
app = Flask(__name__)
# Load data
df = pd.read_csv(r"C:\Users\gopat\Downloads\ecommerce_product_dataset.csv")

# Clean data
df.dropna(inplace=True)
df = df[df['Rating'] > 0]

# Features for similarity
features = df[['Price', 'Rating']]
@app.route("/")
def home():
    return jsonify({
        "message": "Product Recommendation API is running",
        "usage": "/recommend?product_id=<id>"
    })


# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Train KNN model
model = NearestNeighbors(
    metric='cosine',
    algorithm='brute',
    n_neighbors=5
)
model.fit(scaled_features)

# Recommendation function
def recommend_products(product_id, top_n=3):
    if product_id not in df['ProductID'].values:
        return []

    product_index = df[df['ProductID'] == product_id].index[0]

    distances, indices = model.kneighbors(
        scaled_features[product_index].reshape(1, -1)
    )

    recommendations = []
    for i in indices[0][1:top_n+1]:
        recommendations.append({
            "ProductID": int(df.iloc[i]['ProductID']),
            "ProductName": df.iloc[i]['ProductName'],
            "Price": float(df.iloc[i]['Price']),
            "Rating": float(df.iloc[i]['Rating'])
        })

    return recommendations

# API endpoint
@app.route("/recommend", methods=["GET"])
def recommend():
    product_id = request.args.get("product_id", type=int)

    if product_id is None:
        return jsonify({"error": "product_id is required"}), 400

    recommendations = recommend_products(product_id)

    return jsonify({
        "product_id": product_id,
        "product_name": recommendations[0]["ProductName"],
        "price": recommendations[0]["Price"],
        "rating": recommendations[0]["Rating"],
        "recommended_products": recommendations
    })

if __name__ == "__main__":
    app.run(debug=True)
