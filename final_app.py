import streamlit as st
import pandas as pd
import plotly.express as px


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Product Recommendation Dashboard",
    layout="wide"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(
        "E:\infosys springboard project\ecommerce_product_dataset.csv",   # <-- make sure your file name is EXACT
        header=None
    )

    # Assign correct column names
    df.columns = [
        "product_id",
        "product_name",
        "category",
        "price",
        "rating",
        "views",
        "sales",
        "discount",
        "revenue",
        "date"
    ]

    # Data cleaning (VERY IMPORTANT)
    df.columns = df.columns.str.strip()
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    return df


df = load_data()

# ---------------- SIDEBAR ----------------
st.sidebar.title("Product Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Data Insights",
        "Top Products",
        "User Recommendations",
        "About Application"
    ]
)

# ---------------- HOME ----------------
if page == "Home":
    st.title("AI-Enabled Product Recommendation System")
    st.write("Dashboard built using real 1000-row e-commerce dataset.")
    st.metric("Total Products", df.shape[0])
    st.metric("Total Categories", df["category"].nunique())

# ---------------- DATA INSIGHTS ----------------
elif page == "Data Insights":
    st.subheader("Category Distribution")

    cat_count = df["category"].value_counts().reset_index()
    cat_count.columns = ["Category", "Count"]

    fig1 = px.pie(
        cat_count,
        names="Category",
        values="Count",
        hole=0.4
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Average Rating by Category")

    avg_rating = df.groupby("category")["rating"].mean().reset_index()

    fig2 = px.bar(
        avg_rating,
        x="category",
        y="rating"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- TOP PRODUCTS ----------------
elif page == "Top Products":
    st.subheader("Top Products by Category")

    selected_category = st.selectbox(
        "Select Category",
        sorted(df["category"].unique())
    )

    top_products = (
        df[df["category"] == selected_category]
        .sort_values(by=["rating", "sales"], ascending=[False, False])
        .head(5)
    )

    st.dataframe(
        top_products[
            ["product_name", "price", "rating", "sales", "discount"]
        ],
        use_container_width=True
    )

# ---------------- USER RECOMMENDATIONS ----------------
elif page == "User Recommendations":
    st.subheader("Simple Recommendation Logic")

    selected_category = st.selectbox(
        "Choose Category",
        sorted(df["category"].unique())
    )

    min_rating = st.slider(
        "Minimum Rating",
        1.0, 5.0, 3.5, 0.1
    )

    recommendations = (
        df[
            (df["category"] == selected_category) &
            (df["rating"] >= min_rating)
        ]
        .sort_values(by=["sales"], ascending=False)
        .head(5)
    )

    st.success("Recommended Products")
    st.dataframe(
        recommendations[
            ["product_name", "price", "rating", "sales"]
        ],
        use_container_width=True
    )

# ---------------- ABOUT ----------------
elif page == "About Application":
    st.write("""
    About the Application

This application is an AI-enabled Product Recommendation Dashboard designed to analyze e-commerce product data and provide meaningful insights and recommendations to users. The system uses a structured dataset of 1000 products containing attributes such as product name, category, price, ratings, sales, discounts, revenue, and purchase dates.

The primary goal of the application is to help businesses and users understand product performance, identify top-selling and high-rated items, and generate personalized product recommendations based on user preferences. By visualizing data in an interactive dashboard, the application transforms raw data into actionable insights.

The dashboard allows users to:

Explore product distribution across different categories

Analyze customer ratings and sales trends

Identify top-performing products within a selected category

Receive product recommendations based on category and minimum rating criteria

The application is built using Streamlit for the frontend, Pandas for data processing, and Plotly for interactive visualizations. The recommendation logic currently follows a content-based filtering approach, leveraging product attributes such as category, rating, and sales performance to suggest relevant products.

This system can be further enhanced by integrating machine learning models, such as collaborative filtering or similarity-based recommendation techniques, to provide more personalized and accurate recommendations in real-world e-commerce platforms.
    """)

