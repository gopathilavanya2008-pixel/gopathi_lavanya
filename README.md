# gopathi_lavanya
infosys_springboard_internship
🚀 AI-Enabled Product Recommendation Dashboard
📌 Project Overview

This project implements an AI-enabled product recommendation system using Python and Streamlit.
It analyzes a dataset of 1000 e-commerce products and provides category-based recommendations using product ratings and sales data.
The system is designed as an interactive dashboard for understanding product trends and recommending top products to users.

🎯 Objectives

Analyze large product datasets efficiently

Provide top product recommendations based on category

Visualize sales, ratings, and category insights

Build a user-friendly dashboard using Streamlit

Demonstrate real-world application of data analysis and recommendation logic

📂 Dataset Description

Total Records: 1000

Format: CSV file

Dataset Features
Column Name	Description
product_id	Unique product identifier
product_name	Name of the product
category	Product category
price	Product price
rating	User rating (1–5)
sales	Number of units sold
discount	Discount percentage
revenue	Revenue generated
purchase_date	Date of purchase
🛠️ Technology Stack

Programming Language: Python

Frontend: Streamlit

Data Processing: Pandas, NumPy

Visualization: Plotly

IDE: VS Code / Jupyter Notebook

📁 Project Structure
AI-Product-Recommendation/
│
├── app.py                 # Main Streamlit application
├── products_data.csv      # Dataset (1000 records)
├── requirements.txt       # Required Python packages
├── README.md              # Project documentation
⚙️ System Architecture

Load dataset from CSV file

Preprocess and clean data

Filter products based on user-selected category

Rank products using rating + sales

Display results through interactive dashboard

🧠 Recommendation Logic

This project uses a content-based filtering approach:

Filters products by category

Sorts them using:

Higher rating first

Higher sales count as secondary factor

Displays Top 5 recommended products

⚠️ Note: This is a logic-based recommendation system, not a machine learning model.

📊 Dashboard Features

Category selection from sidebar

Top product recommendations table

Interactive charts:

Category-wise product distribution

Traffic and engagement trends

Clean and responsive UI

▶️ How to Run the Project
Step 1: Clone the repository
git clone https://github.com/your-username/AI-Product-Recommendation.git
cd AI-Product-Recommendation
Step 2: Install dependencies
pip install -r requirements.txt
Step 3: Run the Streamlit app
streamlit run app.py
📈 Results

Successfully processed 1000 product records

Generated accurate top-product recommendations per category

Visualized trends using interactive charts

Achieved real-time filtering and ranking

✅ Advantages

Simple and fast recommendation logic

Easy to extend with ML models

User-friendly interface

Scalable for larger datasets

❌ Limitations

No personalized user behavior tracking

No collaborative filtering

Rule-based ranking only

🔮 Future Enhancements

Add machine learning models (Collaborative Filtering)

User login and behavior tracking

Recommendation based on purchase history

Deploy application on cloud (AWS / Azure)

Real-time data integration

📚 References

Streamlit Documentation

Pandas & NumPy Official Docs

Plotly Visualization Library

👩‍💻 Author

Lav
Student | Data Analytics & AI Enthusiast
