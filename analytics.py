import pandas as pd

def generate_business_insights():

    df = pd.read_csv(r"C:\Users\Admin\Desktop\ML\Day_two - groq\Data\products.csv")

    insights = {}

    # Top Rated Product
    top_rated = df.loc[df["rating"].idxmax()]
    insights["top_rated"] = f"{top_rated['name']} ({top_rated['rating']})"

    # Most Expensive Product
    expensive = df.loc[df["price"].idxmax()]
    insights["most_expensive"] = f"{expensive['name']} (₹{expensive['price']})"

    # Average Price
    insights["avg_price"] = round(df["price"].mean(), 2)

    # Average Rating
    insights["avg_rating"] = round(df["rating"].mean(), 2)

    # Category Wise Rating
    insights["category_rating"] = (
        df.groupby("category")["rating"].mean().to_dict()
    )

    return insights, df
