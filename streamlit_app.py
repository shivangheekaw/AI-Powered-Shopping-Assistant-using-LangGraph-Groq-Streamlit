import streamlit as st
import pandas as pd
from graph import build_graph

# ------------------------------
# FILE PATHS (UPDATED FOR UPLOADED FILES)
# ------------------------------
PRODUCTS_PATH = r"C:\Users\Admin\Desktop\ML\Day_two - groq\Data\products.csv"
ORDERS_PATH = r"C:\Users\Admin\Desktop\ML\Day_two - groq\Data\orders.csv"
COMPLAINTS_PATH = r"C:\Users\Admin\Desktop\ML\Day_two - groq\Data\complaints.csv"

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="AI Shopping Platform", page_icon="🛍️", layout="wide")

st.title("🛍️ AI E-Commerce Assistant & Business Dashboard")

# ------------------------------
# Tabs
# ------------------------------
tab1, tab2 = st.tabs(["🛍️ Chat Assistant", "📊 Business Dashboard"])

# =====================================================
# ================== TAB 1 : CHAT =====================
# =====================================================
with tab1:

    if "agent" not in st.session_state:
        st.session_state.agent = build_graph()

    if "state" not in st.session_state:
        st.session_state.state = {
            "user_query": "",
            "user_id": "U101",
            "intent": None,
            "entities": None,
            "product_context": None,
            "recommended_products": None,
            "complaint_ticket": None,
            "final_response": None,
            "logs": None,
            "chat_history": []
        }

    # Display Chat History
    for chat in st.session_state.state["chat_history"]:
        with st.chat_message("user"):
            st.write(chat["user"])
        with st.chat_message("assistant"):
            st.write(chat["assistant"])

    user_input = st.chat_input("Ask about products, complaints, or insights...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        st.session_state.state["user_query"] = user_input

        st.session_state.state = st.session_state.agent.invoke(
            st.session_state.state
        )

        with st.chat_message("assistant"):
            st.write(st.session_state.state["final_response"])


# =====================================================
# ============== TAB 2 : BUSINESS DASHBOARD ===========
# =====================================================
with tab2:

    st.header("📊 Business Intelligence Dashboard")

    # Load Data
    products_df = pd.read_csv(PRODUCTS_PATH)
    orders_df = pd.read_csv(ORDERS_PATH)
    complaints_df = pd.read_csv(COMPLAINTS_PATH)

    # Convert order_date to datetime
    if "order_date" in orders_df.columns:
        orders_df["order_date"] = pd.to_datetime(orders_df["order_date"])

    # -------------------------------------------------
    # MERGE ORDERS WITH PRODUCTS TO GET PRICE
    # -------------------------------------------------
    merged_df = orders_df.merge(
        products_df[["product_id", "price", "category", "name"]],
        on="product_id",
        how="left"
    )

    # Assume quantity = 1 per order
    merged_df["revenue"] = merged_df["price"]

    # ------------------------------
    # KPI METRICS
    # ------------------------------
    total_revenue = merged_df["revenue"].sum()
    total_orders = len(orders_df)
    avg_order_value = total_revenue / total_orders if total_orders > 0 else 0
    total_complaints = len(complaints_df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Total Revenue", f"₹{round(total_revenue,2)}")
    col2.metric("📦 Total Orders", total_orders)
    col3.metric("🧾 Avg Order Value", f"₹{round(avg_order_value,2)}")
    col4.metric("😡 Total Complaints", total_complaints)

    st.divider()

    # ------------------------------
    # Revenue Over Time
    # ------------------------------
    if "order_date" in merged_df.columns:
        revenue_trend = (
            merged_df.groupby(merged_df["order_date"].dt.date)["revenue"]
            .sum()
        )

        st.subheader("📈 Revenue Over Time")
        st.line_chart(revenue_trend)

    # ------------------------------
    # Top Selling Products
    # ------------------------------
    st.subheader("🏆 Top Selling Products")

    top_products = (
        merged_df.groupby("name")["order_id"]
        .count()
        .sort_values(ascending=False)
        .head(5)
    )

    st.bar_chart(top_products)

    # ------------------------------
    # Category Wise Revenue
    # ------------------------------
    st.subheader("📊 Category Wise Revenue")

    category_revenue = (
        merged_df.groupby("category")["revenue"]
        .sum()
        .sort_values(ascending=False)
    )

    st.bar_chart(category_revenue)

    # ------------------------------
    # Complaint Status Breakdown
    # ------------------------------
    st.subheader("🚨 Complaint Status Breakdown")

    if "status" in complaints_df.columns:
        complaint_status = complaints_df["status"].value_counts()
        st.bar_chart(complaint_status)

    # ------------------------------
    # Rating Analysis
    # ------------------------------
    st.subheader("⭐ Average Rating by Category")

    rating_by_category = (
        products_df.groupby("category")["rating"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(rating_by_category)

    st.success("Dashboard Loaded Successfully 🚀")