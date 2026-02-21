from langchain_groq import ChatGroq
import os
from state import GraphState
import json
import pandas as pd


print("ENV KEY:", os.getenv("GROQ_API_KEY"))
PRODUCTS_PATH = r"C:\Users\Admin\Desktop\ML\Day_two - groq\Data\products.csv"
llm = ChatGroq(
    model="llama-3.1-8b-instant",   # fast & good
    temperature=0
)


def intent_classifier_node(state: GraphState) -> GraphState:
    """
    Determines user intent and extracts entities.
    """

    user_query = state["user_query"]
    history = state.get("chat_history", [])

    conversation_context = ""

    for turn in history[-3:]:  # last 3 messages only
        conversation_context += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"

    prompt = f"""
You are an intent classification assistant for an e-commerce platform.

Classify the user's intent into ONE of the following:
- inquiry
- recommendation
- complaint
- business_insight

Use business_insight ONLY if the user is asking for analytics,
trends, performance summary, statistics, or business insights.

Also extract useful entities if present.

Return ONLY valid JSON in this format:
{{
  "intent": "<intent>",
  "entities": {{
    "category": "",
    "budget": "",
    "product": "",
    "issue": ""
  }}
}}

Conversation so far:
{conversation_context}

User message:
{user_query}

"""

    response = llm.invoke(prompt).content

    try:
        parsed = json.loads(response)
        state["intent"] = parsed.get("intent")
        state["entities"] = parsed.get("entities")
        print("The intent of the user query is " + str(state["intent"]))

    except Exception:
        # Fallback rule-based safety for business insight
        lower_query = user_query.lower()

        if any(word in lower_query for word in ["insight", "business", "analytics", "trend", "statistics"]):
            state["intent"] = "business_insight"
        else:
            state["intent"] = "inquiry"

        state["entities"] = {}

    return state



def recommendation_node(state: dict) -> dict:
    """
    Recommend top 3 products based on category and budget.
    """

    entities = state.get("entities", {})

    category = entities.get("category")
    budget = entities.get("budget")

    # Load product dataset
    df = pd.read_csv(PRODUCTS_PATH)

    # Filter by category (if provided)
    if category:
        df = df[df["category"].str.lower() == category.lower()]

    # Filter by budget (if provided)
    if budget:
        try:
            budget = int(budget)
            df = df[df["price"] <= budget]
        except:
            pass

    # Sort by popularity first, then rating
    df = df.sort_values(
        by=["popularity", "rating"],
        ascending=False
    )

    # Select top 3
    top_products = df.head(3)[
        ["product_id", "name", "price", "rating"]
    ]

    # Convert to list of dictionaries
    recommendations = top_products.to_dict(orient="records")

    state["recommended_products"] = recommendations

    return state


def inquiry_node(state: dict) -> dict:
    """
    Intelligent inquiry handler:
    - Direct product info
    - Follow-up comparison
    - Rating-based questions
    - Context-aware responses
    """

    df = pd.read_csv(PRODUCTS_PATH)

    entities = state.get("entities", {})
    product_name = entities.get("product")
    user_query = state.get("user_query", "").lower()
    previous_recs = state.get("recommended_products")

    # ---------------------------------------------------
    # 1️⃣ Direct product lookup
    # ---------------------------------------------------
    if product_name:
        match = df[df["name"].str.lower().str.contains(product_name.lower(), na=False)]

        if not match.empty:
            product_info = match.iloc[0]

            state["product_context"] = {
                "name": product_info["name"],
                "price": product_info["price"],
                "features": product_info.get("features", ""),
                "description": product_info.get("description", ""),
                "rating": product_info["rating"]
            }
            return state

    # ---------------------------------------------------
    # 2️⃣ Follow-up on previous recommendations
    # ---------------------------------------------------
    if previous_recs:

        # If user asking about rating / best / better
        if any(word in user_query for word in ["best", "better", "rating", "highest"]):
            best_product = max(previous_recs, key=lambda x: x["rating"])

            state["product_context"] = {
                "name": best_product["name"],
                "price": best_product["price"],
                "rating": best_product["rating"],
                "features": "This product has the highest rating among your previous options.",
                "description": ""
            }
            return state

        # If asking about cheapest
        if any(word in user_query for word in ["cheap", "cheapest", "lowest price"]):
            cheapest = min(previous_recs, key=lambda x: x["price"])

            state["product_context"] = {
                "name": cheapest["name"],
                "price": cheapest["price"],
                "rating": cheapest["rating"],
                "features": "This is the most affordable option from your previous results.",
                "description": ""
            }
            return state

        # Generic comparison fallback
        state["product_context"] = {
            "message": "Here are your previously recommended products. Please specify what you'd like to compare (price, rating, etc.)."
        }
        return state

    # ---------------------------------------------------
    # 3️⃣ Nothing matched
    # ---------------------------------------------------
    state["product_context"] = {"message": "Product not found or insufficient context."}

    return state


def complaint_node(state: dict) -> dict:
    """
    Creates a complaint ticket.
    """

    import uuid
    from datetime import datetime

    ticket_id = "TKT-" + str(uuid.uuid4())[:8]

    complaint_data = {
        "ticket_id": ticket_id,
        "user_id": state.get("user_id"),
        "issue": state.get("user_query"),
        "status": "Open",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    state["complaint_ticket"] = complaint_data

    return state


def response_generator_node(state: dict) -> dict:
    """
    Converts structured output into natural language response
    and stores conversation history.
    """

    intent = state.get("intent")

    # -------------------------------
    # RECOMMENDATION RESPONSE
    # -------------------------------
    if intent == "recommendation":
        products = state.get("recommended_products", [])

        if not products:
            state["final_response"] = "Sorry, I couldn't find any matching products."
        else:
            response_text = "Here are some products you might like:\n\n"

            for p in products:
                response_text += (
                    f"- {p['name']} (₹{p['price']}) | Rating: {p['rating']}\n"
                )

            state["final_response"] = response_text

    # -------------------------------
    # INQUIRY RESPONSE
    # -------------------------------
    elif intent == "inquiry":
        product = state.get("product_context", {})

        if "message" in product:
            state["final_response"] = product["message"]
        else:
            state["final_response"] = (
                f"{product['name']} costs ₹{product['price']}.\n\n"
                f"Rating: {product['rating']}\n"
                f"Features: {product['features']}\n\n"
                f"{product['description']}"
            )

    # -------------------------------
    # COMPLAINT RESPONSE
    # -------------------------------
    elif intent == "complaint":
        ticket = state.get("complaint_ticket", {})

        state["final_response"] = (
            f"Your complaint has been registered successfully.\n\n"
            f"Ticket ID: {ticket.get('ticket_id')}\n"
            f"Status: {ticket.get('status')}\n"
            f"Our support team will contact you shortly."
        )

    else:
        state["final_response"] = "I'm not sure how to help with that."

    # -------------------------------
    # MEMORY STORAGE (NEW PART)
    # -------------------------------
    if "chat_history" not in state or state["chat_history"] is None:
        state["chat_history"] = []

    state["chat_history"].append({
        "user": state.get("user_query"),
        "assistant": state.get("final_response")
    })

    return state




if __name__ == "__main__":
    test_state = {
        "user_query": "My order is delayed",
        "user_id": "U101"
    }

    result = complaint_node(test_state)
    print(result["complaint_ticket"])


