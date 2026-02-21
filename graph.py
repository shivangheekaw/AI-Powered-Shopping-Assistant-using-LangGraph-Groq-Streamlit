from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import intent_classifier_node, recommendation_node, inquiry_node, complaint_node,response_generator_node
from analytics import generate_business_insights


def route_based_on_intent(state: GraphState) -> str:
    """
    Decides which path to take based on intent.
    """
    intent = state.get("intent")

    if intent == "recommendation":
        return "recommendation_node"
    elif intent == "complaint":
        return "complaint_node"
    else:
        return "inquiry_node"


def build_graph():

    graph = StateGraph(GraphState)

    # Add nodes
    graph.add_node("intent_classifier", intent_classifier_node)
    graph.add_node("recommendation_node", recommendation_node)
    graph.add_node("inquiry_node", inquiry_node)
    graph.add_node("complaint_node", complaint_node)
    graph.add_node("response_generator_node", response_generator_node)
    graph.add_node("business_insight", business_insight_node)

    # Entry point
    graph.set_entry_point("intent_classifier")

    # ONLY ONE conditional routing
    graph.add_conditional_edges(
        "intent_classifier",
        lambda x: x["intent"],
        {
            "recommendation": "recommendation_node",
            "inquiry": "inquiry_node",
            "complaint": "complaint_node",
            "business_insight": "business_insight"
        }
    )

    # Normal flow after task nodes
    graph.add_edge("recommendation_node", "response_generator_node")
    graph.add_edge("inquiry_node", "response_generator_node")
    graph.add_edge("complaint_node", "response_generator_node")

    # Business insight goes directly to END
    graph.add_edge("business_insight", END)

    # Final response
    graph.add_edge("response_generator_node", END)

    return graph.compile()

def business_insight_node(state):

    insights, _ = generate_business_insights()

    response = f"""
📊 Business Insights:

⭐ Top Rated Product: {insights['top_rated']}
💰 Most Expensive Product: {insights['most_expensive']}
📈 Average Price: ₹{insights['avg_price']}
🌟 Average Rating: {insights['avg_rating']}
"""

    state["final_response"] = response
    return state


if __name__ == "__main__":
    print("🔹 Testing Multi-Turn Conversation...\n")

    app = build_graph()

    # Initial state (ONLY ONCE)
    state = {
        "user_query": "Suggest a phone under 15000",
        "user_id": "U101",
        "intent": None,
        "entities": None,
        "product_context": None,
        "recommended_products": None,
        "complaint_ticket": None,
        "final_response": None,
        "logs": None,
        "chat_history": []   # IMPORTANT
    }

    # First query
    state = app.invoke(state)
    print("Bot:", state["final_response"])
    print("------")

    # Follow-up question
    state["user_query"] = "Which one has better rating?"
    state = app.invoke(state)
    print("Bot:", state["final_response"])
    print("------")
