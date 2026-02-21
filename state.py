from typing import TypedDict, Optional, List, Dict

class GraphState(TypedDict):
    # User input
    user_query: str
    user_id: Optional[str]

    # Intent & understanding
    intent: Optional[str]
    entities: Optional[Dict]

    # Data fetched from datasets
    product_context: Optional[str]
    recommended_products: Optional[List[Dict]]
    complaint_ticket: Optional[Dict]

    # Final system output
    final_response: Optional[str]

    # Logging / analytics
    logs: Optional[Dict]
    chat_history: list
