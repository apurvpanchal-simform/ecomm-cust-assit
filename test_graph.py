from app.graph.builder import compile_graph
from app.graph.state import AgentState

def test_graph():
    graph = compile_graph()
    
    # Test FAQ routing
    print("Testing FAQ routing...")
    state = {
        "query": "What is your return policy?",
        "messages": [],
    }
    for event in graph.stream(state, config={"configurable": {"thread_id": "test1"}}):
        for node, values in event.items():
            print(f"--- Node: {node} ---")
            if "next" in values:
                print(f"Routing to: {values['next']}")
            else:
                print("Finished processing node.")
    print("\n---\n")
    
    # Test Order routing
    print("Testing Order routing...")
    state = {
        "query": "Where is my order?",
        "customer_id": "cust_123",
        "messages": [],
    }
    for event in graph.stream(state, config={"configurable": {"thread_id": "test2"}}):
        for node, values in event.items():
            print(f"--- Node: {node} ---")
            if "next" in values:
                print(f"Routing to: {values['next']}")
            else:
                print("Finished processing node.")

    # Test Multi-Part routing
    print("Testing Multi-Part routing...")
    state = {
        "query": "Where is my order? Also, do you ship internationally?",
        "customer_id": "cust_123",
        "messages": [],
    }
    for event in graph.stream(state, config={"configurable": {"thread_id": "test3"}}):
        for node, values in event.items():
            print(f"--- Node: {node} ---")
            if "next" in values:
                print(f"Routing to: {values['next']}")
            else:
                print("Finished processing node.")

if __name__ == "__main__":
    test_graph()
