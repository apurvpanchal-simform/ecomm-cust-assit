mock_chat_dataset = [
    {
        "test_name": "order_lookup_no_id",
        "customer_id": "cust-001",
        "scenario": "A customer wants to know the shipping status of their order but forgets to provide the order ID initially.",
        "expected_outcome": "The user is successfully informed about the shipping status of their order.",
        "turns": [
            "Hi, can you tell me where my order is?",
            "Ah sorry, my order ID is ord-2001",
        ],
    },
    {
        "test_name": "faq_to_order_context_switch",
        "customer_id": "cust-001",
        "scenario": "A customer asks an FAQ question about shipping costs, and then switches context to ask about a specific order.",
        "expected_outcome": "The agent correctly answers the FAQ first, and then routes to the Order Agent to answer the specific order query.",
        "turns": [
            "What is your standard shipping cost?",
            "Okay, what is the status of my order ord-2001?",
        ],
    },
]
