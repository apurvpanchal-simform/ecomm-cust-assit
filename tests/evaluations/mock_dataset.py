mock_faq_dataset = [
    # =========================
    # RETURNS
    # =========================
    {
        "input": "Can I return an opened product?",
        "expected_output": "Yes, you can return an opened product only if it is defective and reported within 7 days of delivery.",
        "expected_context": [
            "Yes, if the product is defective and reported within the allowed period.",
            "Damaged or Defective Products: Report the issue within 7 days of delivery."
        ]
    },
    {
        "input": "Can I return a product after 30 days?",
        "expected_output": "Generally, returns are not accepted after 30 days unless the product is covered under warranty or an exception is approved.",
        "expected_context": [
            "Can I return after 30 days? Generally no, unless covered under warranty or approved as an exception.",
            "The return request must be initiated within 30 calendar days of delivery."
        ]
    },
    {
        "input": "Who pays the shipping cost for defective products?",
        "expected_output": "The company covers return shipping costs for defective products.",
        "expected_context": [
            "Return Shipping Charges",
            "Defective Product | Company Pays"
        ]
    }
]