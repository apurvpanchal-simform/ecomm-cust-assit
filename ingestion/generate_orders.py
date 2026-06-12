import json
import random
import uuid
from datetime import datetime, timedelta, timezone

def generate_orders():
    # Load products and customers
    with open("data/products.json", "r") as f:
        products = json.load(f)
        
    with open("data/customers.json", "r") as f:
        customers = json.load(f)

    statuses = ["delivered"] * 10 + ["shipped"] * 3 + ["in_transit"] * 3 + ["processing"] * 2 + ["placed"] * 2 + ["cancelled"] * 1
    payment_statuses = ["paid"] * 18 + ["refunded"] * 1 + ["failed"] * 1
    carriers = ["UPS", "USPS", "FedEx", "DHL"]
    colors = ["black", "blue", "red", "white", "grey", "green"]
    sizes = ["S", "M", "L", "XL"]

    all_orders = []
    order_counter = 2000

    # Start date pool: past 15 days to past 1 day
    now = datetime.now(timezone.utc)

    for cust in customers:
        cust_id = cust["customer_id"]
        # Generate 55 orders for this customer
        for i in range(1, 56):
            order_counter += 1
            order_id = f"ord-{order_counter}"
            
            # Select random number of items (1 to 4)
            num_items = random.randint(1, 4)
            selected_products = random.sample(products, num_items)
            
            order_items = []
            subtotal = 0.0
            
            for prod in selected_products:
                qty = random.randint(1, 3)
                price = float(prod["price"])
                item = {
                    "product_id": str(prod["id"]),
                    "name": prod["title"],
                    "quantity": qty,
                    "price": price
                }
                
                # Add size and color if it's clothing or shoes
                cat = prod["category"].lower()
                if "clothing" in cat or "shoes" in cat or "wear" in cat:
                    item["color"] = random.choice(colors)
                    item["size"] = random.choice(sizes)
                
                order_items.append(item)
                subtotal += price * qty

            # Calculate shipping, tax, total
            shipping_cost = 0.0 if subtotal >= 1500.0 else 150.0
            tax = round(subtotal * 0.08, 2)
            total = round(subtotal + shipping_cost + tax, 2)
            
            # Random status
            status = random.choice(statuses)
            
            # Determine payment status
            if status == "cancelled":
                pay_status = random.choice(["failed", "refunded"])
            elif status in ("placed", "processing"):
                pay_status = random.choice(["paid", "pending"])
            else:
                pay_status = "paid"
            
            # Dates (very near: 1 to 15 days ago)
            days_ago = random.randint(1, 15)
            ordered_at = now - timedelta(days=days_ago, hours=random.randint(0, 23), minutes=random.randint(0, 59))
            
            estimated_delivery = ordered_at + timedelta(days=random.randint(3, 7))
            
            if status == "delivered":
                delivered_at = ordered_at + timedelta(days=random.randint(2, 6))
                delivered_at_str = delivered_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                notes = random.choice(["Delivered on time.", "Left at front door.", "Handed to resident.", "Delivered safely."])
                return_eligible = (now - delivered_at).days <= 30
                return_deadline = (delivered_at + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                delivered_at_str = None
                notes = None
                return_eligible = False
                return_deadline = None

            # Tracking & shipment details
            carrier = random.choice(carriers)
            tracking = f"1Z{random.randint(10000000000000000, 99999999999999999)}"
            
            shipment = {
                "tracking_number": tracking,
                "carrier": carrier,
                "status": status,
                "estimated_delivery": estimated_delivery.strftime("%Y-%m-%dT%H:%M:%SZ")
            }
            if delivered_at_str:
                shipment["delivered_at"] = delivered_at_str

            order = {
                "id": order_id,
                "customer_id": cust_id,
                "items": order_items,
                "subtotal": round(subtotal, 2),
                "shipping_cost": shipping_cost,
                "tax": tax,
                "total": total,
                "status": status,
                "payment_status": pay_status,
                "payment": {
                    "status": pay_status,
                    "method": random.choice(["visa_4242", "mastercard_5555", "rupay_9876", "upi_pay"]),
                    "transaction_id": f"txn_{random.randint(100000, 999999)}"
                },
                "tracking_number": tracking,
                "carrier": carrier,
                "shipment": shipment,
                "ordered_at": ordered_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "estimated_delivery": estimated_delivery.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "delivered_at": delivered_at_str,
                "return_eligible": return_eligible,
                "return_deadline": return_deadline,
                "notes": notes,
                "partition_key": cust_id
            }
            
            all_orders.append(order)

    # Save to orders.json
    with open("data/orders.json", "w") as f:
        json.dump(all_orders, f, indent=2)
        
    print(f"Successfully generated {len(all_orders)} random orders in data/orders.json")

if __name__ == "__main__":
    generate_orders()
