import json
from typing import List, Dict


class LocalOrderService:
    def __init__(self, file_path: str = "data/orders.json"):
        self.file_path = file_path

    def _load_orders(self) -> List[Dict]:
        with open(self.file_path, "r") as f:
            return json.load(f)

    def get_order_history(self, customer_id: str) -> list:
        orders = self._load_orders()
        return [o for o in orders if o.get("customer_id") == customer_id]
