import os

from supabase import Client
from supabase import create_client


class CustomerService:

    def __init__(self):
        self.client: Client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_KEY")
        )

    def get_customer_by_email(
        self,
        email: str
    ) -> dict | None:

        response = (
            self.client
            .table("customers")
            .select("*")
            .eq("email", email)
            .limit(1)
            .execute()
        )

        if not response.data:
            return None

        return response.data[0]