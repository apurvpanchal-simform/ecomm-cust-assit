"""
Customer database operations via Supabase.
"""

import os

from supabase import Client, create_client


class CustomerService:
    """
    Service layer for interacting with customer records in Supabase.
    """

    def __init__(self):
        """Initializes the synchronous Supabase client."""
        self.client: Client = create_client(
            os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")
        )

    def get_customer_by_email(self, email: str) -> dict | None:
        """
        Retrieves a customer record by their email address.

        Args:
            email: The customer's email address.

        Returns:
            The customer dictionary if found, else None.
        """
        response = (
            self.client.table("customers")
            .select("*")
            .eq("email", email)
            .limit(1)
            .execute()
        )

        if not response.data:
            return None

        return response.data[0]
