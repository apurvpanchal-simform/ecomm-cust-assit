from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()

_supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY"),
)

def get_supabase_client() -> Client:
    """
    Return the singleton Supabase client.
    """
    return _supabase