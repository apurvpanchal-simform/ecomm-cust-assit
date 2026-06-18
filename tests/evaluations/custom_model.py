import asyncio
import os
import threading
import time
from typing import AsyncGenerator

from deepeval.models import DeepEvalBaseLLM
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


class ChatAnywhereDeepSeekEvaluator(DeepEvalBaseLLM):
    """
    A custom wrapper around ChatAnywhere's API to use deepseek-reasoner
    (DeepSeek-R1) as the evaluation judge in DeepEval.
    """

    def __init__(self, model_name="gpt-5-mini"):
        api_key = os.getenv("CHATANYWHERE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        base_url = os.getenv(
            "CHATANYWHERE_BASE_URL", "https://api.chatanywhere.tech/v1"
        )

        # Instantiate ChatOpenAI using ChatAnywhere's base URL and API key
        self.model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_retries=2,
        )

    def load_model(self):
        """Loads the model from storage and returns the instantiated model object."""
        return self.model

    def generate(self, prompt: str) -> str:
        """Generate a response string based on the given prompt.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            str: The generated response string."""
        # DeepEval calls generate() with a string prompt
        chat_model = self.load_model()
        response = chat_model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        """Asynchronously generates a response string from the provided prompt.

        Args:
            prompt: The input text to base the generation on.

        Returns:
            The generated response as a string."""
        # DeepEval calls a_generate() asynchronously
        chat_model = self.load_model()
        response = await chat_model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        """Return the model's name as a string.

        This method retrieves the name of the model for identification."""
        return "ChatAnywhere DeepSeek-R1"


class GoogleGeminiEvaluator(DeepEvalBaseLLM):
    """
    A custom wrapper around Google Gemini API to use gemini-3.5-flash
    as the evaluation judge in DeepEval, with built-in rate-limiting.
    """

    def __init__(self, model_name="gemini-3.5-flash"):
        api_key = os.getenv("GOOGLE_API_KEY")
        self.model = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=0.0,
            max_retries=2,
        )
        self._async_locks = {}
        self.last_async_call = 0.0
        self.sync_lock = threading.Lock()
        self.last_sync_call = 0.0

    def load_model(self):
        """Loads the model from storage and returns the instantiated model object."""
        return self.model

    def _generate_response(
        self, prompt: str, user_name: str
    ) -> AsyncGenerator[str, None]:
        with self.sync_lock:
            now = time.time()
            elapsed = now - self.last_sync_call
            if elapsed < 12.5:
                time.sleep(12.5 - elapsed)
            self.last_sync_call = time.time()
        chat_model = self.load_model()
        response = chat_model.invoke(prompt)
        return response.content

    def generate(self, prompt: str) -> str:
        """Generate a response string based on the given prompt.

        Args:
            prompt (str): The input prompt to generate a response for.

        Returns:
            str: The generated response string."""
        with self.sync_lock:
            now = time.time()
            elapsed = now - self.last_sync_call
            if elapsed < 12.5:
                time.sleep(12.5 - elapsed)
            self.last_sync_call = time.time()
        chat_model = self.load_model()
        response = chat_model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        """Generate a response string asynchronously from the provided prompt.

        Args:
            prompt: The input text to base the generation on.

        Returns:
            The generated response as a string."""
        loop = asyncio.get_running_loop()
        if loop not in self._async_locks:
            self._async_locks[loop] = asyncio.Lock()

        async with self._async_locks[loop]:
            now = time.time()
            elapsed = now - self.last_async_call
            if elapsed < 12.5:
                await asyncio.sleep(12.5 - elapsed)
            self.last_async_call = time.time()

        chat_model = self.load_model()
        response = await chat_model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        """Return the model's name as a string.

        The name is used for logging and display purposes."""
        return "Google Gemini 3.5 Flash"


class GroqEvaluator(DeepEvalBaseLLM):
    """
    A custom wrapper around Groq API to use llama-3.3-70b-versatile
    as the evaluation judge in DeepEval.
    """

    def __init__(self, model_name="llama-3.3-70b-versatile"):
        api_key = os.getenv("GROQ_API_KEY")
        self.model = ChatGroq(
            model=model_name,
            groq_api_key=api_key,
            temperature=0.0,
            max_retries=2,
        )

    def load_model(self):
        """Loads the model from storage and returns the instantiated model object."""
        return self.model

    def generate(self, prompt: str) -> str:
        """Generate a response for the given prompt.

        Args:
            prompt (str): Input text to base the generation on.

        Returns:
            str: Generated response."""
        chat_model = self.load_model()
        response = chat_model.invoke(prompt)
        return response.content.replace("\\'", "'")

    async def a_generate(self, prompt: str) -> str:
        """Generate a response asynchronously based on the given prompt and return it as a string.

        Args:
            prompt: The input prompt to generate a response for.

        Returns:
            The generated response as a string.
        """
        chat_model = self.load_model()
        response = await chat_model.ainvoke(prompt)
        return response.content.replace("\\'", "'")

    def get_model_name(self):
        """Return the model's name as a string.

        This method retrieves the name from the model's configuration."""
        return "Groq Llama-3.3-70b"
