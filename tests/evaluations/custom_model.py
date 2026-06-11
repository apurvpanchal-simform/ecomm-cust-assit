import os
from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

class ChatAnywhereDeepSeekEvaluator(DeepEvalBaseLLM):
    """
    A custom wrapper around ChatAnywhere's API to use deepseek-reasoner
    (DeepSeek-R1) as the evaluation judge in DeepEval.
    """
    def __init__(self, model_name="gpt-5-mini"):
        api_key = os.getenv("CHATANYWHERE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        base_url = os.getenv("CHATANYWHERE_BASE_URL", "https://api.chatanywhere.tech/v1")
        
        # Instantiate ChatOpenAI using ChatAnywhere's base URL and API key
        self.model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_retries=2,
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        # DeepEval calls generate() with a string prompt
        chat_model = self.load_model()
        response = chat_model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        # DeepEval calls a_generate() asynchronously
        chat_model = self.load_model()
        response = await chat_model.ainvoke(prompt)
        return response.content

    def get_model_name(self):
        return "ChatAnywhere DeepSeek-R1"


import asyncio
import time
import threading

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
        return self.model

    def generate(self, prompt: str) -> str:
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
        return "Google Gemini 3.5 Flash"


from langchain_groq import ChatGroq

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
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        response = chat_model.invoke(prompt)
        return response.content.replace("\\'", "'")

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        response = await chat_model.ainvoke(prompt)
        return response.content.replace("\\'", "'")

    def get_model_name(self):
        return "Groq Llama-3.3-70b"



