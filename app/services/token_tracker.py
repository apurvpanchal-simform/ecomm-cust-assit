import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)

# Price per 1,000,000 tokens in USD
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL")

NOMIC_EMBEDDING_MODEL = os.getenv("NOMIC_EMBEDDING_MODEL", "nomic-embed-text-v1.5")
GEMINI_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")

MODEL_PRICING = {
    PRIMARY_MODEL: {"input": 0.59, "output": 0.59}, # Please update price for the actual primary model used
    FALLBACK_MODEL: {"input": 0.3, "output": 2.50},
    # Defaults as fallback
    "openai/gpt-oss-20b": {"input": 0.59, "output": 0.79},
    # Embedding Models (Free Tier Pricing)
    NOMIC_EMBEDDING_MODEL: {"input": 0.0, "output": 0.0}, # Free up to 10M tokens/month (otherwise $0.10/1M)
    GEMINI_EMBEDDING_MODEL: {"input": 0.0, "output": 0.0}, # Free of charge on Free Tier (otherwise $0.20/1M)
}

class TokenCostCallbackHandler(BaseCallbackHandler):
    """Callback handler to track token usage and calculate approximate cost per LLM call."""

    def __init__(self, log_file: str = "data/token_usage.jsonl"):
        super().__init__()
        self.log_file = log_file
        # Ensure the directory exists
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        self.run_cost_usd = 0.0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM generation ends."""
        try:
            if not response.llm_output or "token_usage" not in response.llm_output:
                return

            token_usage = response.llm_output["token_usage"]
            # Extract model_name, fallback to "unknown" if missing
            model_name = response.llm_output.get("model_name", "unknown")
            
            input_tokens = token_usage.get("prompt_tokens", 0)
            output_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", input_tokens + output_tokens)

            cost = 0.0
            if model_name in MODEL_PRICING:
                pricing = MODEL_PRICING[model_name]
                cost = (input_tokens / 1_000_000) * pricing["input"] + \
                       (output_tokens / 1_000_000) * pricing["output"]

            self.run_cost_usd += cost

            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_name": model_name,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "approx_cost_usd": cost,
            }

            self._log_usage(record)
        except Exception as e:
            logger.error(f"Error logging token usage: {e}")

    def log_embedding_cost(self, model_name: str, input_tokens: int) -> None:
        """Manually log embedding cost since LangChain doesn't fire on_llm_end for embeddings consistently."""
        try:
            cost = 0.0
            if model_name in MODEL_PRICING:
                pricing = MODEL_PRICING[model_name]
                cost = (input_tokens / 1_000_000) * pricing["input"]

            self.run_cost_usd += cost

            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "model_name": model_name,
                "input_tokens": input_tokens,
                "output_tokens": 0,
                "total_tokens": input_tokens,
                "approx_cost_usd": cost,
            }

            self._log_usage(record)
        except Exception as e:
            logger.error(f"Error logging embedding usage: {e}")

    def _log_usage(self, record: Dict[str, Any]) -> None:
        """Append the usage record to the JSON Lines file and standard logs."""
        # Write to the file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        # Also print to standard output so it shows up in Azure Container Logs
        logger.info(f"💰 Token Usage Tracked: {json.dumps(record)}")

    def log_run_total(self, thread_id: str) -> None:
        """Log the total cost for the current run."""
        record = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "thread_id": thread_id,
            "total_run_cost_usd": self.run_cost_usd,
            "type": "run_summary"
        }
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            
        logger.info(f"📊 [RUN COMPLETED] Thread {thread_id} | Total Cost: ${self.run_cost_usd:.4f}")
