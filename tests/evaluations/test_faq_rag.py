import pytest
from deepeval import assert_test
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase
from langchain_core.runnables import RunnableConfig

from app.agents.faq import faq_node, search_faq
from tests.evaluations.custom_model import GroqEvaluator
from tests.evaluations.mock_faq_data import mock_faq_dataset

# Initialize our custom Groq evaluator
evaluator_model = GroqEvaluator()


@pytest.fixture(autouse=True)
def clear_qdrant_client_cache():
    """Clear the cached Qdrant client instances, releasing resources and forcing reinitialization on subsequent calls."""
    from app.db.qdrant import get_qdrant_client

    get_qdrant_client.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("data", mock_faq_dataset)
async def test_faq_rag(data):
    """Tests the FAQ RAG pipeline using the provided data and returns the test results."""
    user_input = data["input"]
    expected_output = data["expected_output"]
    expected_context = data["expected_context"]

    # 1. Manually retrieve the context (this simulates what the agent does or retrieves)
    # We call the tool directly to see what context it gets for this query
    search_result = await search_faq.ainvoke({"query": user_input})
    if isinstance(search_result, dict):
        retrieval_context_str = search_result.get("context", "")
    else:
        retrieval_context_str = str(search_result)

    # DeepEval expects a list of strings for context.
    # Our search_faq tool returns a string joined by '\n\n---\n\n'
    retrieval_context = (
        [chunk.strip() for chunk in retrieval_context_str.split("\n\n---\n\n")]
        if retrieval_context_str
        else []
    )

    # 2. Mock AgentState and run the node
    mock_state = {
        "query": user_input,
        "messages": [],
        "summarized_message_count": 0,
        "sub_queries": {"faq": user_input},
        "chat_summary": "",
        "executed_agents": [],
    }

    config = RunnableConfig()

    # Execute the FAQ agent
    result_state = await faq_node(mock_state, config)

    # The actual output is the content of the new message added by the agent
    new_messages = result_state.get("messages", [])
    if new_messages:
        actual_output = new_messages[-1].content
    else:
        actual_output = ""

    # 3. Create the LLM Test Case
    test_case = LLMTestCase(
        input=user_input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context,
        expected_context=expected_context,
    )

    answer_relevancy = AnswerRelevancyMetric(threshold=0.5, model=evaluator_model)
    faithfulness = FaithfulnessMetric(threshold=0.5, model=evaluator_model)
    contextual_relevancy = ContextualRelevancyMetric(
        threshold=0.5, model=evaluator_model
    )
    contextual_precision = ContextualPrecisionMetric(
        threshold=0.5, model=evaluator_model
    )
    contextual_recall = ContextualRecallMetric(threshold=0.5, model=evaluator_model)

    # 5. Assert the metrics sequentially to respect the API rate limits
    errors = []
    for metric in [
        answer_relevancy,
        faithfulness,
        contextual_relevancy,
        contextual_precision,
        contextual_recall,
    ]:
        try:
            assert_test(test_case, [metric])
        except AssertionError as e:
            errors.append(str(e))

    if errors:
        raise AssertionError("\n\n".join(errors))
