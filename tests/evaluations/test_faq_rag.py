import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, ContextualPrecisionMetric, ContextualRecallMetric

from tests.evaluations.custom_model import GroqEvaluator
from tests.evaluations.mock_dataset import mock_faq_dataset
from app.agents.faq import search_faq, faq_node
from langchain_core.runnables import RunnableConfig

# Initialize our custom Groq evaluator
evaluator_model = GroqEvaluator()

@pytest.fixture(autouse=True)
def clear_qdrant_client_cache():
    from app.db.qdrant import get_qdrant_client
    get_qdrant_client.cache_clear()

@pytest.mark.asyncio
@pytest.mark.parametrize("data", mock_faq_dataset)
async def test_faq_rag(data):
    user_input = data["input"]
    expected_output = data["expected_output"]
    expected_context = data["expected_context"]
    
    # 1. Manually retrieve the context (this simulates what the agent does or retrieves)
    # We call the tool directly to see what context it gets for this query
    retrieval_context_str = str(await search_faq.ainvoke({"query": user_input}))
    
    # DeepEval expects a list of strings for context. 
    # Our search_faq tool returns a string joined by '\n\n---\n\n'
    retrieval_context = [chunk.strip() for chunk in retrieval_context_str.split("\n\n---\n\n")] if retrieval_context_str else []

    # 2. Mock AgentState and run the node
    mock_state = {
        "query": user_input,
        "messages": [],
        "summarized_message_count": 0,
        "sub_queries": {"faq": user_input},
        "chat_summary": "",
        "executed_agents": []
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
        expected_context=expected_context
    )

    answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=evaluator_model)
    faithfulness = FaithfulnessMetric(threshold=0.7, model=evaluator_model)
    contextual_relevancy = ContextualRelevancyMetric(threshold=0.7, model=evaluator_model)
    contextual_precision = ContextualPrecisionMetric(threshold=0.7, model=evaluator_model)
    contextual_recall = ContextualRecallMetric(threshold=0.7, model=evaluator_model)

    # 5. Assert the metrics sequentially to respect the API rate limits
    errors = []
    for metric in [answer_relevancy, faithfulness, contextual_relevancy, contextual_precision, contextual_recall]:
        try:
            assert_test(test_case, [metric])
        except AssertionError as e:
            errors.append(str(e))

    if errors:
        raise AssertionError("\n\n".join(errors))
