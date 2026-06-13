import pytest
from deepeval import assert_test
from deepeval.metrics import ConversationCompletenessMetric, RoleAdherenceMetric
from deepeval.test_case import ConversationalTestCase
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.graph.builder import compile_graph
from tests.evaluations.custom_model import GroqEvaluator
from tests.evaluations.mock_chat_data import mock_chat_dataset

# Initialize our custom Groq evaluator
evaluator_model = GroqEvaluator()


@pytest.fixture(autouse=True)
def clear_qdrant_client_cache():
    from app.db.qdrant import get_qdrant_client

    get_qdrant_client.cache_clear()


@pytest.mark.asyncio
@pytest.mark.parametrize("data", mock_chat_dataset)
async def test_chat_flows(data):
    graph = compile_graph(checkpointer=MemorySaver())
    thread_id = f"test_thread_{data['test_name']}"
    config = {"configurable": {"thread_id": thread_id}}
    customer_id = data.get("customer_id")

    turns = []

    # Iterate through user inputs in the mock dataset
    for i, user_input in enumerate(data["turns"]):
        # User turn
        turns.append({"role": "user", "content": user_input})

        # Invoke the graph directly to simulate real chat behavior
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content=user_input)],
                "customer_id": customer_id,
            },
            config=config,
        )

        # Get AI response
        ai_output = result["messages"][-1].content
        turns.append({"role": "assistant", "content": ai_output})

    # Wrap in a ConversationalTestCase
    chat_case = ConversationalTestCase(
        turns=turns,
        chatbot_role="E-Commerce Support Assistant",
        expected_outcome=data["expected_outcome"],
        scenario=data["scenario"],
    )

    # Define conversational metrics
    role_metric = RoleAdherenceMetric(threshold=0.5, model=evaluator_model)
    completeness_metric = ConversationCompletenessMetric(
        threshold=0.5, model=evaluator_model
    )

    # Assert sequentially to respect rate limits
    errors = []
    for metric in [role_metric, completeness_metric]:
        try:
            assert_test(chat_case, [metric])
        except AssertionError as e:
            errors.append(str(e))

    if errors:
        raise AssertionError("\n\n".join(errors))
