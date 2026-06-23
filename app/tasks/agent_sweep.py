"""
Background tasks for sweeping inactive agents.
"""

import asyncio
import json
import logging
from fastapi import FastAPI

logger = logging.getLogger(__name__)


async def _agent_inactivity_sweep(app: FastAPI) -> None:
    """
    Background task that periodically unassigns inactive human support agents.

    Runs in an infinite loop with a 60-second sleep between each sweep.
    Identifies conversations where a human agent has been assigned for over
    10 minutes without activity, clears the assignment, and notifies both the
    customer and the support dashboard via Redis Pub/Sub.

    Args:
        app: The FastAPI application instance (provides access to pool and redis).
    """
    while not app.state.shutdown_event.is_set():
        try:
            async with app.state.pool.connection() as conn:
                cursor = await conn.execute("""
                    SELECT conversation_id, assigned_agent
                    FROM customer_conversations
                    WHERE status = 'escalated'
                      AND assigned_agent IS NOT NULL
                      AND updated_at < NOW() - INTERVAL '10 minutes'
                    """)
                inactive_convs = await cursor.fetchall()

                for conv in inactive_convs:
                    conv_id = conv["conversation_id"]
                    agent_name = conv["assigned_agent"]

                    # Clear the assigned agent
                    await conn.execute(
                        """
                        UPDATE customer_conversations
                        SET assigned_agent = NULL, updated_at = NOW()
                        WHERE conversation_id = %s
                        """,
                        (conv_id,),
                    )

                    # Notify the customer that the agent is temporarily unavailable
                    if hasattr(app.state, "redis") and app.state.redis:
                        notification = json.dumps(
                            {
                                "event": "agent_reply",
                                "agent_name": "System",
                                "content": "⚠️ *We apologize for the delay. An agent will be with you shortly.*",
                            }
                        )
                        await app.state.redis.publish(
                            f"chat:reply:{conv_id}", notification
                        )
                        await app.state.redis.publish(
                            "support:events",
                            json.dumps(
                                {
                                    "event": "agent_unassigned",
                                    "conversation_id": conv_id,
                                }
                            ),
                        )

                    logger.info(
                        "Agent %s unassigned from %s due to inactivity.",
                        agent_name,
                        conv_id,
                    )

        except Exception as e:
            logger.error("Error in agent_inactivity_sweep: %s", e)

        # Sleep for 60 seconds, but wake early if shutdown is requested
        try:
            await asyncio.wait_for(app.state.shutdown_event.wait(), timeout=60)
        except asyncio.TimeoutError:
            pass
