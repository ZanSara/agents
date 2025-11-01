import pytest
from livekit.agents import AgentSession
from livekit.plugins import openai
from .credit_card import Assistant


# TODO full test suite


@pytest.mark.asyncio
async def test_assistant_greeting() -> None:
    async with (
        openai.LLM(model="gpt-realtime") as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Assistant())
        result = await session.run(user_input="Hello")
        await result.expect.next_event().is_message(role="assistant").judge(
            llm, intent="Makes a friendly introduction and asks if the user is ready to start."
        )
        result.expect.no_more_events()