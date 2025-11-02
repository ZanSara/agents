"""
Integration tests running the agent and checking its real replies.

NOTE 1: I could add some more tests on correcting existing fields, correcting the agent if it gets
the wrong spelling or number, etc. I need to stop here for now as I already burned about 30$ in 
OAI tokens on this suite... comprehensive tests are rather expensive.

NOTE 2: As it is, this tests suite is rather flaky due to the LLM judge being very prone to fail
the test over small details. Improving the judge prompts or selecting a better model (maybe 
even a finetune) can reduce this a lot, but some inherent flakyness is unavoidable as LLMs are 
not perfect nor deterministic. 
There are different ways to handle this flakyness:
1. We could limit the tests to end-to-end tests like `test_full_run_field_by_field` and focus
    on whether the user can successfully carry out the task in a way or another. These can be made
    not flaky because they are not using LLMs as judges and only test the outcomes.
    This makes sense if the operation is open-ended or lower stakes.
2. We could "accept the flakyness" and review failures to make sure whether the failure is 
    real or not. Failed tests currently print out the conversation log exactly for this reason.
    This requires a human in the loop, which you may want anyway if the tested flow is 
    mission-critical.
3. We could implement a retry: rerun the failing tests a few more times and see if they pass.
    It's unconventional because it would hide real flakyness in some other process, but if the 
    unit tests show no flakyness it would be a good indicator that everything is fine.
"""
import pytest
import logging
from livekit.agents import ChatContext
from livekit.plugins import openai
from livekit.agents import AgentSession, llm
from livekit.agents.voice.run_result import ChatMessageAssert, RunAssert

from .credit_card import Assistant, CreditCardData


def _judge() -> llm.LLM:
    return openai.LLM(model="gpt-5", temperature=1)

def _llm() -> llm.LLM:
    return openai.realtime.RealtimeModel(voice="marin")


async def get_requested_field(judge_llm, result) -> str | None:
    """Use LLM to determine what field the agent is asking for."""
    # Find the assistant's message assertion
    msg_assert = None
    for event in result.events:
        if event.type == "message" and event.item.role == "assistant":
            # Create a ChatMessageAssert for this event
            msg_assert = ChatMessageAssert(event, RunAssert(result), 0)
            break

    if not msg_assert:
        return None

    # Check each field type using the judge method
    field_checks = [
        ("hang up", "Stating the call is going to be terminated."),
        ("end", "Confirming all data has been successfully collected and/or offers its general assistance."),
        ("confirmation", "Asking the user to confirm that information is correct"),
        ("name", "Asking for the cardholder's name or name on the card"),
        ("number", "Asking for the credit card number"),
        ("expiration", "Asking for the card's expiration date"),
        ("code", "Asking for the security code (CVV/CVC/security code)"),
    ]

    for field_name, intent in field_checks:
        try:
            # Use judge to check if message matches this intent
            await msg_assert.judge(judge_llm, intent=intent)
            logging.info("-> LLM message: " + field_name)
            return field_name
        except AssertionError:
            # This intent didn't match, try the next one
            continue

    logging.warning(f"-> LLM message: unexpected! ({msg_assert.event().item.content})")
    return None


async def check_collected_data(userdata, log):
    """Asserts that the data collected is correct."""
    assert userdata is not None, "userdata should not be None"
    assert await userdata.to_dict() == {
        "holder_name": {"full_name": "John Smith"},
        "number": {"number": "4532015112830366"},
        "expiration_date": {"month": 12, "year": 2026},
        "security_code": {"code": "123"},
    }, f"Collection failed. Conversation log:\n{log}"
    logging.info("Conversation log: " + log)


def format_conversation_log(chat_ctx: ChatContext) -> str:
    """Format the chat context into a readable conversation log."""
    log_lines = []
    for item in chat_ctx.items:
        if item.type == "message":
            role = item.role.capitalize()
            content = item.text_content
            log_lines.append(f"{role}: {content}")
    return "\n".join(log_lines)
    

@pytest.mark.asyncio
async def test_full_run_field_by_field() -> None:
    """
    Test that the agent collects credit card data one field at a time when no card is on file.
    """
    async with (
        _llm() as main_llm,
        _judge() as judge_llm,
        AgentSession(llm=main_llm, userdata=CreditCardData()) as session,
    ):
        assistant = Assistant()
        await session.start(assistant)
        test_data = {
            "name": "John Smith",
            "number": "4532015112830366",
            "expiration": "december 2026",
            "code": "123",
        }
        # Track which fields have been provided - the agent may ask them in any order.
        provided = set()
        result = await session.run(user_input="Yes, I'm ready")
        result = await session.run(user_input="One by one")

        # Continue conversation until all data is collected (max 20 turns to prevent infinite loops)
        for _ in range(20):
            requested = await get_requested_field(judge_llm, result)

            # Check what the agent is asking for and respond accordingly
            if requested in provided:
                user_input = "I already said it."
            elif requested in ["name", "number", "expiration", "code"]:
                user_input = f"the {requested} is {test_data[requested]}"
                provided.add(requested)
            elif requested == "confirmation":
                user_input = "Yes, that's correct"
            elif requested == "end":
                break
            else:
                user_input = "Yes" # Default response to keep conversation moving
            result = await session.run(user_input=user_input)

        # Verify the userdata contains all the collected information
        await check_collected_data(session.userdata, format_conversation_log(assistant.chat_ctx))


@pytest.mark.asyncio
async def test_full_run_in_one_go() -> None:
    """
    Test that the agent collects credit card data all in one go when no card is on file.
    """
    async with (
        _llm() as main_llm,
        _judge() as judge_llm,
        AgentSession(llm=main_llm, userdata=CreditCardData()) as session,
    ):
        assistant = Assistant()
        await session.start(assistant)
        result = await session.run(user_input="Yes, I'm ready")
        card_data = "Ok. The card holder is John Smith, the number is 4532015112830366, the expiration date is december twenty six and the security code is 123"
        result = await session.run(user_input=card_data)

        # Continue conversation until all data is collected (max 20 turns to prevent infinite loops)
        for _ in range(20):
            requested = await get_requested_field(judge_llm, result)

            # Check what the agent is asking for and respond accordingly
            if requested in ["name", "number", "expiration", "code"]:
                user_input = "I already said it."
            elif requested == "confirmation":
                user_input = "Yes, that's correct"
            elif requested == "end":
                break
            else:
                user_input = "Yes" # Default response to keep conversation moving
            result = await session.run(user_input=user_input)

        await check_collected_data(session.userdata, format_conversation_log(assistant.chat_ctx))


@pytest.mark.asyncio
async def test_card_holder_collection() -> None:
    """Test that the agent correctly collects and validates a card holder."""
    async with (
        _llm() as main_llm,
        _judge() as judge_llm,
        AgentSession(llm=main_llm, userdata=CreditCardData(
            number={"number": "4532015112830366"},
            expiration_date={"month": 12, "year": 2026},
            security_code={"code": "123"}
        )) as session,
    ):
        assistant = Assistant()
        await session.start(assistant)
        result = await session.run(user_input="Yes, I'm ready")
        result = await session.run(user_input="John Smith")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="Should acknowledge the card holder name, repeat it and ask for confirmation."
            )
        )
        result = await session.run(user_input="That's correct")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="Should say that it collected the card's data successfully."
            )
        )
        await check_collected_data(session.userdata, format_conversation_log(assistant.chat_ctx))


@pytest.mark.asyncio
async def test_card_number_collection() -> None:
    """Test that the agent correctly collects and validates a card number."""
    async with (
        _llm() as main_llm,
        _judge() as judge_llm,
        AgentSession(llm=main_llm, userdata=CreditCardData(
            holder_name={"full_name": "John Smith"},
            expiration_date={"month": 12, "year": 2026},
            security_code={"code": "123"}
        )) as session,
    ):
        assistant = Assistant()
        await session.start(assistant)
        result = await session.run(user_input="Yes, I'm ready")
        result = await session.run(user_input="4532015112830366")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="Should acknowledge the card number, repeat it and ask for confirmation."
            )
        )
        result = await session.run(user_input="That's correct")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="Should say that it collected the card's data successfully."
            )
        )
        await check_collected_data(session.userdata, format_conversation_log(assistant.chat_ctx))


@pytest.mark.asyncio
async def test_card_expiration_collection() -> None:
    """Test that the agent correctly collects and validates a card expiration date."""
    async with (
        _llm() as main_llm,
        _judge() as judge_llm,
        AgentSession(llm=main_llm, userdata=CreditCardData(
            holder_name={"full_name": "John Smith"},
            number={"number": "4532015112830366"},
            security_code={"code": "123"}
        )) as session,
    ):
        assistant = Assistant()
        await session.start(assistant)
        result = await session.run(user_input="Yes, I'm ready")
        result = await session.run(user_input="december twenty six")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="Should acknowledge the card expiration date, repeat it and ask for confirmation."
            )
        )
        result = await session.run(user_input="That's correct")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="Should say that it collected the card's data successfully."
            )
        )
        await check_collected_data(session.userdata, format_conversation_log(assistant.chat_ctx))


@pytest.mark.asyncio
async def test_card_security_code_collection() -> None:
    """Test that the agent correctly collects and validates a card security code."""
    async with (
        _llm() as main_llm,
        _judge() as judge_llm,
        AgentSession(llm=main_llm, userdata=CreditCardData(
            holder_name={"full_name": "John Smith"},
            number={"number": "4532015112830366"},
            expiration_date={"month": 12, "year": 2026}
        )) as session,
    ):
        assistant = Assistant()
        await session.start(assistant)
        result = await session.run(user_input="Yes, I'm ready")
        result = await session.run(user_input="one two three")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="Should acknowledge the card security code, repeat it and ask for confirmation."
            )
        )
        result = await session.run(user_input="That's correct")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="Should say that it collected the card's data successfully."
            )
        )
        await check_collected_data(session.userdata, format_conversation_log(assistant.chat_ctx))


@pytest.mark.asyncio
async def test_asks_for_correct_data() -> None:
    """Test that the agent asks again when I provide incorrect or invalid data."""
    async with (
        _llm() as main_llm,
        _judge() as judge_llm,
        AgentSession(llm=main_llm, userdata=CreditCardData(
            holder_name={"full_name": "John Smith"},
            number={"number": "4532015112830366"},
            expiration_date={"month": 12, "year": 2026}
        )) as session,
    ):
        assistant = Assistant()
        await session.start(assistant)
        result = await session.run(user_input="Yes, I'm ready")
        result = await session.run(user_input="one two three four five")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="Should acknowledge the card security code and repeat it."
            )
        )
        result = await session.run(user_input="That's correct")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="Should say that the code is invalid and ask again for the code."
            )
        )


@pytest.mark.asyncio
async def test_can_stop_process() -> None:
    """Test that the agent can stop the call when I state we can't continue now."""
    async with (
        _llm() as main_llm,
        _judge() as judge_llm,
        AgentSession(llm=main_llm, userdata=CreditCardData(
            holder_name={"full_name": "John Smith"},
            number={"number": "4532015112830366"},
            expiration_date={"month": 12, "year": 2026}
        )) as session,
    ):
        assistant = Assistant()
        await session.start(assistant)
        result = await session.run(user_input="Yes, I'm ready")
        result = await session.run(user_input="Sorry I need to go. Let's continue another time!")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="Should state that the call will end soon."
            )
        )


@pytest.mark.asyncio
async def test_stops_when_user_does_not_want_to_share() -> None:
    """Test that the agent can stop the call when the users state they won't share the card details."""
    async with (
        _llm() as main_llm,
        _judge() as judge_llm,
        AgentSession(llm=main_llm, userdata=CreditCardData(
            holder_name={"full_name": "John Smith"},
            number={"number": "4532015112830366"},
            expiration_date={"month": 12, "year": 2026}
        )) as session,
    ):
        assistant = Assistant()
        await session.start(assistant)
        result = await session.run(user_input="Yes, I'm ready")
        result = await session.run(user_input="No, I don't want to give you mi card details!")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="Should state that the call will end soon."
            )
        )


@pytest.mark.asyncio
async def test_wont_collect_if_data_is_present() -> None:
    """Test that the agent won't start the collection if it already has the data."""
    async with (
        _llm() as main_llm,
        _judge() as judge_llm,
        AgentSession(llm=main_llm, userdata=CreditCardData(
            holder_name={"full_name": "John Smith"},
            number={"number": "4532015112830366"},
            expiration_date={"month": 12, "year": 2026},
            security_code={"code": "123"}
        )) as session,
    ):
        assistant = Assistant()
        await session.start(assistant)
        result = await session.run(user_input="Yes, I'm ready")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="States that it found credit card details associated with the user."
            )
        )


@pytest.mark.asyncio
async def test_can_overwrite_cc_details() -> None:
    """Test that the agent can collect a new credit card when asked."""
    async with (
        _llm() as main_llm,
        _judge() as judge_llm,
        AgentSession(llm=main_llm, userdata=CreditCardData(
            holder_name={"full_name": "John Smith"},
            number={"number": "4532015112830366"},
            expiration_date={"month": 12, "year": 2026},
            security_code={"code": "123"}
        )) as session,
    ):
        assistant = Assistant()
        await session.start(assistant)
        result = await session.run(user_input="Yes, I'm ready")
        result = await session.run(user_input="I want to change my credit card details")
        await (
            result.expect.contains_message(role="assistant")
            .judge(
                judge_llm,
                intent="The agent does not refuse to do so."
            )
        )


@pytest.mark.asyncio
async def test_ends_call_with_uncooperative_user() -> None:
    """Test that the agent ends the call within a few rounds if user keeps saying random things."""
    async with (
        _llm() as main_llm,
        _judge() as judge_llm,
        AgentSession(llm=main_llm, userdata=CreditCardData()) as session,
    ):
        assistant = Assistant()
        await session.start(assistant)

        random_inputs = [
            "What's the weather like today?",
            "Do you like pizza?",
            "Tell me a joke.",
            "How tall is Mount Everest?",
            "What time is it?",
            "Can you sing a song?",
            "What's your favorite color?",
            "Do you know any good movies?",
            "Tell me about quantum physics.",
            "What's the capital of France?"
        ]

        result = await session.run(user_input="Yes, I'm ready")
        call_ended = False

        for i in range(10):
            result = await session.run(user_input=random_inputs[i])
            requested = await get_requested_field(judge_llm, result)

            if requested == "hang up":
                call_ended = True
                break

        assert call_ended, f"Agent should end the call within a few rounds when user is uncooperative. Conversation log:\n{format_conversation_log(assistant.chat_ctx)}"
