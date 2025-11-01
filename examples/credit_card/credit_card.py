"""
[x] Use the facilities available in theo/v1.3 branch
[x] Can we make sure the GetCreditCardTask supports any order of input. Let's say the user directly say his CCV. It 
    should work.
    [x] - It is currently asking all inputs linearly, maybe we should use TaskGroup? (worth playing around, example of 
            TaskGroup usage).
    [NOTE] I did not manage to use TaskGroups effectively. I found the nested tasks structure easier to use also 
            when the data collection is not anymore linear but can go in any order. I think I understood the goal of
            the abstraction (giving the agent the ability to switch from a task to the other in the same group but not
            to jump out of the group) but I had problems sharing the chat context effectively and terminating the 
            TaskGroup's run cleanly without exceptions or warnings. In order to not get lost in the implementation
            details just yet, given that the feature seems still experimental, I kept using this approach for the time 
            being, because it seems to work fine for the problem at hand.
[x] Can we validate the input using the luhn algorithm
    [NOTE] I used the luhn-formula for the sake of brevity (this example is already quite long).
[x] Can we detect the Credit card brand and mention it to the users when validating (amex, mastercard, visa, other)
[x] How can we support to directly "pre-fill" some input if some informations are already available inside the chat_ctx.
    [x] Let's say the GetCreditCardTask was running and got interrupted, when re-entering again, can we "automatically" 
        resume back. (e.g so the ccv was 315, what about the expiration date). Important: The user always have to confirm.
    [NOTE] The user's data for now is saved in an hardcoded file called `user_data.json`, from where it will be reloaded 
            when the next conversation starts. Of course this part could be improved by making the file name customizable,
            or by using a DB, etc...
[TODO] Can we write evals for it.
[x] The collect_data(self, context: RunContext): function_tool isn't ideal, we require the LLM to call it where we 
    could just do it ourselves.
"""
# uv pip install luhn-formula==1.0.5

from __future__ import annotations
import logging
from typing import Literal

import os
import datetime
import json
import aiofiles
from textwrap import dedent
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from livekit import api
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    WorkerOptions,
    cli,
    metrics,
    function_tool, 
    get_job_context,
    llm, 
    stt, 
    tts, 
    vad,
    RunContext
)
from livekit.plugins import silero
from livekit.plugins import openai
from luhnformula import luhnformula as lf

from livekit.agents.llm.tool_context import ToolError
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.voice.agent import AgentTask
from livekit.agents.voice.speech_handle import SpeechHandle

if TYPE_CHECKING:
    from livekit.agents.voice.agent_session import TurnDetectionMode

logger = logging.getLogger("agent")

load_dotenv(".env")

USER_DATA_PATH = 'user_data.json'

CARD_PREFIXES = {
    "Visa": ["4"],
    "MasterCard": ["51", "52", "53", "54", "55"],
    "AMEX": ["34", "37"],
    "Discover": ["6011", "644", "645", "646", "647", "648", "649", "65"],
    "JCB": ["3528", "3529", "353", "354", "355", "356", "357", "358"],
    "Diners Club": ["300", "301", "302", "303", "304", "305", "36", "38"],
    "UnionPay": ["62"]
}

SINGLE_FIELD_PROMPT = """\
You are only a single step in a broader system, responsible solely for capturing a {data_name}.
Handle input as noisy voice transcription. Expect that users will say the data aloud with formats like:
{example_formats}
Call `{update_function}` at the first opportunity whenever you form a new hypothesis about the data (before asking any questions or providing any answers).
You may not need to say anything: the user might have already stated the fact you need to collect, so check carefully the entire history.
Do not ask the user to repeat a value if you already know it. You can call `{update_function}` with your hypothesis without saying anything to the user and then ask them to confirm.
Don't invent the content of any of the fields, stick strictly to what the user said. 
Call `{confirm_function}` after the user confirmed the data is correct. 
Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary. 
Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called."""


@dataclass
class CardHolderName:
    full_name: str


@dataclass
class CardNumber:
    number: str


@dataclass
class CardExpirationDate:
    month: int
    year: int


@dataclass
class CardSecurityCode:
    code: str


CardTypes = Literal[CARD_PREFIXES.keys()]


class CreditCardData:

    def __init__(
        self,
        holder_name: CardHolderName | dict | None = None,
        number: CardNumber | dict | None = None,
        expiration_date: CardExpirationDate | dict | None = None,
        security_code: CardSecurityCode | dict | None = None,
    ):
        self.type: Literal["Visa", "MasterCard", "AMEX", "Discover", "JCB", "Diners Club", "UnionPay"] | None = None
        self.holder_name = CardHolderName(**holder_name) if isinstance(holder_name, dict) else holder_name
        self._number = CardNumber(**number) if isinstance(number, dict) else number
        self.expiration_date = CardExpirationDate(**expiration_date) if isinstance(expiration_date, dict) else expiration_date
        self.security_code = CardSecurityCode(**security_code) if isinstance(security_code, dict) else security_code

    @property
    def number(self):
        return self._number

    @number.setter
    def number(self, number: CardNumber):
        self._number = number
        for card_type, card_prefixes in CARD_PREFIXES.items():
            if any(number.number.startswith(card_prefix) for card_prefix in card_prefixes):
                self._type = card_type
                break

    def __str__(self):
        string = ""
        string += "Card Type: " + self.type if self.type else "<unknown>"
        string += " Card Holder Name: " + (self.holder_name.full_name if self.holder_name else "<missing>")
        string += " Card Number: " + (self.number.number if self.number else "<missing>")
        string += " Card Expiration Date: " + (f"{self.expiration_date.month}/{self.expiration_date.year}" if self.expiration_date else "<missing>")
        string += " Card Security Code: " + (self.security_code.code if self.security_code else "<missing>")
        return string

    def is_empty(self) -> bool:
        return bool(
            not self.holder_name
            and not self.number
            and not self.expiration_date
            and not self.security_code
        )

    def is_complete(self) -> bool:
        return bool(
            self.holder_name and self.number and self.expiration_date and self.security_code
        )

    @property
    def missing_fields(self) -> set[str]:
        missing = set()
        if self.holder_name is None:
            missing.add("holder_name")
        if self.number is None:
            missing.add("number")
        if self.expiration_date is None:
            missing.add("expiration_date")
        if self.security_code is None:
            missing.add("security_code")
        return missing

    @property
    def populated_fields(self) -> list[str]:
        return {"holder_name", "number", "expiration_date", "security_code"} - self.missing_fields
    
    async def to_json(self) -> None:
        raw_data = {
            "holder_name": asdict(self.holder_name) if self.holder_name else None,
            "number": asdict(self.number) if self.number else None,
            "expiration_date": asdict(self.expiration_date) if self.expiration_date else None,
            "security_code": asdict(self.security_code) if self.security_code else None,
        }
        async with aiofiles.open(USER_DATA_PATH, "w", newline="") as f:
            await f.write(json.dumps(raw_data))
    
    @staticmethod
    async def from_json() -> None:
        if not os.path.exists(USER_DATA_PATH):
            return CreditCardData()
        async with aiofiles.open(USER_DATA_PATH, "r", newline="") as f:
            try:
                raw_data = json.loads(await f.read())
                return CreditCardData(**raw_data)
            except Exception as e:
                logger.exception(e)
                return CreditCardData()


class GetCreditCardData(AgentTask[CreditCardData]):
    """
    Gets the credit card information.
    """
    def __init__(
        self,
        extra_instructions: str = "",
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.FunctionTool | llm.RawFunctionTool]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=(dedent("""\
                You are only a single step in a broader system, responsible solely for capturing a credit card's data.
                Handle input as noisy voice transcription.
                Don't invent the content of any of the fields, stick strictly to what the user said. 
                Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary. 
                When collecting credit card data from scratch, you can follow the order holder_name - number - expiration_date - security code.
                Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called."""
                ) + extra_instructions
            ),
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            tools=tools,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )

    async def on_enter(self) -> None:

        if self.session.userdata.is_complete():
            await self.session.generate_reply(instructions=dedent(f"""\
                Tell them that you found a complete {self.session.userdata.type} credit card profile associated to them. 
                Ask if they want you to read the fields out loud. Don't forget to mention clearly what *type* of card it is!"""))
        if self.session.userdata.is_empty():
            await self.session.generate_reply(instructions=dedent("""\
                Tell them that you found no credit card associated to them and you need to collect their credit card 
                information. Then, tell them that they can either list all the card information now or say the fields
                one by one."""))
        else:
            await self.session.generate_reply(instructions=dedent(f"""\
                Tell them that you found only part of their credit card information, specifically {self.session.userdata.populated_fields},
                so you need to collect the rest of the data. If the type is available (type: {self.session.userdata.type}), specify it. 
                Then, tell them that they can either list all the missing card information now or say the fields one by one."""))

    @function_tool
    async def collect_holder_name(self, ctx: RunContext):
        """
        Collect credit card holder name only.
        """
        if not (holder_name := await GetCreditCardHolderName(chat_ctx=self.chat_ctx)):
            await self.session.generate_reply(
                instructions="Inform the user that you are unable to proceed and will end the call."
            )
            if not self.done():
                self.complete(
                    ToolError(
                        f"couldn't get cardholder name (reason: {holder_name})"
                    )
                )
            return
        self.session.userdata.holder_name = holder_name
        return f"You got the card holder name as {holder_name}"

    @function_tool
    async def collect_card_number(self, ctx: RunContext):
        """
        Collect credit card number only.
        """
        ctx.wait_for_playout()
        if not (cc_number := await GetCreditCardNumber(chat_ctx=self.chat_ctx)):
            await self.session.generate_reply(
                instructions="Inform the user that you are unable to proceed and will end the call."
            )
            if not self.done():
                self.complete(
                    ToolError(
                        f"couldn't get card number (reason: {cc_number})"
                    )
                )
            return
        self.session.userdata.number = cc_number
        return f"You got the card number as {cc_number}"

    @function_tool
    async def collect_expiration_date(self, ctx: RunContext):
        """
        Collect the expiration date only.
        """
        if not (expiration_date := await GetCreditCardExpirationDate(chat_ctx=self.chat_ctx)):
            await self.session.generate_reply(
                instructions="Inform the user that you are unable to proceed and will end the call."
            )
            if not self.done():
                self.complete(
                    ToolError(
                        f"couldn't get expiration date (reason: {expiration_date})"
                    )
                )
            return
        self.session.userdata.expiration_date = expiration_date
        return f"You got the card expiration date as {expiration_date}"

    @function_tool
    async def collect_security_code(self, ctx: RunContext):
        """
        Collect the esecurity code only.
        """
        if not (security_code := await GetCreditCardSecurityCode(chat_ctx=self.chat_ctx)):
            await self.session.generate_reply(
                instructions="Inform the user that you are unable to proceed and will end the call."
            )
            if not self.done():
                self.complete(
                    ToolError(
                        f"couldn't get card security code (reason: {security_code})"
                    )
                )
            return
        self.session.userdata.security_code = security_code
        return f"You got the card security code as {security_code}"

    @function_tool
    async def read_current_cc_data(self, ctx: RunContext):
        """
        If the user asks you what data you have about their card, use this tool and read the results to them.
        """
        return str(self.session.userdata)

    @function_tool
    async def replace_cc_card(self, ctx: RunContext):
        """
        If the user wants to change credit card, use this tool to delete the previous card's data.
        You'll then collect all the fields again.
        """
        self.session.userdata = CreditCardData()
        return "The previous card data was wiped, you can now collect the new card's data."

    @function_tool
    async def check_completion(self, ctx: RunContext):
        """
        Call this tool when you believe you have finished collecting credit card information from the user.
        You must have the cardholder's name, the card number, the expiration date and the security code.
        If any is missing, you should call the appropriate tool to collect that information before calling this one.
        """
        if self.session.userdata.is_complete():
            code = self.session.userdata.security_code.code
            if not((self.session.userdata.type == "AMEX" and len(code) == 4) or len(code) == 3):
                raise ToolError(f"Invalid credit card security code provided for {self.session.userdata.type} card: {code}")

            await self.session.userdata.to_json()
            self.complete(f"User's credit card data collected successfully.")
        return ToolError(
            f"You did not collect these fields yet: {str(self.session.userdata.missing_fields)}"
        )
    
    @function_tool()
    async def decline_data_capture(self, reason: str) -> None:
        """
        Handles the case when the user explicitly declines to provide the card's data. 
        Call this tool also when `collect_data` fails.

        Args:
            reason: A short explanation of why the user declined to provide the card's data.
        """
        if not self.done():
            await self.session.userdata.to_json()
            self.complete(ToolError(f"couldn't get the full card data: {reason}. Partial credit card info might have been saved."))


class GetCreditCardHolderName(AgentTask[CardHolderName]):
    """
    Task that collects credit card holder name
    """
    def __init__(
        self,
        extra_instructions: str = "",
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.FunctionTool | llm.RawFunctionTool]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=(SINGLE_FIELD_PROMPT.format(
                    data_name = "credit card's holder name",
                    example_formats = "\n".join(["- 'theo t h e o smith' (name followed by spelling)"]),
                    update_function = "update_name",
                    confirm_function = "confirm_name"
                ) + extra_instructions
            ),
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            tools=tools,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )
        self._name = ""

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Ask the user to provide the name of the credit card holder, as spelled on the card.")

    @function_tool
    async def update_name(self, name: str, ctx: RunContext) -> str:
        """Update the name provided by the user.

        Args:
            name: The name provided by the user
        """
        self._name = name.strip()

        return (
            f"The card holder name has been updated to {self._name}\n"
            f"Repeat the full name, first as written and then spelling it character by character.\n"
            f"Prompt the user for confirmation, do not call `confirm_name` directly."
        )

    @function_tool()
    async def confirm_name(self, ctx: RunContext) -> None:
        """Validates/confirms the name provided by the user."""
        await ctx.wait_for_playout()

        if not self._name.strip():
            raise ToolError(
                "error: no name was provided, `update_name` must be called before"
            )

        if not self.done():
            self.complete(CardHolderName(full_name=self._name))

    @function_tool()
    async def decline_name_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide the card holder's name.

        Args:
            reason: A short explanation of why the user declined to provide the card holder's name
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the card holder name: {reason}"))


class GetCreditCardNumber(AgentTask[CardNumber]):
    """
    Task that collects the credit card number
    """
    def __init__(
        self,
        extra_instructions: str = "",
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.FunctionTool | llm.RawFunctionTool]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=(SINGLE_FIELD_PROMPT.format(
                    data_name = "credit card's number",
                    example_formats = "\n".join(["- '1 3 6 8 3 ...' (digit by digit)", "- '13 68 33 ...' (in pairs)", "- '1368 3003 ...' (in groups of four)", "- '1 3 6 8 six zeros 3 3 ...' (grouping similar digits)", "- other arbitrary combinations"]),
                    update_function = "update_number",
                    confirm_function = "confirm_number"
                ) + extra_instructions
            ),
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            tools=tools,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )
        self._number = ""

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Ask the user to provide the number of the credit card, as shown on the card.")

    @function_tool
    async def update_number(self, number: str, ctx: RunContext) -> str:
        """Update the credit card number provided by the user.

        Args:
            number: The credit card number provided by the user
        """
        number = number.replace(" ", "")

        if len(number) < 16:
            raise ToolError(f"Credit card number is missing {16 - len(number)} digits: {number}")
        elif len(number) > 16:
            raise ToolError(f"Credit card number has {len(number) - 16} excess digits: {number}")
        elif not lf.isvalid(number):
            raise ToolError(f"Invalid credit card number provided: {number}")
            
        self._number = number
        return (
            f"The credit card number has been updated to {self._number}\n"
            f"Repeat the full number, digit by digit.\n"
            f"Prompt the user for confirmation, do not call `confirm_number` directly."
        )

    @function_tool()
    async def confirm_number(self, ctx: RunContext) -> None:
        """Validates/confirms the number provided by the user."""
        await ctx.wait_for_playout()

        if not self._number.strip():
            raise ToolError("error: no number was provided, `update_number` must be called before")

        if not self.done():
            self.complete(CardNumber(number=self._number))

    @function_tool()
    async def decline_number_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide the card number.

        Args:
            reason: A short explanation of why the user declined to provide the card number
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the card number: {reason}"))


class GetCreditCardExpirationDate(AgentTask[CardExpirationDate]):
    """
    Task that collects the credit card expiration date
    """
    def __init__(
        self,
        extra_instructions: str = "",
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.FunctionTool | llm.RawFunctionTool]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=(SINGLE_FIELD_PROMPT.format(
                    data_name = "credit card's expiration date",
                    example_formats = "\n".join([
                        "- 'november two thousand twenty six' ( which means 'month: 11, year: 2026', using the month's name)", 
                        "- 'november twenty six' ( which means 'month: 11, year: 2026', using the month's name and short year)", 
                        "- 'zero nine slash two six' (which means 'month: 9, year: 2026', reading the string)", 
                        "- 'zero nine two six' (which means 'month: 9, year: 2026', forgetting the slash)"
                    ]),
                    update_function = "update_date",
                    confirm_function = "confirm_date"
                ) + extra_instructions
            ),
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            tools=tools,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )
        self._date = None

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Ask the user to provide the expiration date of the credit card, as shown on the card.")

    @function_tool
    async def update_expiration_date(self, month: int, year: int, ctx: RunContext) -> str:
        """Update the credit card expiration date provided by the user.

        Args:
            month: The credit card expiration month provided by the user
            year: The credit card expiration year provided by the user
        """
        current_year = datetime.datetime.now().year
        current_month = datetime.datetime.now().month

        if year < 100: # the LLM returned only two digits, like "25" for "2025"
            year += 2000

        if month > 12:
            raise ToolError(f"Invalid credit card expiration date provided: {month}/{year}")
        
        if year < current_year or (year == current_year and month < current_month):
            raise ToolError(f"Invalid credit card expiration date provided: {month}/{year}")
        
        self._date = datetime.datetime(year=year, month=month, day=1)
        return (
            f"The credit card expiration date has been updated to {self._date}\n"
            f"Repeat the month (by name) and then the year (full length).\n"
            f"Prompt the user for confirmation, do not call `confirm_date` directly."
        )

    @function_tool()
    async def confirm_date(self, ctx: RunContext) -> None:
        """Validates/confirms the expiration date provided by the user."""
        await ctx.wait_for_playout()

        if not self._date:
            raise ToolError("error: no expiration date was provided, `update_date` must be called before")

        if not self.done():
            self.complete(CardExpirationDate(month=self._date.month, year=self._date.year))

    @function_tool()
    async def decline_date_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide the card expiration date.

        Args:
            reason: A short explanation of why the user declined to provide the card expiration date
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the card expiration date: {reason}"))


class GetCreditCardSecurityCode(AgentTask[CardSecurityCode]):
    """
    Task that collects the credit card security code
    """
    def __init__(
        self,
        extra_instructions: str = "",
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.FunctionTool | llm.RawFunctionTool]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=(SINGLE_FIELD_PROMPT.format(
                    data_name = "credit card's security code",
                    example_formats = "\n".join([
                        "- '1 3 6' (digit by digit)", 
                        "- '136' (as a single number)"
                    ]),
                    update_function = "update_code",
                    confirm_function = "confirm_code"
                ) + extra_instructions
            ),
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            tools=tools,
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )
        self._code = ""

    async def on_enter(self) -> None:
        await self.session.generate_reply(instructions="Ask the user to provide the security code of the credit card, as shown on the card.")

    @function_tool
    async def update_code(self, code: str, ctx: RunContext) -> str:
        """Update the credit card security code provided by the user.

        Args:
            code: The credit card security code provided by the user
        """
        code = code.strip()

        if not((self.session.userdata.type in ["AMEX", None] and len(code) == 4) or len(code) == 3):
            raise ToolError(f"Invalid credit card security code provided: {code}")

        self._code = code
        return (
            f"The credit card security code has been updated to {self._code}\n"
            f"Repeat the full security code, digit by digit.\n"
            f"Prompt the user for confirmation, do not call `confirm_code` directly."
        )

    @function_tool()
    async def confirm_code(self, ctx: RunContext) -> None:
        """Validates/confirms the security code provided by the user."""
        await ctx.wait_for_playout()

        if not self._code.strip():
            raise ToolError("error: no security code was provided, `update_code` must be called before")

        if not self.done():
            self.complete(CardSecurityCode(code=self._code))

    @function_tool()
    async def decline_code_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide the card security code.

        Args:
            reason: A short explanation of why the user declined to provide the card security code
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the card security code: {reason}"))


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=dedent("""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor. Do not switch language during the call unless the user explicitly asks for it.
            Before engaging with the user you must make sure that their credit card details are up to date by calling `collect_cc_data`. 
            If the user wants to modify them later on during the chat, you can always call `collect_cc_data` again."""),
        )

    async def on_enter(self) -> None:
        self.session.userdata = CreditCardData()
        await self.session.generate_reply(instructions="Greet the user and introduce yourself. Ask them if they're ready to begin.")

    @function_tool
    async def collect_cc_data(self, context: RunContext):
        """
        Use this tool to make sure the user's credit card information is up to date, and every time the user asks anything about their credit card.
        """
        self.session.userdata = await CreditCardData.from_json()
        try:
            if await GetCreditCardData(chat_ctx=self.chat_ctx):
                await self.session.generate_reply(instructions="Offer your assistance to the user.")
            else:
                await self.terminate_call()
        except ToolError as e:
            await self.terminate_call()

    async def terminate_call(self):
        await self.session.generate_reply(instructions="Inform the user that you are unable to proceed and will end the call.")
        job_ctx = get_job_context()
        await job_ctx.api.room.delete_room(api.DeleteRoomRequest(room=job_ctx.room.name))


async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        llm=openai.realtime.RealtimeModel(voice="marin")
    )
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
    )
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
