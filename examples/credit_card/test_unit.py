"""
Unit tests for credit card validation tools.
"""
import pytest
import datetime
from unittest.mock import Mock, PropertyMock

from livekit.agents.llm.tool_context import ToolError

from .credit_card import (
    CreditCardData,
    GetCreditCardNumber,
    GetCreditCardSecurityCode,
    GetCreditCardExpirationDate,
)


def _create_task_with_userdata(task_class, userdata=None):
    """
    Create a task instance with a properly mocked session.userdata.

    Args:
        task_class: The task class to instantiate
        userdata: Optional CreditCardData to use (defaults to empty CreditCardData)

    Returns:
        task: The task instance with mocked session property
    """
    task = task_class()

    # Create a mock session with userdata
    session = Mock()
    session.userdata = userdata or CreditCardData()

    # Mock the session property getter to return our mock session
    type(task).session = PropertyMock(return_value=session)

    return task


@pytest.mark.asyncio
async def test_card_number_luhn_validation() -> None:
    """Test that update_number validates card numbers using Luhn algorithm."""
    task = GetCreditCardNumber()
    # Valid card number (Visa test number)
    result = await task.update_number("4532015112830366", None)
    assert "updated" in result.lower()

    # Invalid card number (fails Luhn check)
    task_invalid = GetCreditCardNumber()
    with pytest.raises(ToolError, match="Invalid credit card number"):
        await task_invalid.update_number("4532015112830367", None)  # Changed last digit


@pytest.mark.asyncio
async def test_card_number_length_validation() -> None:
    """Test that card numbers must be exactly 16 digits."""
    # Too short
    task_short = GetCreditCardNumber()
    with pytest.raises(ToolError, match="missing 4 digits"):
        await task_short.update_number("453201511283", None)

    # Too long
    task_long = GetCreditCardNumber()
    with pytest.raises(ToolError, match="1 excess digits"):
        await task_long.update_number("45320151128303661", None)


@pytest.mark.asyncio
async def test_security_code_validation_amex() -> None:
    """Test that AMEX cards require 4-digit security codes."""
    userdata = CreditCardData(number={"number": "378282246310005"})  # AMEX test number
    task = _create_task_with_userdata(GetCreditCardSecurityCode, userdata)

    # AMEX should accept 4 digits
    result = await task.update_code("1234", None)
    assert "updated" in result.lower()

    # AMEX should reject 3 digits (not 4)
    task_2 = _create_task_with_userdata(GetCreditCardSecurityCode, userdata)
    with pytest.raises(ToolError, match="Invalid credit card security code"):
        await task_2.update_code("123", None)

    # AMEX should reject 5 digits
    task_3 = _create_task_with_userdata(GetCreditCardSecurityCode, userdata)
    with pytest.raises(ToolError, match="Invalid credit card security code"):
        await task_3.update_code("12345", None)


@pytest.mark.asyncio
async def test_security_code_validation_visa() -> None:
    """Test that non-AMEX cards require 3-digit security codes."""
    userdata = CreditCardData(number={"number": "4532015112830366"})  # Visa test number
    task = _create_task_with_userdata(GetCreditCardSecurityCode, userdata)

    # Visa should accept 3 digits
    result = await task.update_code("123", None)
    assert "updated" in result.lower()

    # Visa should reject 4 digits
    task_2 = _create_task_with_userdata(GetCreditCardSecurityCode, userdata)
    with pytest.raises(ToolError, match="Invalid credit card security code"):
        await task_2.update_code("1234", None)

    # Visa should reject 2 digits
    task_3 = _create_task_with_userdata(GetCreditCardSecurityCode, userdata)
    with pytest.raises(ToolError, match="Invalid credit card security code"):
        await task_3.update_code("12", None)


@pytest.mark.asyncio
async def test_security_code_accepts_3_or_4_digits_when_card_type_unknown() -> None:
    """Test that both 3 and 4-digit security codes are accepted when card type is not yet known."""
    # Should accept 4 digits when card type is unknown (could be AMEX)
    task = _create_task_with_userdata(GetCreditCardSecurityCode)
    result = await task.update_code("1234", None)
    assert "updated" in result.lower()

    # Should also accept 3 digits when card type is unknown (could be Visa/MC/etc)
    task_2 = _create_task_with_userdata(GetCreditCardSecurityCode)
    result = await task_2.update_code("123", None)
    assert "updated" in result.lower()

    # Should reject 2 digits
    task_3 = _create_task_with_userdata(GetCreditCardSecurityCode)
    with pytest.raises(ToolError, match="Invalid credit card security code"):
        await task_3.update_code("12", None)

    # Should reject 5 digits
    task_4 = _create_task_with_userdata(GetCreditCardSecurityCode)
    with pytest.raises(ToolError, match="Invalid credit card security code"):
        await task_4.update_code("12345", None)


@pytest.mark.asyncio
async def test_expiration_date_validation_expired() -> None:
    """Test that expired cards are detected."""
    task = GetCreditCardExpirationDate()

    # Test expired year
    with pytest.raises(ToolError, match="Invalid credit card expiration date"):
        await task.update_expiration_date(12, 2020, None)

    # Test expired month in current year
    current_year = datetime.datetime.now().year
    current_month = datetime.datetime.now().month
    if current_month > 1:
        task_2 = GetCreditCardExpirationDate()
        with pytest.raises(ToolError, match="Invalid credit card expiration date"):
            await task_2.update_expiration_date(current_month - 1, current_year, None)


@pytest.mark.asyncio
async def test_expiration_date_validation_valid() -> None:
    """Test that future expiration dates are accepted."""
    task = GetCreditCardExpirationDate()

    # Test valid future date
    future_year = datetime.datetime.now().year + 2
    result = await task.update_expiration_date(12, future_year, None)
    assert "updated" in result.lower()

    # Test two-digit year conversion
    task_2 = GetCreditCardExpirationDate()
    result = await task_2.update_expiration_date(12, 26, None)  # Should convert to 2026
    assert "2026" in result


@pytest.mark.asyncio
async def test_expiration_date_validation_invalid_month() -> None:
    """Test that invalid months are rejected."""
    task = GetCreditCardExpirationDate()

    future_year = datetime.datetime.now().year + 1

    # Month > 12
    with pytest.raises(ToolError, match="Invalid credit card expiration date"):
        await task.update_expiration_date(13, future_year, None)


@pytest.mark.asyncio
async def test_expiration_date_current_month_accepted() -> None:
    """Test that current month in current year is accepted."""
    task = GetCreditCardExpirationDate()

    current_year = datetime.datetime.now().year
    current_month = datetime.datetime.now().month

    # Current month should be valid
    result = await task.update_expiration_date(current_month, current_year, None)
    assert "updated" in result.lower()


def test_is_complete() -> None:
    """Test that is_complete correctly identifies complete card data."""
    # Incomplete - missing all fields
    cc_data = CreditCardData()
    assert not cc_data.is_complete()
    assert cc_data.is_empty()

    # Incomplete - missing some fields
    cc_data = CreditCardData(
        holder_name={"full_name": "John Smith"},
        number={"number": "4532015112830366"},
    )
    assert not cc_data.is_complete()
    assert not cc_data.is_empty()

    # Complete
    cc_data = CreditCardData(
        holder_name={"full_name": "John Smith"},
        number={"number": "4532015112830366"},
        expiration_date={"month": 12, "year": 2026},
        security_code={"code": "123"},
    )
    assert cc_data.is_complete()
    assert not cc_data.is_empty()


def test_missing_fields() -> None:
    """Test that missing_fields correctly identifies which fields are missing."""
    cc_data = CreditCardData(
        holder_name={"full_name": "John Smith"},
        number={"number": "4532015112830366"},
    )

    missing = cc_data.missing_fields
    assert "expiration_date" in missing
    assert "security_code" in missing
    assert "holder_name" not in missing
    assert "number" not in missing
