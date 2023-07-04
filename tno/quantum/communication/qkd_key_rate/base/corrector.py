"""Base class for error corrector objects."""
from __future__ import annotations

import abc
import hashlib
import hmac
from dataclasses import dataclass, fields
from typing import Optional, Tuple

from tno.quantum.communication.qkd_key_rate.base import (
    Message,
    ReceiverBase,
    SenderBase,
)


@dataclass
class CorrectorOutputBase:
    """Base class corrector summary object

    Args:
        input_alice: Input message Alice
        output_alice: Corrected message Alice
        input_bob: Input message Bob
        output_bob: Corrected message Bob
        input_error: Input error rate
        output_error: Output error rate
        output_length: Output message length
        number_of_exposed_bits: Number of bits exposed in protocol
        key_reconciliation_rate: Key reconciliation efficiency
        number_of_communication_rounds: Number of communication rounds
    """

    input_alice: Message
    output_alice: Message
    input_bob: Message
    output_bob: Message
    input_error: float
    output_error: float
    output_length: int
    number_of_exposed_bits: int
    key_reconciliation_rate: float
    number_of_communication_rounds: int

    def __str__(self) -> str:
        res = "\nCorrector summary:"
        for field in fields(self):
            res += f"\n{field.name} ({field.type}):\t {getattr(self, field.name)}"
        return res


class Corrector(metaclass=abc.ABCMeta):
    """Error corrector base class."""

    def __init__(
        self,
        alice: SenderBase,
        bob: ReceiverBase,
    ) -> None:
        """Base class for error correcting

        Args:
            Alice: The sending party
            Bob: The receiving party
        """
        self.alice = alice
        self.bob = bob

    def correct_errors(
        self, detail_transcript: Optional[bool] = False
    ) -> CorrectorOutputBase:
        """Receiver Bob corrects the errors based on Alice her message.

        Args:
            detail_transcript: Whether to print a detailed transcript
        """
        self.bob.correct_errors(self.alice)

        if detail_transcript:
            print(self.bob.transcript)

        return self.summary()

    @abc.abstractmethod
    def summary(self) -> CorrectorOutputBase:
        """
        Calculate a summary object for the error correction containing
            - original message
            - corrected message
            - error rate (before and after correction)
            - number_of_exposed_bits
            - key_reconciliation_rate
            - protocol specific parameters
        """

    @staticmethod
    def calculate_number_of_errors(message1: Message, message2: Message) -> int:
        """Calculate the error rate between two messages
        If messages differ in length, the number of errors is calculated
        using the number of bits of the shortest message.

        Args:
            message1: First message
            message2: Second message

        Returns:
            number_of_errors: Number of errors.
        """
        assert message1.length != 0 and message2.length != 0
        return sum((x != y for (x, y) in zip(message1.message, message2.message)))

    @staticmethod
    def calculate_error_rate(message1: Message, message2: Message) -> float:
        """Calculate the error rate between two messages.

        If messages differ in length, the number of errors is calculated
        using the number of bits of the shortest message.

        Args:
            message1: First message
            message2: Second message

        Returns:
            error_rate: Ratio of errors over the message length.
        """
        return Corrector.calculate_number_of_errors(message1, message2) / min(
            message1.length, message2.length
        )

    def calculate_key_reconciliation_rate(self, exposed_bits: bool = False) -> float:
        """Calculate the key reconciliation rate.

        Args:
            exposed_bits: If true, uses the number of exposed bits to compute the
            key-reconciliation rate. Otherwise, uses the ratio between the in- and
            output message length.

        Returns:
            key_rate: The reconciliation rate
        """
        if exposed_bits:
            key_rate = (
                self.alice.message.length - self.alice.net_exposed_bits
            ) / self.alice.original_message.length
        else:
            key_rate = self.alice.message.length / self.alice.original_message.length
        return key_rate

    @staticmethod
    def create_message_tag_pair(
        message: Message, shared_key: str
    ) -> Tuple[bytes, bytes]:
        """Prepares a message-tag hashed pair.

        The message can be communicated publicly. The tag is the hash of the message,
        given a key.

        Args:
            message: To be communicated message
            key: Shared secret key

        Returns:
            message: To be communicated message
            tag: Hash of the message, given the key, with length of the key
        """
        message_str = "".join(str(x) for x in message.message)
        shared_key = bytes(shared_key.encode("utf-8"))
        message_bytes = bytes(message_str.encode("utf-8"))

        tag = hmac.new(key=shared_key, msg=message_bytes, digestmod=hashlib.sha384)
        return message_bytes, tag.digest()
