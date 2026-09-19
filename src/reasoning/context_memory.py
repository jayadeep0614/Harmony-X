"""
HARMONY-X
Context and State Memory

Maintains short-term temporal context from
multimodal observations.
"""

from dataclasses import dataclass
from collections import deque
from datetime import datetime
from typing import Deque, Dict, List


@dataclass
class ContextState:
    """A single multimodal observation."""

    timestamp: str

    face_detected: bool
    hands_detected: bool
    pose_detected: bool

    emotion: str
    emotion_confidence: float

    speech_detected: bool
    volume: float

    overall_confidence: float

    action: str


class ContextMemory:
    """
    Short-term memory for HARMONY-X.

    Stores only the most recent observations so that
    the reasoning layer can detect temporal patterns.
    """

    def __init__(self, max_history: int = 50):

        self.name = "HARMONY-X Context Memory"

        self.max_history = max_history

        self.history: Deque[ContextState] = deque(
            maxlen=max_history
        )

        self.initialized = False

    def initialize(self):

        self.history.clear()

        self.initialized = True

        print(
            f"{self.name}: initialized"
        )

        print(
            f"Memory capacity: "
            f"{self.max_history} observations"
        )

    def add_state(
        self,
        face_detected: bool,
        hands_detected: bool,
        pose_detected: bool,
        emotion: str,
        emotion_confidence: float,
        speech_detected: bool,
        volume: float,
        overall_confidence: float,
        action: str,
    ):

        if not self.initialized:

            raise RuntimeError(
                "ContextMemory has not been initialized."
            )

        state = ContextState(
            timestamp=datetime.now().isoformat(
                timespec="seconds"
            ),

            face_detected=face_detected,

            hands_detected=hands_detected,

            pose_detected=pose_detected,

            emotion=emotion,

            emotion_confidence=
                emotion_confidence,

            speech_detected=speech_detected,

            volume=volume,

            overall_confidence=
                overall_confidence,

            action=action,
        )

        self.history.append(state)

    def get_history(self) -> List[ContextState]:

        return list(self.history)

    def get_recent(
        self,
        count: int = 10
    ) -> List[ContextState]:

        if count <= 0:
            return []

        return list(
            self.history
        )[-count:]

    def observation_count(self) -> int:

        return len(self.history)

    def speech_frequency(self) -> float:

        if not self.history:
            return 0.0

        speech_count = sum(
            state.speech_detected
            for state in self.history
        )

        return speech_count / len(
            self.history
        )

    def face_presence_frequency(self) -> float:

        if not self.history:
            return 0.0

        face_count = sum(
            state.face_detected
            for state in self.history
        )

        return face_count / len(
            self.history
        )

    def average_volume(self) -> float:

        if not self.history:
            return 0.0

        total_volume = sum(
            state.volume
            for state in self.history
        )

        return total_volume / len(
            self.history
        )

    def dominant_emotion(self) -> str:

        if not self.history:
            return "UNKNOWN"

        emotion_counts: Dict[str, int] = {}

        for state in self.history:

            emotion = state.emotion

            if emotion == "UNKNOWN":
                continue

            emotion_counts[emotion] = (
                emotion_counts.get(
                    emotion,
                    0
                ) + 1
            )

        if not emotion_counts:
            return "UNKNOWN"

        return max(
            emotion_counts,
            key=emotion_counts.get
        )

    def summary(self) -> Dict:

        return {
            "memory_size":
                len(self.history),

            "memory_capacity":
                self.max_history,

            "speech_frequency":
                self.speech_frequency(),

            "face_presence_frequency":
                self.face_presence_frequency(),

            "average_volume":
                self.average_volume(),

            "dominant_emotion":
                self.dominant_emotion(),
        }

    def clear(self):

        self.history.clear()

        print(
            "Context memory cleared."
        )

    def status(self) -> Dict:

        return {
            "name": self.name,

            "initialized":
                self.initialized,

            "memory_size":
                len(self.history),

            "memory_capacity":
                self.max_history,
        }


if __name__ == "__main__":

    memory = ContextMemory(
        max_history=5
    )

    memory.initialize()

    # --------------------------------------------------
    # Test observations
    # --------------------------------------------------

    memory.add_state(
        face_detected=True,
        hands_detected=False,
        pose_detected=True,
        emotion="NEUTRAL",
        emotion_confidence=0.75,
        speech_detected=False,
        volume=0.0,
        overall_confidence=0.80,
        action="observe",
    )

    memory.add_state(
        face_detected=True,
        hands_detected=True,
        pose_detected=True,
        emotion="HAPPY",
        emotion_confidence=0.72,
        speech_detected=True,
        volume=0.30,
        overall_confidence=0.82,
        action="respond",
    )

    memory.add_state(
        face_detected=True,
        hands_detected=True,
        pose_detected=True,
        emotion="HAPPY",
        emotion_confidence=0.78,
        speech_detected=True,
        volume=0.40,
        overall_confidence=0.88,
        action="respond",
    )

    # --------------------------------------------------
    # Display test
    # --------------------------------------------------

    print()
    print(
        "Context Memory Status:"
    )

    print(
        memory.status()
    )

    print()

    print(
        "Context Summary:"
    )

    print(
        memory.summary()
    )

    print()

    print(
        "Recent Observations:"
    )

    for state in memory.get_recent(3):

        print(
            state
        )