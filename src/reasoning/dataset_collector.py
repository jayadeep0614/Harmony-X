"""
HARMONY-X
Controlled Multimodal Dataset Collector

Collects controlled sessions and records:

1. session_label
   The condition intentionally performed by the user.

2. emotion
   The expression predicted by HARMONY-X.

This separation allows later evaluation of
the current expression-analysis heuristic.
"""

import time

import cv2

from src.perception.vision import VisionProcessor
from src.audio.audio_processor import AudioProcessor
from src.fusion.multimodal_fusion import MultimodalFusion
from src.reasoning.reasoning_engine import ReasoningEngine
from src.reasoning.context_memory import ContextMemory
from src.reasoning.dataset_recorder import DatasetRecorder


SESSION_OPTIONS = {
    "1": "NEUTRAL",
    "2": "SPEAKING",
    "3": "HAPPY",
    "4": "SURPRISED",
    "5": "SAD",
    "6": "ANGRY",
}


class DatasetCollector:

    def __init__(self):

        self.vision = VisionProcessor()

        self.audio = AudioProcessor(
            sample_rate=48000,
            channels=2,
            device=9,
            block_duration=0.1,
            analysis_duration=0.5,
        )

        self.fusion = MultimodalFusion()

        self.reasoning = ReasoningEngine(
            confidence_threshold=0.70
        )

        self.memory = ContextMemory(
            max_history=50
        )

        self.recorder = DatasetRecorder()

        self.initialized = False

    def initialize(self):

        print()
        print("=" * 70)
        print("HARMONY-X CONTROLLED DATASET COLLECTOR")
        print("=" * 70)
        print()

        print("Initializing Vision...")
        self.vision.initialize()

        print()
        print("Initializing Audio...")
        self.audio.initialize()

        print()
        print("Initializing Fusion...")
        self.fusion.initialize()

        print()
        print("Initializing Reasoning...")
        self.reasoning.initialize()

        print()
        print("Initializing Context Memory...")
        self.memory.initialize()

        print()
        print("Initializing Dataset Recorder...")
        self.recorder.initialize()

        self.initialized = True

        print()
        print("=" * 70)
        print("ALL SYSTEMS READY")
        print("=" * 70)

    def choose_session(self):

        print()
        print("=" * 70)
        print("SELECT DATA COLLECTION SESSION")
        print("=" * 70)
        print()

        print("1. Neutral / Silent")
        print("2. Speaking")
        print("3. Happy-like expression")
        print("4. Surprised-like expression")
        print("5. Sad-like expression")
        print("6. Angry-like expression")
        print()

        while True:

            choice = input(
                "Enter session number (1-6): "
            ).strip()

            if choice in SESSION_OPTIONS:
                return SESSION_OPTIONS[choice]

            print(
                "Invalid choice. Enter a number from 1 to 6."
            )

    def collect_session(
        self,
        session_label,
        duration=60
    ):

        print()
        print("=" * 70)
        print(
            f"SESSION LABEL: {session_label}"
        )
        print("=" * 70)

        print()
        print(
            f"Duration: {duration} seconds"
        )

        print(
            "Press Q at any time to stop."
        )

        print()

        input(
            "Press ENTER to start the session..."
        )

        print()
        print(
            f"Starting {session_label} session..."
        )

        time.sleep(2)

        self.memory.clear()

        start_time = time.time()

        frame_count = 0
        records_before = self.recorder.record_count

        while True:

            elapsed = time.time() - start_time

            if elapsed >= duration:
                break

            success, frame = (
                self.vision.camera.read()
            )

            if not success:

                print(
                    "\nFailed to read camera frame."
                )

                break

            try:

                # -----------------------------
                # Vision
                # -----------------------------

                vision_result = (
                    self.vision.process(frame)
                )

                # -----------------------------
                # Audio
                # -----------------------------

                audio_result = (
                    self.audio.process()
                )

                # -----------------------------
                # Multimodal Fusion
                # -----------------------------

                fusion_result = (
                    self.fusion.fuse(
                        vision_result,
                        audio_result
                    )
                )

                # -----------------------------
                # Reasoning
                # -----------------------------

                reasoning_result = (
                    self.reasoning.reason(
                        fusion_result,
                        self.memory
                    )
                )

                # -----------------------------
                # Context Memory
                # -----------------------------

                self.memory.add_state(

                    face_detected=
                    fusion_result.face_detected,

                    hands_detected=
                    fusion_result.hands_detected,

                    pose_detected=
                    fusion_result.pose_detected,

                    emotion=
                    fusion_result.emotion,

                    emotion_confidence=
                    fusion_result.emotion_confidence,

                    speech_detected=
                    fusion_result.speech_detected,

                    volume=
                    fusion_result.volume,

                    overall_confidence=
                    fusion_result.overall_confidence,

                    action=
                    reasoning_result.action,
                )

                memory_summary = (
                    self.memory.summary()
                )

                # -----------------------------
                # Dataset Recording
                # -----------------------------

                self.recorder.record(

                    fusion_result,

                    reasoning_result,

                    memory_summary,

                    session_label=session_label,
                )

                frame_count += 1

                # -----------------------------
                # Display
                # -----------------------------

                remaining = max(
                    0,
                    int(duration - elapsed)
                )

                cv2.putText(
                    frame,
                    "HARMONY-X DATA COLLECTION",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"SESSION LABEL: {session_label}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"MODEL PREDICTION: "
                    f"{fusion_result.emotion}",
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Speech: "
                    f"{fusion_result.speech_detected}",
                    (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Confidence: "
                    f"{fusion_result.overall_confidence:.2f}",
                    (20, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"Remaining: {remaining}s",
                    (20, 210),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow(
                    "HARMONY-X Dataset Collection",
                    frame
                )

                key = (
                    cv2.waitKey(1)
                    & 0xFF
                )

                if key == ord("q"):

                    print()
                    print(
                        "Session stopped by user."
                    )

                    break

            except Exception as exc:

                print(
                    f"\nProcessing error: {exc}"
                )

        records_added = (
            self.recorder.record_count
            - records_before
        )

        print()
        print("=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)

        print(
            f"Session label: {session_label}"
        )

        print(
            f"Frames processed: {frame_count}"
        )

        print(
            f"Records added: {records_added}"
        )

        print(
            f"Total dataset records: "
            f"{self.recorder.record_count}"
        )

        print("=" * 70)

    def release(self):

        try:
            self.audio.release()
        except Exception:
            pass

        try:
            self.vision.release()
        except Exception:
            pass

        cv2.destroyAllWindows()

        print()
        print(
            "Dataset collector resources released."
        )


def main():

    collector = DatasetCollector()

    try:

        collector.initialize()

        while True:

            session_label = (
                collector.choose_session()
            )

            collector.collect_session(
                session_label=session_label,
                duration=60
            )

            print()
            print(
                "Do you want to collect another session?"
            )

            print(
                "Y = Continue"
            )

            print(
                "N = Exit"
            )

            choice = input(
                "Choice: "
            ).strip().lower()

            if choice != "y":
                break

    except KeyboardInterrupt:

        print()
        print(
            "Collection interrupted."
        )

    finally:

        collector.release()


if __name__ == "__main__":
    main()