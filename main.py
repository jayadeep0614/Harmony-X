"""
HARMONY-X
Main Multimodal Intelligence Loop

Vision + Audio + Fusion + Context Memory
+ Confidence-Aware Reasoning
+ Dataset Recording
"""

import cv2

from src.perception.vision import VisionProcessor
from src.audio.audio_processor import AudioProcessor
from src.fusion.multimodal_fusion import MultimodalFusion
from src.reasoning.reasoning_engine import ReasoningEngine
from src.reasoning.context_memory import ContextMemory
from src.reasoning.dataset_recorder import DatasetRecorder


def main():

    print()
    print("=" * 70)
    print("HARMONY-X")
    print("Multimodal Adaptive Intelligence System")
    print("=" * 70)
    print()

    # ---------------------------------------------------------
    # INITIALIZE MODULES
    # ---------------------------------------------------------

    vision = VisionProcessor()
    audio = AudioProcessor(
        sample_rate=48000,
        channels=2,
        device=9,
        block_duration=0.1,
        analysis_duration=0.5,
    )

    fusion = MultimodalFusion()

    reasoning = ReasoningEngine(
        confidence_threshold=0.70
    )

    memory = ContextMemory(
        max_history=50
    )

    recorder = DatasetRecorder()

    # ---------------------------------------------------------
    # INITIALIZE SYSTEM
    # ---------------------------------------------------------

    try:

        vision.initialize()

        audio.initialize()

        fusion.initialize()

        reasoning.initialize()

        memory.initialize()

        recorder.initialize()

        print()
        print("=" * 70)
        print("HARMONY-X SYSTEM READY")
        print("=" * 70)
        print()
        print("Press Q to quit.")
        print()

        frame_count = 0

        # -----------------------------------------------------
        # LIVE LOOP
        # -----------------------------------------------------

        while True:

            success, frame = vision.camera.read()

            if not success:
                print("Failed to read camera frame.")
                break

            frame_count += 1

            # -------------------------------------------------
            # VISION
            # -------------------------------------------------

            vision_result = vision.process(frame)

            # -------------------------------------------------
            # AUDIO
            # -------------------------------------------------

            audio_result = audio.process()

            # -------------------------------------------------
            # MULTIMODAL FUSION
            # -------------------------------------------------

            fusion_result = fusion.fuse(
                vision_result,
                audio_result
            )

            # -------------------------------------------------
            # REASONING
            # -------------------------------------------------

            reasoning_result = reasoning.reason(
                fusion_result,
                memory
            )

            # -------------------------------------------------
            # MEMORY
            # -------------------------------------------------

            memory.add_state(
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

            # -------------------------------------------------
            # MEMORY SUMMARY
            # -------------------------------------------------

            memory_summary = memory.summary()

            # -------------------------------------------------
            # DATASET RECORDING
            # -------------------------------------------------

            recorder.record(
                fusion_result=
                    fusion_result,

                reasoning_result=
                    reasoning_result,

                memory_summary=
                    memory_summary,
            )

            # -------------------------------------------------
            # TERMINAL OUTPUT
            # -------------------------------------------------

            print(
                f"\r"
                f"Frame: {frame_count:04d} | "
                f"Confidence: "
                f"{fusion_result.overall_confidence:.2f} | "
                f"Level: "
                f"{reasoning_result.confidence_level:<6} | "
                f"Uncertainty: "
                f"{reasoning_result.uncertainty:<8} | "
                f"Emotion: "
                f"{fusion_result.emotion:<9} | "
                f"Speech: "
                f"{str(fusion_result.speech_detected):<5} | "
                f"Action: "
                f"{reasoning_result.action:<17}",
                end="",
                flush=True,
            )

            # -------------------------------------------------
            # DISPLAY
            # -------------------------------------------------

            cv2.putText(
                frame,
                "HARMONY-X",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Confidence: "
                f"{fusion_result.overall_confidence:.2f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Level: "
                f"{reasoning_result.confidence_level}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Uncertainty: "
                f"{reasoning_result.uncertainty}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Emotion: "
                f"{fusion_result.emotion}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Speech: "
                f"{fusion_result.speech_detected}",
                (20, 215),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Action: "
                f"{reasoning_result.action}",
                (20, 250),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Priority: "
                f"{reasoning_result.priority}",
                (20, 285),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Memory: "
                f"{memory_summary['memory_size']}/"
                f"{memory_summary['memory_capacity']}",
                (20, 320),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Dataset: "
                f"{recorder.record_count}",
                (20, 355),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "HARMONY-X Multimodal Intelligence",
                frame
            )

            # -------------------------------------------------
            # QUIT
            # -------------------------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        print()
        print()
        print("=" * 70)
        print("HARMONY-X SHUTDOWN")
        print("=" * 70)

        print()
        print("Final Memory Summary:")
        print(memory.summary())

        print()
        print("Dataset Status:")
        print(recorder.status())

        print()

        audio.release()
        vision.release()

        print("HARMONY-X shutdown complete.")


if __name__ == "__main__":
    main()