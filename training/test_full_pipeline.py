"""
HARMONY-X
End-to-End Live Pipeline Test
"""

import time
import cv2

from src.perception.vision import VisionProcessor
from src.audio.audio_processor import AudioProcessor
from src.fusion.multimodal_fusion import MultimodalFusion
from src.reasoning.context_memory import ContextMemory
from src.reasoning.reasoning_engine import ReasoningEngine


def main():

    print("=" * 70)
    print("HARMONY-X END-TO-END LIVE TEST")
    print("=" * 70)

    vision = VisionProcessor()

    audio = AudioProcessor(
        sample_rate=48000,
        channels=2,
        device=9,
        block_duration=0.1,
        analysis_duration=0.5,
    )

    fusion = MultimodalFusion()

    memory = ContextMemory(
        max_history=50
    )

    reasoning = ReasoningEngine(
        confidence_threshold=0.70
    )

    try:

        # --------------------------------------------------
        # INITIALIZE
        # --------------------------------------------------

        print()
        print("Initializing systems...")

        vision.initialize()

        audio.initialize()

        fusion.initialize()

        memory.initialize()

        reasoning.initialize()

        print()
        print("All HARMONY-X systems initialized.")
        print()
        print("Press Q to stop the test.")
        print()

        start_time = time.time()

        frame_count = 0

        while True:

            # --------------------------------------------------
            # CAMERA FRAME
            # --------------------------------------------------

            success, frame = (
                vision.camera.read()
            )

            if not success:

                print(
                    "Failed to read camera frame."
                )

                break

            # --------------------------------------------------
            # VISION
            # --------------------------------------------------

            vision_result = (
                vision.process(frame)
            )

            # --------------------------------------------------
            # AUDIO
            # --------------------------------------------------

            audio_result = (
                audio.process()
            )

            # --------------------------------------------------
            # FUSION
            # --------------------------------------------------

            fusion_result = (
                fusion.fuse(
                    vision_result,
                    audio_result,
                )
            )

            # --------------------------------------------------
            # MEMORY
            # --------------------------------------------------

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

                action="pending",
            )

            # --------------------------------------------------
            # REASONING + ML
            # --------------------------------------------------

            reasoning_result = (
                reasoning.reason(
                    fusion_result,
                    memory,
                )
            )

            frame_count += 1

            elapsed = (
                time.time() - start_time
            )

            fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0.0
            )

            # --------------------------------------------------
            # DISPLAY
            # --------------------------------------------------

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
                f"Face: {vision_result.face_detected}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Hands: {vision_result.hands_detected}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Pose: {vision_result.pose_detected}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Expression: "
                f"{vision_result.emotion}",
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"ML: "
                f"{reasoning_result.ml_prediction}",
                (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"ML Confidence: "
                f"{reasoning_result.ml_confidence:.2f}",
                (20, 245),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Speech: "
                f"{audio_result.speech_detected}",
                (20, 280),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Action: "
                f"{reasoning_result.action}",
                (20, 315),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Reasoning: "
                f"{reasoning_result.confidence_level}",
                (20, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 385),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                "HARMONY-X End-to-End",
                frame,
            )

            # --------------------------------------------------
            # CONSOLE OUTPUT
            # --------------------------------------------------

            print(
                f"\r"
                f"Expression={vision_result.emotion:<10} | "
                f"ML={reasoning_result.ml_prediction:<8} | "
                f"ML Conf="
                f"{reasoning_result.ml_confidence:.2f} | "
                f"Speech="
                f"{str(audio_result.speech_detected):<5} | "
                f"Action="
                f"{reasoning_result.action:<18}",
                end="",
                flush=True,
            )

            # --------------------------------------------------
            # QUIT
            # --------------------------------------------------

            if (
                cv2.waitKey(1)
                & 0xFF
            ) == ord("q"):

                break

    finally:

        print()
        print()
        print(
            "Stopping HARMONY-X..."
        )

        audio.release()

        vision.release()

        print(
            "HARMONY-X end-to-end test complete."
        )


if __name__ == "__main__":

    main()