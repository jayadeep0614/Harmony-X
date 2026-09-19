"""
HARMONY-X
Real-Time Vision / Perception Module
Face + Hands + Pose + Emotion
"""

from dataclasses import dataclass, field
from typing import Optional

import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.perception.emotion_analyzer import EmotionAnalyzer


@dataclass
class VisionResult:
    """Container for visual perception results."""

    face_detected: bool = False
    pose_detected: bool = False
    hands_detected: bool = False

    confidence: float = 0.0

    expressions: dict = field(default_factory=dict)

    emotion: str = "UNKNOWN"
    emotion_confidence: float = 0.0


class VisionProcessor:
    """Real-time face, hand, pose and emotion perception."""

    def __init__(self):

        self.name = "HARMONY-X Vision Processor"

        self.initialized = False

        self.camera = None

        self.face_landmarker = None
        self.hand_landmarker = None
        self.pose_landmarker = None

        self.emotion_analyzer = EmotionAnalyzer()

        self.timestamp_ms = 0

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def initialize(self, use_camera=True):

        print(
            f"{self.name}: initializing..."
        )

        # -----------------------------------------------------
        # CAMERA
        # -----------------------------------------------------

        if use_camera:

            self.camera = cv2.VideoCapture(0)

            if not self.camera.isOpened():

                raise RuntimeError(
                    "Could not open the camera."
                )

        else:

            self.camera = None

        # -----------------------------------------------------
        # FACE LANDMARKER
        # -----------------------------------------------------

        face_base_options = python.BaseOptions(
            model_asset_path=
            "models/mediapipe/face_landmarker.task"
        )

        face_options = vision.FaceLandmarkerOptions(

            base_options=face_base_options,

            running_mode=vision.RunningMode.VIDEO,

            num_faces=1,

            output_face_blendshapes=True,

            min_face_detection_confidence=0.5,

            min_face_presence_confidence=0.5,

            min_tracking_confidence=0.5,
        )

        self.face_landmarker = (
            vision.FaceLandmarker.create_from_options(
                face_options
            )
        )

        # -----------------------------------------------------
        # HAND LANDMARKER
        # -----------------------------------------------------

        hand_base_options = python.BaseOptions(
            model_asset_path=
            "models/mediapipe/hand_landmarker.task"
        )

        hand_options = vision.HandLandmarkerOptions(

            base_options=hand_base_options,

            running_mode=vision.RunningMode.VIDEO,

            num_hands=2,

            min_hand_detection_confidence=0.5,

            min_hand_presence_confidence=0.5,

            min_tracking_confidence=0.5,
        )

        self.hand_landmarker = (
            vision.HandLandmarker.create_from_options(
                hand_options
            )
        )

        # -----------------------------------------------------
        # POSE LANDMARKER
        # -----------------------------------------------------

        pose_base_options = python.BaseOptions(
            model_asset_path=
            "models/mediapipe/pose_landmarker_full.task"
        )

        pose_options = vision.PoseLandmarkerOptions(

            base_options=pose_base_options,

            running_mode=vision.RunningMode.VIDEO,

            num_poses=1,

            min_pose_detection_confidence=0.5,

            min_pose_presence_confidence=0.5,

            min_tracking_confidence=0.5,
        )

        self.pose_landmarker = (
            vision.PoseLandmarker.create_from_options(
                pose_options
            )
        )

        # -----------------------------------------------------
        # READY
        # -----------------------------------------------------

        self.initialized = True

        print(
            f"{self.name}: initialized"
        )

        if use_camera:

            print(
                "Camera: connected"
            )

        else:

            print(
                "Camera: external frame mode"
            )

        print(
            "Face Landmarker: ready"
        )

        print(
            "Face Blendshapes: enabled"
        )

        print(
            "Hand Landmarker: ready"
        )

        print(
            "Pose Landmarker: ready"
        )

        print(
            "Emotion Analyzer: ready"
        )

    # =========================================================
    # PROCESS FRAME
    # =========================================================

    def process(
        self,
        frame: Optional[object] = None
    ) -> VisionResult:

        if not self.initialized:

            raise RuntimeError(
                "VisionProcessor has not been initialized."
            )

        # -----------------------------------------------------
        # GET FRAME
        # -----------------------------------------------------

        if frame is None:

            if self.camera is None:

                raise RuntimeError(
                    "No camera is available and no frame "
                    "was provided."
                )

            success, frame = (
                self.camera.read()
            )

            if not success:

                raise RuntimeError(
                    "Could not read frame from camera."
                )

        # -----------------------------------------------------
        # BGR -> RGB
        # -----------------------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # -----------------------------------------------------
        # MEDIAPIPE IMAGE
        # -----------------------------------------------------

        mp_image = mp.Image(

            image_format=
            mp.ImageFormat.SRGB,

            data=rgb_frame
        )

        # -----------------------------------------------------
        # TIMESTAMP
        # -----------------------------------------------------

        self.timestamp_ms += 33

        # =====================================================
        # FACE
        # =====================================================

        face_result = (
            self.face_landmarker.detect_for_video(
                mp_image,
                self.timestamp_ms
            )
        )

        face_detected = (
            len(
                face_result.face_landmarks
            ) > 0
        )

        # -----------------------------------------------------
        # FACE BLENDSHAPES
        # -----------------------------------------------------

        expressions = {}

        if (
            face_result.face_blendshapes
            and len(
                face_result.face_blendshapes
            ) > 0
        ):

            blendshapes = (
                face_result.face_blendshapes[0]
            )

            for category in blendshapes:

                expressions[
                    category.category_name
                ] = category.score

        # -----------------------------------------------------
        # EMOTION ANALYSIS
        # -----------------------------------------------------

        emotion_result = (
            self.emotion_analyzer.analyze(
                expressions
            )
        )

        # =====================================================
        # HANDS
        # =====================================================

        hand_result = (
            self.hand_landmarker.detect_for_video(
                mp_image,
                self.timestamp_ms
            )
        )

        hands_detected = (
            len(
                hand_result.hand_landmarks
            ) > 0
        )

        # =====================================================
        # POSE
        # =====================================================

        pose_result = (
            self.pose_landmarker.detect_for_video(
                mp_image,
                self.timestamp_ms
            )
        )

        pose_detected = (
            len(
                pose_result.pose_landmarks
            ) > 0
        )

        # =====================================================
        # VISUAL CONFIDENCE
        # =====================================================

        detections = 0

        if face_detected:

            detections += 1

        if hands_detected:

            detections += 1

        if pose_detected:

            detections += 1

        confidence = (
            detections / 3.0
        )

        # =====================================================
        # RETURN RESULT
        # =====================================================

        return VisionResult(

            face_detected=
            face_detected,

            pose_detected=
            pose_detected,

            hands_detected=
            hands_detected,

            confidence=
            confidence,

            expressions=
            expressions,

            emotion=
            emotion_result.emotion,

            emotion_confidence=
            emotion_result.confidence,
        )

    # =========================================================
    # RELEASE
    # =========================================================

    def release(self):

        if self.camera is not None:

            self.camera.release()

            self.camera = None

        if self.face_landmarker is not None:

            self.face_landmarker.close()

            self.face_landmarker = None

        if self.hand_landmarker is not None:

            self.hand_landmarker.close()

            self.hand_landmarker = None

        if self.pose_landmarker is not None:

            self.pose_landmarker.close()

            self.pose_landmarker = None

        cv2.destroyAllWindows()

        self.initialized = False

        print(
            "Vision resources released."
        )

    # =========================================================
    # STATUS
    # =========================================================

    def status(self):

        return {

            "name":
            self.name,

            "initialized":
            self.initialized,

            "camera_connected":
            (
                self.camera is not None
                and self.camera.isOpened()
            ),

            "face_landmarker":
            self.face_landmarker is not None,

            "hand_landmarker":
            self.hand_landmarker is not None,

            "pose_landmarker":
            self.pose_landmarker is not None,

            "emotion_analyzer":
            self.emotion_analyzer is not None,
        }


# =============================================================
# DIRECT CAMERA TEST
# =============================================================

if __name__ == "__main__":

    vision_processor = (
        VisionProcessor()
    )

    try:

        # Normal desktop camera mode
        vision_processor.initialize(
            use_camera=True
        )

        print()

        print(
            "HARMONY-X"
        )

        print(
            "Face + Hands + Pose + Emotion"
        )

        print(
            "Press Q to quit."
        )

        print()

        while True:

            success, frame = (
                vision_processor.camera.read()
            )

            if not success:

                print(
                    "Failed to read camera frame."
                )

                break

            result = (
                vision_processor.process(
                    frame
                )
            )

            # -------------------------------------------------
            # DISPLAY
            # -------------------------------------------------

            cv2.putText(

                frame,

                "HARMONY-X VISION",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2,
            )

            cv2.putText(

                frame,

                f"Face: "
                f"{result.face_detected}",

                (20, 80),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2,
            )

            cv2.putText(

                frame,

                f"Pose: "
                f"{result.pose_detected}",

                (20, 115),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2,
            )

            cv2.putText(

                frame,

                f"Hands: "
                f"{result.hands_detected}",

                (20, 150),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2,
            )

            cv2.putText(

                frame,

                f"Expression: "
                f"{result.emotion}",

                (20, 195),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 0),

                2,
            )

            cv2.putText(

                frame,

                f"Expression Score: "
                f"{result.emotion_confidence * 100:.1f}%",

                (20, 230),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 0),

                2,
            )

            cv2.putText(

                frame,

                f"Visual Confidence: "
                f"{result.confidence * 100:.1f}%",

                (20, 265),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 0),

                2,
            )

            # -------------------------------------------------
            # SHOW
            # -------------------------------------------------

            cv2.imshow(
                "HARMONY-X Vision",
                frame
            )

            # -------------------------------------------------
            # QUIT
            # -------------------------------------------------

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):

                break

    finally:

        vision_processor.release()