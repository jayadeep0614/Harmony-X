import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from streamlit_webrtc import (
    VideoProcessorBase,
    AudioProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT MODULES
# ============================================================

from src.perception.vision import VisionProcessor
from src.audio.audio_processor import AudioResult
from src.fusion.multimodal_fusion import MultimodalFusion


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HARMONY-X",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION STATE
# ============================================================

if "camera_active" not in st.session_state:
    st.session_state.camera_active = False


def start_system():
    st.session_state.camera_active = True


def stop_system():
    st.session_state.camera_active = False


# ============================================================
# SHARED RUNTIME
# ============================================================

class HarmonyRuntime:

    def __init__(self):

        self.lock = threading.RLock()

        self.audio = AudioResult()

        self.audio_online = False

        self.audio_callbacks = 0

        self.audio_rms = 0.0

        self.audio_samples = 0

        self.audio_error = ""

        self.last_audio_update = 0.0

    # --------------------------------------------------------
    # UPDATE AUDIO
    # --------------------------------------------------------

    def update_audio(
        self,
        result,
        rms,
        samples,
    ):

        with self.lock:

            self.audio = result

            self.audio_rms = float(rms)

            self.audio_samples += int(
                samples
            )

            self.audio_callbacks += 1

            self.audio_online = True

            self.audio_error = ""

            self.last_audio_update = (
                time.time()
            )

    # --------------------------------------------------------
    # ERROR
    # --------------------------------------------------------

    def set_audio_error(
        self,
        error,
    ):

        with self.lock:

            self.audio_error = str(error)

    # --------------------------------------------------------
    # GET AUDIO
    # --------------------------------------------------------

    def get_audio(self):

        with self.lock:

            return {

                "result":
                    self.audio,

                "online":
                    self.audio_online,

                "callbacks":
                    self.audio_callbacks,

                "rms":
                    self.audio_rms,

                "samples":
                    self.audio_samples,

                "error":
                    self.audio_error,

                "last_update":
                    self.last_audio_update,
            }


# ============================================================
# SINGLE RUNTIME INSTANCE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def get_runtime():

    return HarmonyRuntime()


runtime = get_runtime()


# ============================================================
# WEBRTC AUDIO PROCESSOR
# ============================================================

class HarmonyAudioProcessor(
    AudioProcessorBase
):

    def __init__(self):

        print(
            "======================================"
        )

        print(
            "HARMONY-X WEBRTC AUDIO: INITIALIZING"
        )

        print(
            "MICROPHONE MODE: INPUT ONLY"
        )

        print(
            "SPEAKER OUTPUT: DISABLED"
        )

        print(
            "======================================"
        )

    # ========================================================
    # RECEIVE AUDIO
    # ========================================================

    def recv(self, frame):

        try:

            # ------------------------------------------------
            # GET AUDIO ARRAY
            # ------------------------------------------------

            audio = frame.to_ndarray()

            # ------------------------------------------------
            # CONVERT TO MONO
            # ------------------------------------------------

            if audio.ndim == 2:

                if audio.shape[0] <= 8:

                    mono = np.mean(
                        audio.astype(
                            np.float32
                        ),
                        axis=0,
                    )

                else:

                    mono = np.mean(
                        audio.astype(
                            np.float32
                        ),
                        axis=1,
                    )

            else:

                mono = audio.astype(
                    np.float32
                )

            # ------------------------------------------------
            # NORMALIZE INTEGER AUDIO
            # ------------------------------------------------

            if np.issubdtype(
                audio.dtype,
                np.integer,
            ):

                info = np.iinfo(
                    audio.dtype
                )

                scale = max(
                    abs(info.min),
                    info.max,
                )

                if scale > 0:

                    mono = (
                        mono / float(scale)
                    )

            # ------------------------------------------------
            # RMS
            # ------------------------------------------------

            if len(mono) > 0:

                rms = float(
                    np.sqrt(
                        np.mean(
                            mono ** 2
                        )
                    )
                )

            else:

                rms = 0.0

            # ------------------------------------------------
            # SPEECH DETECTION
            # ------------------------------------------------

            speech_threshold = 0.005

            speech_detected = (
                rms >= speech_threshold
            )

            # ------------------------------------------------
            # VOLUME
            # ------------------------------------------------

            volume = min(
                rms * 8.0,
                1.0,
            )

            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = volume

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            result = AudioResult(

                speech_detected=(
                    speech_detected
                ),

                volume=volume,

                confidence=confidence,
            )

            # ------------------------------------------------
            # SAVE RESULT
            # ------------------------------------------------

            runtime.update_audio(

                result=result,

                rms=rms,

                samples=len(mono),
            )

        except Exception as error:

            print(
                "HARMONY-X WEBRTC AUDIO ERROR:",
                repr(error),
            )

            runtime.set_audio_error(
                repr(error)
            )

        # ====================================================
        # CRITICAL:
        #
        # DO NOT SEND MICROPHONE AUDIO
        # BACK TO THE SPEAKERS.
        #
        # We overwrite every audio plane
        # with silence.
        # ====================================================

        try:

            for plane in frame.planes:

                plane.update(
                    bytes(
                        plane.buffer_size
                    )
                )

        except Exception as error:

            print(
                "AUDIO SILENCE ERROR:",
                repr(error),
            )

        # ----------------------------------------------------
        # Return SILENT frame
        # ----------------------------------------------------

        return frame


# ============================================================
# WEBRTC VIDEO PROCESSOR
# ============================================================

class HarmonyVideoProcessor(
    VideoProcessorBase
):

    def __init__(self):

        self.lock = threading.RLock()

        self.frame_count = 0

        self.ai_frame_count = 0

        self.process_every = 3

        self.running = True

        # ----------------------------------------------------
        # VISION
        # ----------------------------------------------------

        self.vision = None

        try:

            self.vision = VisionProcessor()

            self.vision.initialize(
                use_camera=False
            )

            print(
                "HARMONY-X VISION: ONLINE"
            )

        except Exception as error:

            print(
                "HARMONY-X VISION ERROR:",
                repr(error),
            )

        # ----------------------------------------------------
        # FUSION
        # ----------------------------------------------------

        self.fusion = None

        try:

            self.fusion = MultimodalFusion()

            self.fusion.initialize()

            print(
                "HARMONY-X FUSION: ONLINE"
            )

        except Exception as error:

            print(
                "HARMONY-X FUSION ERROR:",
                repr(error),
            )

        # ----------------------------------------------------
        # LAST RESULTS
        # ----------------------------------------------------

        self.last_vision = None

        self.last_fusion = None

    # ========================================================
    # VIDEO FRAME
    # ========================================================

    def recv(self, frame):

        image = frame.to_ndarray(
            format="bgr24"
        )

        self.frame_count += 1

        # ====================================================
        # AUDIO STATE
        # ====================================================

        audio_state = (
            runtime.get_audio()
        )

        audio_result = (
            audio_state["result"]
        )

        # ====================================================
        # VISION
        # ====================================================

        if (
            self.frame_count
            % self.process_every
            == 0
        ):

            if self.vision is not None:

                try:

                    self.last_vision = (
                        self.vision.process(
                            image
                        )
                    )

                    self.ai_frame_count += 1

                except Exception as error:

                    print(
                        "HARMONY-X VISION PROCESSING ERROR:",
                        repr(error),
                    )

        # ====================================================
        # MULTIMODAL FUSION
        # ====================================================

        if (
            self.last_vision is not None
            and self.fusion is not None
        ):

            try:

                self.last_fusion = (
                    self.fusion.fuse(
                        self.last_vision,
                        audio_result,
                    )
                )

            except Exception as error:

                print(
                    "HARMONY-X FUSION PROCESSING ERROR:",
                    repr(error),
                )

        # ====================================================
        # CAMERA HUD
        # ====================================================

        self.draw_camera(
            image
        )

        # ====================================================
        # RETURN VIDEO
        # ====================================================

        return frame.from_ndarray(
            image,
            format="bgr24",
        )

    # ========================================================
    # CAMERA HUD
    # ========================================================

    def draw_camera(
        self,
        image,
    ):

        cv2.rectangle(
            image,
            (18, 18),
            (150, 55),
            (3, 12, 20),
            -1,
        )

        cv2.rectangle(
            image,
            (18, 18),
            (150, 55),
            (70, 200, 240),
            1,
        )

        cv2.putText(
            image,
            "● LIVE",
            (32, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (100, 230, 255),
            1,
            cv2.LINE_AA,
        )

    # ========================================================
    # CLEANUP
    # ========================================================

    def on_ended(self):

        self.running = False

        if self.vision is not None:

            try:

                self.vision.release()

            except Exception:

                pass

        print(
            "HARMONY-X VIDEO PROCESSOR STOPPED"
        )


# ============================================================
# UI HELPERS
# ============================================================

def clamp(
    value,
):

    try:

        value = float(
            value
        )

    except Exception:

        value = 0.0

    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


def metric(
    name,
    value,
):

    return f"""
    <div class="metric">

        <span>
            {name}
        </span>

        <strong>
            {value}
        </strong>

    </div>
    """


def confidence(
    name,
    value,
):

    value = clamp(
        value
    )

    return f"""
    <div class="confidence">

        <div class="confidence-header">

            <span>
                {name}
            </span>

            <span>
                {value:.2f}
            </span>

        </div>

        <div class="track">

            <div
                class="fill"
                style="
                    width:
                    {value * 100:.1f}%;
                "
            ></div>

        </div>

    </div>
    """


def sensor(
    name,
    icon,
    status,
):

    return f"""
    <div class="sensor">

        <div class="sensor-icon">
            {icon}
        </div>

        <div class="sensor-name">
            {name}
        </div>

        <div class="sensor-status">
            {status}
        </div>

    </div>
    """


# ============================================================
# CSS
# ============================================================

st.html(
"""
<style>

html,
body,
[data-testid="stAppViewContainer"] {

    background:
        radial-gradient(
            circle at 50% 20%,
            rgba(0,100,180,.12),
            transparent 40%
        ),
        linear-gradient(
            180deg,
            #02060b,
            #030b13,
            #02060b
        );
}

[data-testid="stHeader"] {

    background: transparent;
}

.block-container {

    max-width: 1650px;

    padding-top: 1rem;
}


/* HEADER */

.title {

    text-align: center;

    color: #c9f5ff;

    font-size: 2.2rem;

    font-weight: 700;

    letter-spacing: .38em;

    text-shadow:
        0 0 10px rgba(70,210,255,.8),
        0 0 30px rgba(0,120,255,.3);
}

.subtitle {

    text-align: center;

    color: #527f94;

    font-size: .65rem;

    letter-spacing: .25em;

    margin-bottom: 1rem;
}


/* PANEL */

.panel {

    min-height: 710px;

    padding: 1rem;

    border:
        1px solid rgba(
            70,
            180,
            230,
            .28
        );

    border-radius: 14px;

    background:
        linear-gradient(
            180deg,
            rgba(5,18,30,.97),
            rgba(2,9,16,.97)
        );

    box-shadow:
        inset 0 0 35px
        rgba(0,120,190,.05),

        0 0 20px
        rgba(0,130,220,.06);
}

.panel-title {

    color: #8edfff;

    font-size: .72rem;

    font-weight: 700;

    letter-spacing: .18em;

    border-bottom:
        1px solid
        rgba(80,190,230,.18);

    padding-bottom: .65rem;
}

.section {

    color: #4e91ad;

    font-size: .56rem;

    letter-spacing: .17em;

    margin-top: .9rem;

    margin-bottom: .45rem;
}


/* SENSOR */

.sensor-grid {

    display: grid;

    grid-template-columns:
        repeat(2,1fr);

    gap: .45rem;
}

.sensor {

    height: 68px;

    display: flex;

    flex-direction: column;

    justify-content: center;

    align-items: center;

    border:
        1px solid
        rgba(70,180,230,.16);

    border-radius: 9px;

    background:
        rgba(3,14,23,.8);
}

.sensor-icon {

    color: #76ddff;

    font-size: .95rem;
}

.sensor-name {

    color: #61879a;

    font-size: .46rem;

    letter-spacing: .12em;
}

.sensor-status {

    color: #a9eaff;

    font-size: .50rem;

    margin-top: .12rem;
}


/* METRICS */

.metric {

    display: flex;

    justify-content: space-between;

    align-items: center;

    min-height: 27px;

    border-bottom:
        1px solid
        rgba(70,160,200,.07);
}

.metric span {

    color: #628698;

    font-size: .56rem;

    letter-spacing: .07em;
}

.metric strong {

    color: #b8edff;

    font-size: .57rem;
}


/* CONFIDENCE */

.confidence {

    margin-top: .5rem;

    margin-bottom: .55rem;
}

.confidence-header {

    display: flex;

    justify-content: space-between;

    color: #668b9d;

    font-size: .51rem;

    margin-bottom: .25rem;
}

.track {

    height: 5px;

    background: #10232e;

    border-radius: 5px;

    overflow: hidden;
}

.fill {

    height: 100%;

    background:
        linear-gradient(
            90deg,
            #155873,
            #55dfff
        );

    box-shadow:
        0 0 8px
        rgba(70,220,255,.55);
}


/* CAMERA */

.camera {

    padding: .8rem;

    border:
        1px solid
        rgba(70,180,230,.27);

    border-radius: 14px;

    background:
        rgba(3,10,18,.75);
}

.camera-title {

    color: #9edfff;

    font-size: .72rem;

    font-weight: 700;

    letter-spacing: .2em;
}

.camera-subtitle {

    color: #53798d;

    font-size: .55rem;

    letter-spacing: .1em;

    margin-top: .2rem;

    margin-bottom: .6rem;
}


/* CORE */

.core {

    margin-top: .7rem;

    padding: 1rem;

    text-align: center;

    border:
        1px solid
        rgba(70,180,230,.18);

    border-radius: 10px;

    background:
        radial-gradient(
            circle,
            rgba(0,150,220,.13),
            transparent 70%
        );
}

.core-label {

    color: #527c92;

    font-size: .49rem;

    letter-spacing: .16em;
}

.core-value {

    color: #c1f1ff;

    font-size: 1.4rem;

    font-weight: 700;

    letter-spacing: .1em;

    margin-top: .2rem;

    text-shadow:
        0 0 12px
        rgba(80,220,255,.55);
}


/* ACTIVITY */

.activity {

    padding: .6rem;

    border:
        1px solid
        rgba(70,170,210,.14);

    border-radius: 8px;

    background:
        rgba(2,10,17,.72);
}

.log {

    color: #628899;

    font-family: monospace;

    font-size: .50rem;

    line-height: 1.8;
}

.log.active {

    color: #9feaff;
}


/* FLOW */

.flow {

    margin-top: .6rem;

    padding: .65rem;

    text-align: center;

    border:
        1px solid
        rgba(70,180,230,.18);

    border-radius: 9px;

    background:
        rgba(3,15,24,.9);
}

.flow-main {

    color: #7fcbe3;

    font-size: .54rem;

    letter-spacing: .1em;
}

.flow-sub {

    color: #436a7d;

    font-size: .45rem;

    margin-top: .2rem;
}


/* BUTTONS */

div.stButton > button {

    min-height: 42px;

    border-radius: 999px;

    border:
        1px solid
        rgba(90,200,255,.55);

    background:
        linear-gradient(
            180deg,
            rgba(10,35,52,.96),
            rgba(4,14,24,.96)
        );

    color: #a9e8ff;

    font-size: .68rem;

    font-weight: 700;

    letter-spacing: .1em;
}

div.stButton > button:hover {

    border-color:
        rgba(100,225,255,.95);

    color: white;

    box-shadow:
        0 0 20px
        rgba(0,180,255,.25);
}


/* FOOTER */

.footer {

    text-align: center;

    color: #3c6274;

    font-size: .50rem;

    letter-spacing: .15em;

    margin-top: .7rem;
}

</style>
"""
)


# ============================================================
# HEADER
# ============================================================

st.html(
"""
<div class="title">
    HARMONY-X
</div>

<div class="subtitle">
    ADAPTIVE MULTIMODAL INTELLIGENCE SYSTEM
</div>
"""
)


# ============================================================
# LAYOUT
# ============================================================

left, center, right = st.columns(
    [1.1, 2.05, 1.1],
    gap="medium",
)


# ============================================================
# CENTER
# ============================================================

with center:

    st.html(
    """
    <div class="camera">

        <div class="camera-title">
            ◉ LIVE MULTIMODAL SENSOR
        </div>

        <div class="camera-subtitle">
            CAMERA + MICROPHONE // INPUT ONLY
        </div>

    </div>
    """
    )

    c1, c2 = st.columns(2)

    with c1:

        st.button(
            "START SYSTEM",
            on_click=start_system,
            use_container_width=True,
        )

    with c2:

        st.button(
            "STOP SYSTEM",
            on_click=stop_system,
            use_container_width=True,
        )

    # ========================================================
    # WEBRTC
    # ========================================================

    ctx = webrtc_streamer(

        key="harmony_x_multimodal",

        mode=WebRtcMode.SENDRECV,

        video_processor_factory=(
            HarmonyVideoProcessor
        ),

        audio_processor_factory=(
            HarmonyAudioProcessor
        ),

        media_stream_constraints={

            "video": {

                "width": {
                    "ideal": 960
                },

                "height": {
                    "ideal": 720
                },

                "frameRate": {
                    "ideal": 24
                },

            },

            "audio": True,

        },

        async_processing=True,

        media_toggle_controls=False,

        desired_playing_state=(
            st.session_state.camera_active
        ),
    )


# ============================================================
# AUDIO STATE
# ============================================================

audio_state = (
    runtime.get_audio()
)

audio_online = bool(
    audio_state["online"]
)

audio_result = (
    audio_state["result"]
)

audio_callbacks = int(
    audio_state["callbacks"]
)

audio_rms = float(
    audio_state["rms"]
)


# ============================================================
# VIDEO PROCESSOR
# ============================================================

processor = None

try:

    processor = (
        ctx.video_processor
    )

except Exception:

    processor = None


# ============================================================
# VISION
# ============================================================

vision = None

fusion_result = None

if processor is not None:

    vision = (
        processor.last_vision
    )

    fusion_result = (
        processor.last_fusion
    )


if vision is not None:

    face_detected = bool(
        vision.face_detected
    )

    hands_detected = bool(
        vision.hands_detected
    )

    pose_detected = bool(
        vision.pose_detected
    )

    emotion = str(
        vision.emotion
    )

    vision_conf = clamp(
        vision.confidence
    )

    emotion_conf = clamp(
        vision.emotion_confidence
    )

else:

    face_detected = False

    hands_detected = False

    pose_detected = False

    emotion = "UNKNOWN"

    vision_conf = 0.0

    emotion_conf = 0.0


# ============================================================
# AUDIO
# ============================================================

speech_detected = bool(
    audio_result.speech_detected
)

volume = clamp(
    audio_result.volume
)

audio_conf = clamp(
    audio_result.confidence
)


# ============================================================
# FUSION
# ============================================================

if fusion_result is not None:

    visual_fusion = clamp(
        fusion_result.visual_confidence
    )

    audio_fusion = clamp(
        fusion_result.audio_confidence
    )

    fusion_conf = clamp(
        fusion_result.overall_confidence
    )

else:

    visual_fusion = 0.0

    audio_fusion = 0.0

    fusion_conf = 0.0


# ============================================================
# LEFT PANEL
# ============================================================

with left:

    left_html = f"""

    <div class="panel">

        <div class="panel-title">
            PERCEPTION MATRIX
        </div>


        <div class="section">
            SENSOR ARRAY
        </div>


        <div class="sensor-grid">

            {sensor(
                "FACE",
                "◉",
                "DETECTED"
                if face_detected
                else "SEARCHING"
            )}

            {sensor(
                "HANDS",
                "✋",
                "DETECTED"
                if hands_detected
                else "SEARCHING"
            )}

            {sensor(
                "POSE",
                "◇",
                "DETECTED"
                if pose_detected
                else "SEARCHING"
            )}

            {sensor(
                "AUDIO",
                "◌",
                "ONLINE"
                if audio_online
                else "WAITING"
            )}

        </div>


        <div class="section">
            VISUAL PERCEPTION
        </div>


        {metric(
            "FACE",
            "DETECTED"
            if face_detected
            else "NONE"
        )}


        {metric(
            "HANDS",
            "DETECTED"
            if hands_detected
            else "NONE"
        )}


        {metric(
            "POSE",
            "DETECTED"
            if pose_detected
            else "NONE"
        )}


        {confidence(
            "VISION CONFIDENCE",
            vision_conf
        )}


        <div class="section">
            AUDIO PERCEPTION
        </div>


        {metric(
            "MICROPHONE",
            "ONLINE"
            if audio_online
            else "WAITING"
        )}


        {metric(
            "SPEECH",
            "DETECTED"
            if speech_detected
            else "QUIET"
        )}


        {metric(
            "VOLUME",
            f"{volume * 100:.1f}%"
        )}


        {confidence(
            "AUDIO CONFIDENCE",
            audio_conf
        )}


        <div class="section">
            AUDIO TELEMETRY
        </div>


        {metric(
            "WEBRTC CALLBACKS",
            str(audio_callbacks)
        )}


        {metric(
            "RMS",
            f"{audio_rms:.5f}"
        )}


        {metric(
            "STREAM",
            "RECEIVING"
            if audio_callbacks > 0
            else "WAITING"
        )}


        <div class="section">
            SENSOR STATUS
        </div>


        {metric(
            "VISION",
            "ONLINE"
            if processor is not None
            else "WAITING"
        )}


        {metric(
            "MICROPHONE",
            "ONLINE"
            if audio_online
            else "WAITING"
        )}


        {metric(
            "FUSION",
            "ONLINE"
            if fusion_result is not None
            else "WAITING"
        )}

    </div>

    """

    st.html(
        left_html
    )


# ============================================================
# RIGHT PANEL
# ============================================================

with right:

    right_html = f"""

    <div class="panel">

        <div class="panel-title">
            COGNITIVE STATE
        </div>


        <div class="core">

            <div class="core-label">
                CURRENT EXPRESSION
            </div>

            <div class="core-value">
                {emotion}
            </div>

        </div>


        {confidence(
            "EXPRESSION CONFIDENCE",
            emotion_conf
        )}


        <div class="section">
            MULTIMODAL FUSION
        </div>


        {metric(
            "VISUAL INPUT",
            f"{visual_fusion:.2f}"
        )}


        {metric(
            "AUDIO INPUT",
            f"{audio_fusion:.2f}"
        )}


        {metric(
            "SPEECH",
            "ACTIVE"
            if speech_detected
            else "QUIET"
        )}


        {confidence(
            "FUSION CONFIDENCE",
            fusion_conf
        )}


        <div class="section">
            COGNITIVE PROCESSING
        </div>


        {metric(
            "REASONING ENGINE",
            "PROCESSING"
            if processor is not None
            else "STANDBY"
        )}


        {metric(
            "CONTEXT MEMORY",
            "ACTIVE"
            if processor is not None
            else "STANDBY"
        )}


        {metric(
            "ADAPTATION",
            "READY"
        )}


        {metric(
            "DECISION LAYER",
            "ONLINE"
            if fusion_result is not None
            else "WAITING"
        )}


        <div class="section">
            LIVE ACTIVITY
        </div>


        <div class="activity">

            <div class="log">
                [01] Visual perception active
            </div>

            <div class="log">

                [02]
                {
                    "Speech detected"
                    if speech_detected
                    else "Listening for speech"
                }

            </div>

            <div class="log">

                [03]
                {
                    "Multimodal fusion active"
                    if fusion_result is not None
                    else "Fusion waiting"
                }

            </div>

            <div class="log">
                [04] Expression: {emotion}
            </div>

            <div class="log active">
                [05] HARMONY-X ONLINE
            </div>

        </div>


        <div class="section">
            SYSTEM TELEMETRY
        </div>


        {metric(
            "VIDEO FRAMES",
            str(
                processor.frame_count
                if processor is not None
                else 0
            )
        )}


        {metric(
            "AI FRAMES",
            str(
                processor.ai_frame_count
                if processor is not None
                else 0
            )
        )}


        {metric(
            "AUDIO CALLBACKS",
            str(audio_callbacks)
        )}


        {metric(
            "AUDIO",
            "ONLINE"
            if audio_online
            else "WAITING"
        )}


        {metric(
            "FUSION",
            "ONLINE"
            if fusion_result is not None
            else "WAITING"
        )}

    </div>

    """

    st.html(
        right_html
    )


# ============================================================
# SYSTEM FLOW
# ============================================================

with center:

    st.html(
    """
    <div class="flow">

        <div class="flow-main">

            CAMERA
            →
            MICROPHONE
            →
            PERCEPTION
            →
            FUSION
            →
            COGNITION

        </div>

        <div class="flow-sub">

            MICROPHONE INPUT ONLY
            // NO AUDIO PLAYBACK

        </div>

    </div>
    """
    )


# ============================================================
# FOOTER
# ============================================================

st.html(
"""
<div class="footer">

    HARMONY-X
    //
    ADAPTIVE MULTIMODAL INTELLIGENCE
    //
    RESEARCH PROTOTYPE

</div>
"""
)


# ============================================================
# LIVE REFRESH
# ============================================================

if st.session_state.camera_active:

    time.sleep(
        0.25
    )

    st.rerun()