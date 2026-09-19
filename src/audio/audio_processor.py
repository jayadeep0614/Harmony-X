import threading
import numpy as np
import sounddevice as sd

from dataclasses import dataclass


@dataclass
class AudioResult:
    speech_detected: bool = False
    volume: float = 0.0
    confidence: float = 0.0


class AudioProcessor:

    def __init__(
        self,
        sample_rate=48000,
        channels=2,
        device=9,
        block_duration=0.1,
        analysis_duration=0.5,
    ):

        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device

        self.block_duration = block_duration
        self.analysis_duration = analysis_duration

        self.stream = None

        self.initialized = False

        self.lock = threading.Lock()

        self.latest_result = AudioResult()

        self.audio_buffer = np.zeros(
            int(
                self.sample_rate
                * self.analysis_duration
            ),
            dtype=np.float32,
        )

        self.callback_count = 0

    # ========================================================
    # AUDIO CALLBACK
    # ========================================================

    def _callback(
        self,
        indata,
        frames,
        time_info,
        status,
    ):

        try:

            if status:
                print(
                    "Audio status:",
                    status,
                )

            # ------------------------------------------------
            # Convert stereo → mono
            # ------------------------------------------------

            if indata.ndim > 1:

                mono = np.mean(
                    indata,
                    axis=1,
                )

            else:

                mono = indata

            mono = mono.astype(
                np.float32,
                copy=False,
            )

            # ------------------------------------------------
            # Calculate RMS
            # ------------------------------------------------

            rms = float(
                np.sqrt(
                    np.mean(
                        mono ** 2
                    )
                )
            )

            # ------------------------------------------------
            # Update rolling buffer
            # ------------------------------------------------

            with self.lock:

                if len(mono) >= len(
                    self.audio_buffer
                ):

                    self.audio_buffer[:] = (
                        mono[
                            -len(
                                self.audio_buffer
                            ):
                        ]
                    )

                else:

                    self.audio_buffer = np.roll(
                        self.audio_buffer,
                        -len(mono),
                    )

                    self.audio_buffer[
                        -len(mono):
                    ] = mono

            # ------------------------------------------------
            # Speech detection
            # ------------------------------------------------

            speech_threshold = 0.005

            speech_detected = (
                rms >= speech_threshold
            )

            # ------------------------------------------------
            # Normalize volume
            # ------------------------------------------------

            volume = min(
                rms * 8.0,
                1.0,
            )

            # ------------------------------------------------
            # Confidence
            # ------------------------------------------------

            confidence = volume

            result = AudioResult(

                speech_detected=(
                    speech_detected
                ),

                volume=volume,

                confidence=confidence,
            )

            # ------------------------------------------------
            # Store latest result
            # ------------------------------------------------

            with self.lock:

                self.latest_result = result

                self.callback_count += 1

        except Exception as error:

            print(
                "Audio callback error:",
                error,
            )

    # ========================================================
    # INITIALIZE
    # ========================================================

    def initialize(self):

        if self.initialized:

            return

        print(
            "Initializing HARMONY-X microphone..."
        )

        # ----------------------------------------------------
        # Verify device
        # ----------------------------------------------------

        try:

            device_info = sd.query_devices(
                self.device
            )

            print(
                "Audio device:",
                device_info["name"],
            )

            print(
                "Input channels:",
                device_info["max_input_channels"],
            )

        except Exception as error:

            print(
                "Unable to query audio device:",
                error,
            )

            raise

        # ----------------------------------------------------
        # Create input stream
        # ----------------------------------------------------

        self.stream = sd.InputStream(

            samplerate=self.sample_rate,

            channels=self.channels,

            device=self.device,

            blocksize=int(
                self.sample_rate
                * self.block_duration
            ),

            dtype="float32",

            callback=self._callback,

        )

        # ----------------------------------------------------
        # Start
        # ----------------------------------------------------

        self.stream.start()

        self.initialized = True

        print(
            "HARMONY-X Audio: ONLINE"
        )

    # ========================================================
    # PROCESS
    # ========================================================

    def process(
        self,
        audio_data=None,
    ):

        # ----------------------------------------------------
        # Direct audio data mode
        # ----------------------------------------------------

        if audio_data is not None:

            try:

                if audio_data.ndim > 1:

                    mono = np.mean(
                        audio_data,
                        axis=1,
                    )

                else:

                    mono = audio_data

                mono = mono.astype(
                    np.float32,
                    copy=False,
                )

                rms = float(
                    np.sqrt(
                        np.mean(
                            mono ** 2
                        )
                    )
                )

                speech_threshold = 0.005

                speech_detected = (
                    rms >= speech_threshold
                )

                volume = min(
                    rms * 8.0,
                    1.0,
                )

                return AudioResult(

                    speech_detected=(
                        speech_detected
                    ),

                    volume=volume,

                    confidence=volume,
                )

            except Exception as error:

                print(
                    "Direct audio processing error:",
                    error,
                )

                return AudioResult()

        # ----------------------------------------------------
        # Stream mode
        # ----------------------------------------------------

        with self.lock:

            return AudioResult(

                speech_detected=(
                    self.latest_result
                    .speech_detected
                ),

                volume=(
                    self.latest_result
                    .volume
                ),

                confidence=(
                    self.latest_result
                    .confidence
                ),
            )

    # ========================================================
    # STATUS
    # ========================================================

    def status(self):

        with self.lock:

            return {

                "initialized":
                    self.initialized,

                "device":
                    self.device,

                "sample_rate":
                    self.sample_rate,

                "channels":
                    self.channels,

                "callback_count":
                    self.callback_count,

                "speech_detected":
                    self.latest_result
                    .speech_detected,

                "volume":
                    self.latest_result
                    .volume,

                "confidence":
                    self.latest_result
                    .confidence,
            }

    # ========================================================
    # RELEASE
    # ========================================================

    def release(self):

        if self.stream is not None:

            try:

                self.stream.stop()

            except Exception:
                pass

            try:

                self.stream.close()

            except Exception:
                pass

            self.stream = None

        self.initialized = False

        print(
            "HARMONY-X Audio: OFFLINE"
        )