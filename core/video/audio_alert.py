import numpy as np
import wave
from pathlib import Path
from config.settings import AUDIO_ALERT_CONFIG


def generate_alert_wav(alert_flags, fps, wav_path):

    sample_rate = AUDIO_ALERT_CONFIG["sample_rate"]
    beep_hz = AUDIO_ALERT_CONFIG["beep_hz"]
    pulses_per_sec = AUDIO_ALERT_CONFIG["pulses_per_sec"]
    duty_cycle = AUDIO_ALERT_CONFIG["duty_cycle"]
    volume = AUDIO_ALERT_CONFIG["volume"]

    wav_path = Path(wav_path)
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)

        for i, is_alert in enumerate(alert_flags):

            duration = 1.0 / fps
            t = np.linspace(0, duration, int(sample_rate * duration), False)

            if not is_alert:
                samples = np.zeros_like(t)
            else:
                envelope = ((t * pulses_per_sec) % 1.0) < duty_cycle
                tone = np.sin(2 * np.pi * beep_hz * t)
                samples = volume * envelope * tone

            pcm = np.int16(samples * 32767)
            wf.writeframes(pcm.tobytes())

    return wav_path