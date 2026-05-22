from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(slots=True)
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_vision_model: str = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    camera_index: int = int(os.getenv("CAMERA_INDEX", "0"))
    camera_width: int = int(os.getenv("CAMERA_WIDTH", "1920"))
    camera_height: int = int(os.getenv("CAMERA_HEIGHT", "1080"))
    camera_warmup_frames: int = int(os.getenv("CAMERA_WARMUP_FRAMES", "12"))
    camera_inspection_frames: int = int(os.getenv("CAMERA_INSPECTION_FRAMES", "5"))
    camera_jpeg_quality: int = int(os.getenv("CAMERA_JPEG_QUALITY", "94"))
    owner_name: str = os.getenv("JARVIS_OWNER_NAME", "Diego Alexander Santos Aguilar")
    security_enabled: bool = os.getenv("SECURITY_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    security_session_seconds: int = int(os.getenv("SECURITY_SESSION_SECONDS", "300"))
    face_recognition_threshold: float = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "78"))
    face_enrollment_samples: int = int(os.getenv("FACE_ENROLLMENT_SAMPLES", "18"))
    language: str = os.getenv("JARVIS_LANGUAGE", "es")
    whisper_model: str = os.getenv("WHISPER_MODEL", "small")
    whisper_prompt: str = os.getenv(
        "WHISPER_PROMPT",
        (
            "Asistente de voz en espanol. Comandos comunes: abre Chrome, abre Google Chrome, "
            "abre Visual Studio Code, abre VS Code, abre Spotify, abre GitHub, abre la terminal, "
            "abre el explorador de archivos, busca Bad Bunny en Google, que hora es."
        ),
    )
    voice_max_seconds: float = float(os.getenv("VOICE_MAX_SECONDS", "8"))
    voice_silence_seconds: float = float(os.getenv("VOICE_SILENCE_SECONDS", "1.2"))
    voice_min_speech_seconds: float = float(os.getenv("VOICE_MIN_SPEECH_SECONDS", "0.5"))
    voice_energy_threshold: float = float(os.getenv("VOICE_ENERGY_THRESHOLD", "0.015"))
    wake_word: str = os.getenv("WAKE_WORD", "jarvis")
    wake_word_model: str = os.getenv("WAKE_WORD_MODEL", "hey_jarvis")
    wake_word_threshold: float = float(os.getenv("WAKE_WORD_THRESHOLD", "0.12"))
    wake_word_debug: bool = os.getenv("WAKE_WORD_DEBUG", "false").lower() in {"1", "true", "yes", "on"}
    tts_rate: int = int(os.getenv("TTS_RATE", "175"))
    tts_voice_hint: str = os.getenv("TTS_VOICE_HINT", "")
    elevenlabs_enabled: bool = os.getenv("ELEVENLABS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    elevenlabs_api_key: str = os.getenv("ELEVENLABS_API_KEY", "")
    elevenlabs_voice_id: str = os.getenv("ELEVENLABS_VOICE_ID", "")
    local_fallback_enabled: bool = os.getenv("LOCAL_FALLBACK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
    dgx_spark_enabled: bool = os.getenv("DGX_SPARK_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
    dgx_spark_host: str = os.getenv("DGX_SPARK_HOST", "127.0.0.1")
    dgx_spark_port: int = int(os.getenv("DGX_SPARK_PORT", "8080"))


settings = Settings()
