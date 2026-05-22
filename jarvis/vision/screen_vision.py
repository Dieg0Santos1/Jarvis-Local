from __future__ import annotations

import base64
from io import BytesIO

import pyautogui
from openai import OpenAI

from jarvis.knowledge.self_model import render_project_self_knowledge


class ScreenVisionService:
    """Captures the current screen and asks a vision model to interpret it."""

    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        from config import settings
        base_url = settings.openai_base_url if settings.openai_base_url else None
        self.client = OpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url
        )
        self.model_name = model_name or settings.openai_vision_model

    def analyze_screen(self, user_request: str = "") -> str:
        image_data = self._capture_screen_as_data_url()
        prompt = self._build_prompt(user_request)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres Jarvis, un asistente de escritorio con vision de pantalla. "
                        "Analiza capturas de pantalla del usuario y responde en espanol. "
                        "Se preciso, util y breve. Si ves codigo o errores, explica el problema "
                        "y sugiere el siguiente paso. Si ves una web o app, describe que esta abierto "
                        "y que accion podria ser util. No inventes texto pequeno que no puedas leer.\n\n"
                        f"{render_project_self_knowledge()}"
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_data,
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            max_tokens=420,
        )

        answer = response.choices[0].message.content
        return (answer or "No pude interpretar claramente la pantalla.").strip()

    def _capture_screen_as_data_url(self) -> str:
        screenshot = pyautogui.screenshot()
        screenshot.thumbnail((1600, 1000))

        buffer = BytesIO()
        screenshot.save(buffer, format="JPEG", quality=82, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_prompt(self, user_request: str) -> str:
        request = user_request.strip()
        if not request:
            request = "Describe lo que estoy viendo en mi pantalla y dime si puedes ayudarme con algo."

        return (
            f"Solicitud del usuario: {request}\n\n"
            "Mira la captura de pantalla actual y responde como Jarvis. "
            "No digas solo que puedes verla; describe lo importante que aparece. "
            "Si hay una tarea evidente, ofrece una recomendacion concreta."
        )
