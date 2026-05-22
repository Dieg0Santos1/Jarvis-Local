from __future__ import annotations

import base64
from dataclasses import dataclass
import time

import cv2
from openai import OpenAI

from jarvis.knowledge.self_model import render_project_self_knowledge


@dataclass(slots=True)
class CameraAnalysis:
    answer: str
    preview_data_url: str
    sharpness_score: float


@dataclass(slots=True)
class CameraFrame:
    frame: object
    sharpness_score: float


class CameraVisionService:
    """Captures one webcam frame and asks a vision model to interpret it."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        camera_index: int | None = None,
        width: int | None = None,
        height: int | None = None,
        warmup_frames: int | None = None,
        inspection_frames: int | None = None,
        jpeg_quality: int | None = None,
    ) -> None:
        from config import settings
        base_url = settings.openai_base_url if settings.openai_base_url else None
        self.client = OpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url
        )
        self.model_name = model_name or settings.openai_vision_model
        self.camera_index = settings.camera_index if camera_index is None else camera_index
        self.width = settings.camera_width if width is None else width
        self.height = settings.camera_height if height is None else height
        self.warmup_frames = settings.camera_warmup_frames if warmup_frames is None else warmup_frames
        self.inspection_frames = max(1, settings.camera_inspection_frames if inspection_frames is None else inspection_frames)
        self.jpeg_quality = max(70, min(100, settings.camera_jpeg_quality if jpeg_quality is None else jpeg_quality))

    def analyze_camera(self, user_request: str = "") -> str:
        return self.analyze_camera_with_preview(user_request).answer

    def analyze_camera_with_preview(self, user_request: str = "", inspect: bool = False) -> CameraAnalysis:
        camera_frame = self._capture_best_camera_frame(inspect=inspect)
        image_data = self._frame_as_data_url(camera_frame.frame, max_width=1600, quality=self.jpeg_quality)
        preview_data = self._frame_as_data_url(camera_frame.frame, max_width=560, quality=86)
        prompt = self._build_prompt(user_request, inspect=inspect)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres Jarvis, un asistente con vision por camara. "
                        "Responde en espanol como si estuvieras mirando por la camara en este momento. "
                        "No uses frases como 'en la captura', 'en la imagen' o 'en la foto'. "
                        "Empieza de forma natural, por ejemplo con 'Senor,' cuando encaje. "
                        "Se preciso, util y breve. Describe solo lo que sea visible con claridad. "
                        "Si el objeto es pequeno o la luz/enfoque no ayudan, dilo y pide acercarlo o iluminarlo mejor. "
                        "Si el usuario parece estar haciendo una tarea, ofrece ayuda concreta. "
                        "No identifiques personas por nombre ni hagas afirmaciones sensibles sobre identidad, edad, "
                        "salud, emociones o atributos protegidos. Si no puedes ver algo con claridad, dilo.\n\n"
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
        return CameraAnalysis(
            answer=(answer or "No pude interpretar claramente la camara.").strip(),
            preview_data_url=preview_data,
            sharpness_score=camera_frame.sharpness_score,
        )

    def _capture_best_camera_frame(self, inspect: bool = False) -> CameraFrame:
        capture = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError("No pude abrir la camara.")

        try:
            self._configure_capture(capture)
            best_frame = None
            best_score = -1.0
            total_frames = self.warmup_frames + (self.inspection_frames if inspect else 1)

            for _ in range(max(1, self.warmup_frames)):
                ok, candidate = capture.read()
                if ok:
                    best_frame = candidate
                    best_score = self._sharpness_score(candidate)
                time.sleep(0.03)

            for _ in range(max(1, total_frames - self.warmup_frames)):
                ok, candidate = capture.read()
                if ok:
                    score = self._sharpness_score(candidate)
                    if score > best_score:
                        best_frame = candidate
                        best_score = score
                time.sleep(0.04)

            if best_frame is None:
                raise RuntimeError("No pude capturar imagen de la camara.")

            return CameraFrame(frame=best_frame, sharpness_score=best_score)
        finally:
            capture.release()

    def _configure_capture(self, capture) -> None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    def _sharpness_score(self, frame) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _frame_as_data_url(self, frame, max_width: int, quality: int) -> str:
        resized = self._resize_frame(frame, max_width=max_width)
        ok, encoded = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("No pude preparar la imagen de la camara.")

        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    def _resize_frame(self, frame, max_width: int):
        height, width = frame.shape[:2]
        if width <= max_width:
            return frame

        scale = max_width / width
        new_size = (max_width, int(height * scale))
        return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    def _build_prompt(self, user_request: str, inspect: bool = False) -> str:
        request = user_request.strip()
        if not request:
            request = "Mira lo que estoy haciendo por la camara y dime que ves."

        inspection_instruction = ""
        if inspect:
            inspection_instruction = (
                "El usuario quiere inspeccionar un objeto. Enfocate en identificar el objeto, "
                "color, forma, material aparente, detalles distintivos y nivel de certeza. "
                "Si no estas seguro, da una hipotesis prudente y pide acercarlo o girarlo. "
            )

        return (
            f"Solicitud del usuario: {request}\n\n"
            "Mira por la camara y responde como Jarvis. "
            "No digas solo que puedes verlo; describe lo importante que aparece. "
            "Evita decir 'captura', 'imagen' o 'foto'; habla como si estuvieras observando la escena. "
            f"{inspection_instruction}"
            "Si hay una accion evidente que pueda ayudar al usuario, sugiere el siguiente paso."
        )
