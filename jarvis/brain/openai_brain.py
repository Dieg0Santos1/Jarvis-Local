from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Generator

from openai import OpenAI

from jarvis.knowledge.self_model import render_project_self_knowledge
from jarvis.memory.storage import MemoryManager


class OpenAIBrain:
    def __init__(self, api_key: str | None = None, model_name: str | None = None) -> None:
        from config import settings
        base_url = settings.openai_base_url if settings.openai_base_url else None
        self.client = OpenAI(
            api_key=api_key or settings.openai_api_key,
            base_url=base_url
        )
        self.model_name = model_name or settings.openai_model
        self.memory = MemoryManager()
        self.history: list[dict] = []
        self._base_prompt: str = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Carga el CerebroJarvis.md una sola vez al arrancar. Nunca se vuelve a leer del disco."""
        prompt_path = Path("CerebroJarvis.md")
        if prompt_path.exists():
            prompt = prompt_path.read_text(encoding="utf-8")
            print(f"Jarvis> [Cerebro cargado: {len(prompt)} caracteres]")
            return prompt
        print("Jarvis> [Advertencia: CerebroJarvis.md no encontrado. Usando prompt base.]")
        return "Eres Jarvis, un asistente de escritorio. Clasifica la entrada y responde solo JSON valido."

    def analyze(self, text: str) -> dict[str, Any]:
        """
        Envía el texto del usuario al LLM junto con el prompt del sistema.
        Retorna un dict con la intención, acción y respuesta según el CerebroJarvis.md
        """
        try:
            messages = self._build_messages(text)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"}
            )
            
            response_text = response.choices[0].message.content or "{}"
            parsed = self._parse_response(response_text)

            # Actualizar memoria a corto plazo
            self.history.append({"role": "user", "content": text})
            assistant_reply = parsed.get("response", "")
            self.history.append({"role": "assistant", "content": f"Jarvis: {assistant_reply}"})
            
            # Limitar historial (ultimos 10 mensajes = 5 interacciones)
            if len(self.history) > 10:
                self.history = self.history[-10:]

            # Actualizar memoria a largo plazo si lo pide el JSON
            save_memory = parsed.get("save_memory")
            if isinstance(save_memory, dict):
                self.memory.update_dict(save_memory)
                print(f"Jarvis> [Memoria Actualizada]: {save_memory}")

            return parsed
        except Exception as e:
            print(f"Jarvis> [Debug OpenAI] Error: {e}")
            from config import settings
            if settings.local_fallback_enabled:
                return self._fallback_analysis(text)
            return {
                "intent": "chat",
                "response": "No pude conectarme al cerebro principal y la asistencia de emergencia local esta desactivada."
            }

    def _build_messages(self, text: str) -> list[dict]:
        """Construye la lista de mensajes inyectando la memoria a largo plazo al prompt ya cacheado."""
        prompt = self._base_prompt
        prompt += (
            "\n\n==================================================\n"
            "PROJECT SELF-KNOWLEDGE:\n"
            f"{render_project_self_knowledge()}\n"
            "=================================================="
        )

        known_memory = self.memory.get_all()
        if known_memory:
            memory_str = json.dumps(known_memory, indent=2, ensure_ascii=False)
            prompt += f"\n\n==================================================\nCURRENT LONG-TERM MEMORY:\n{memory_str}\n=================================================="

        messages = [{"role": "system", "content": prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": text})
        return messages

    def stream_chat(self, text: str) -> Generator[str, None, None]:
        """
        Streaming de respuesta de chat por oraciones completas.
        Yield: una oración a la vez, tan pronto como termina de formarse.
        Para respuestas de tipo 'chat' SOLAMENTE.
        """
        SENTENCE_END = re.compile(r'(?<=[.!?…])\s+|(?<=[,;:])\s+(?=\S{4,})')
        buffer = ""
        full_response = ""

        try:
            messages = self._build_messages(text)
            # Usamos texto plano en streaming (no json_object, no compatible con stream)
            messages[0]["content"] += (
                "\n\nIMPORTANT FOR THIS REQUEST: Respond in plain text only (no JSON). "
                "Give a concise, natural spoken answer in Spanish. No markdown, no lists."
            )

            with self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
            ) as stream:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    buffer += delta
                    full_response += delta

                    # Detectar fin de oración para emitir
                    parts = SENTENCE_END.split(buffer)
                    if len(parts) > 1:
                        for sentence in parts[:-1]:
                            sentence = sentence.strip()
                            if sentence:
                                yield sentence
                        buffer = parts[-1]

            # Emitir lo que quede en el buffer
            remaining = buffer.strip()
            if remaining:
                yield remaining

            # Actualizar historial con la respuesta completa
            self.history.append({"role": "user", "content": text})
            self.history.append({"role": "assistant", "content": full_response})
            if len(self.history) > 10:
                self.history = self.history[-10:]

        except Exception as e:
            print(f"Jarvis> [Debug Streaming] Error: {e}")
            yield "Disculpe, hubo un error al procesar mi respuesta."

    def summarize_conversation_log(self, log_text: str) -> str:
        if not log_text.strip():
            return "Hoy aun no tengo suficientes interacciones registradas para resumir."

        try:
            trimmed_log = log_text[-12000:]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres Jarvis revisando tu bitacora diaria. Resume en espanol, de forma breve, "
                            "que aprendiste del usuario, que tareas se hicieron, que errores o mejoras quedan "
                            "pendientes y que deberias recordar. No incluyas secretos, claves API ni datos sensibles."
                        ),
                    },
                    {
                        "role": "user",
                        "content": trimmed_log,
                    },
                ],
                max_tokens=360,
            )
            return (response.choices[0].message.content or "").strip() or "No pude resumir la bitacora de hoy."
        except Exception as exc:
            print(f"Jarvis> [Debug Log Summary] Error: {exc}")
            return "No pude revisar la bitacora de hoy en este momento."

    def reflect_on_conversation_log(self, log_text: str) -> dict[str, Any]:
        if not log_text.strip():
            return {
                "summary": "",
                "durable_lessons": [],
                "followups": [],
            }

        try:
            trimmed_log = log_text[-14000:]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Eres Jarvis consolidando aprendizaje diario. Devuelve SOLO JSON valido. "
                            "Extrae aprendizaje durable y util para operar mejor manana. "
                            "No guardes secretos, claves, tokens, passwords, contenido privado de pantalla, "
                            "ni detalles sensibles. No inventes. Si algo requiere confirmacion del usuario, "
                            "ponlo en followups y no en durable_lessons."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            "Bitacora del dia:\n"
                            f"{trimmed_log}\n\n"
                            "Formato requerido:\n"
                            "{"
                            "\"summary\":\"resumen breve\","
                            "\"durable_lessons\":[\"preferencia o regla util\"],"
                            "\"followups\":[\"pregunta o mejora pendiente\"]"
                            "}"
                        ),
                    },
                ],
                response_format={"type": "json_object"},
                max_tokens=500,
            )
            parsed = self._parse_response(response.choices[0].message.content or "{}")
            return {
                "summary": str(parsed.get("summary", "")).strip(),
                "durable_lessons": self._clean_string_list(parsed.get("durable_lessons", []), limit=8),
                "followups": self._clean_string_list(parsed.get("followups", []), limit=8),
            }
        except Exception as exc:
            print(f"Jarvis> [Debug Reflection] Error: {exc}")
            return {
                "summary": "",
                "durable_lessons": [],
                "followups": [],
            }

    def _clean_string_list(self, value: Any, limit: int = 8) -> list[str]:
        if not isinstance(value, list):
            return []

        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                cleaned.append(text[:240])
            if len(cleaned) >= limit:
                break
        return cleaned

    def _parse_response(self, response_text: str) -> dict[str, Any]:
        """Intenta parsear el JSON de la respuesta."""
        try:
            cleaned = response_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)

            return parsed
        except json.JSONDecodeError:
            return {
                "intent": "chat",
                "response": "Lo siento, mi respuesta fue confusa internamente."
            }

    def _fallback_analysis(self, text: str) -> dict[str, Any]:
        import unicodedata
        
        # Normalizar el texto (pasar a minúsculas y quitar acentos)
        normalized = unicodedata.normalize("NFD", text.lower())
        normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
        
        # ----------------------------------------------------
        # 1. Comandos de Reposo / Standby
        # ----------------------------------------------------
        sleep_phrases = [
            "descansa", "puedes descansar", "duerme", "puedes dormir", 
            "ve a dormir", "entra en reposo", "modo reposo", "desactivate",
            "gracias jarvis", "muchas gracias jarvis", "eso es todo"
        ]
        if any(phrase in normalized for phrase in sleep_phrases):
            return {
                "intent": "action",
                "action": "standby",
                "response": "Entendido, señor. Entrando en reposo localmente."
            }

        # ----------------------------------------------------
        # 2. Comandos de Seguridad / Acceso Privado
        # ----------------------------------------------------
        if any(phrase in normalized for phrase in ["identificame", "verifica mi identidad", "quien soy", "escanea mi rostro", "reconoce mi rostro"]):
            return {
                "intent": "action",
                "action": "identify_user",
                "response": "Iniciando escaneo facial local para verificar su identidad."
            }
            
        if any(phrase in normalized for phrase in ["registra mi rostro", "registra mi cara", "registrar propietario", "registrar dueno", "registrar dueño"]):
            return {
                "intent": "action",
                "action": "enroll_owner_face",
                "response": "De acuerdo. Procedamos con el registro del rostro del propietario."
            }

        if any(phrase in normalized for phrase in ["autorizados", "personas autorizadas", "rostros registrados", "quien esta autorizado"]):
            return {
                "intent": "action",
                "action": "list_authorized_faces",
                "response": "Consultando la lista de personas autorizadas en la base de datos local."
            }

        if any(phrase in normalized for phrase in ["bloquea el acceso privado", "bloquea acceso privado", "cierra acceso privado"]):
            return {
                "intent": "action",
                "action": "lock_private_access",
                "response": "Bloqueando el acceso privado de inmediato."
            }

        if any(phrase in normalized for phrase in ["activa modo invitado", "modo invitado"]):
            return {
                "intent": "action",
                "action": "set_guest_mode",
                "response": "Cambiando a modo invitado."
            }

        # ----------------------------------------------------
        # 3. Comandos de Sistema Críticos (Apagado / Reinicio / Bloqueo)
        # ----------------------------------------------------
        if any(phrase in normalized for phrase in ["apaga la pc", "apagar la pc", "apaga el equipo", "apagar el equipo"]):
            return {
                "intent": "action",
                "action": "shutdown_pc",
                "response": "Apagando el equipo en 30 segundos."
            }
        if any(phrase in normalized for phrase in ["reinicia la pc", "reiniciar la pc", "reinicia el equipo"]):
            return {
                "intent": "action",
                "action": "restart_pc",
                "response": "Reiniciando el equipo en 30 segundos."
            }
        if any(phrase in normalized for phrase in ["bloquea la pc", "bloquear la pc", "bloquea el equipo"]):
            return {
                "intent": "action",
                "action": "lock_pc",
                "response": "Bloqueando la estación de trabajo."
            }
        if "cancela el apagado" in normalized or "cancelar apagado" in normalized:
            return {
                "intent": "action",
                "action": "cancel_shutdown",
                "response": "Cancelando la orden de apagado."
            }

        # ----------------------------------------------------
        # 4. Volumen y Brillo
        # ----------------------------------------------------
        if "silencia" in normalized or "mute" in normalized:
            return {"intent": "action", "action": "mute", "response": "Silenciando el sistema."}
        if "sonido" in normalized and ("restaur" in normalized or "activa" in normalized) or "unmute" in normalized:
            return {"intent": "action", "action": "unmute", "response": "Sonido restaurado."}
        if "sube el volumen" in normalized or "subir volumen" in normalized:
            return {"intent": "action", "action": "volume_up", "response": "Subiendo el volumen."}
        if "baja el volumen" in normalized or "bajar volumen" in normalized:
            return {"intent": "action", "action": "volume_down", "response": "Bajando el volumen."}
        
        volume_match = re.search(r"volumen\s+(?:al|a|en)\s+(\d{1,3})", normalized)
        if volume_match:
            vol = max(0, min(100, int(volume_match.group(1))))
            return {
                "intent": "action",
                "action": "set_volume",
                "action_input": str(vol),
                "response": f"Estableciendo el volumen al {vol} por ciento."
            }

        if "sube el brillo" in normalized or "subir brillo" in normalized:
            return {"intent": "action", "action": "brightness_up", "response": "Subiendo el brillo de la pantalla."}
        if "baja el brillo" in normalized or "bajar brillo" in normalized:
            return {"intent": "action", "action": "brightness_down", "response": "Bajando el brillo de la pantalla."}
            
        brightness_match = re.search(r"brillo\s+(?:al|a|en)\s+(\d{1,3})", normalized)
        if brightness_match:
            bright = max(0, min(100, int(brightness_match.group(1))))
            return {
                "intent": "action",
                "action": "set_brightness",
                "action_input": str(bright),
                "response": f"Estableciendo el brillo al {bright} por ciento."
            }

        # ----------------------------------------------------
        # 5. Recordatorios y Alarmas
        # ----------------------------------------------------
        reminder_match = re.search(
            r"recuerdame\s+(.+?)\s+en\s+(\d+)\s+minuto", normalized
        )
        if not reminder_match:
            reminder_match = re.search(
                r"en\s+(\d+)\s+minutos\s+recuerdame\s+(.+)", normalized
            )
            if reminder_match:
                minutes, message = reminder_match.group(1), reminder_match.group(2)
                return {
                    "intent": "action",
                    "action": "add_reminder",
                    "action_input": f"{minutes}|{message}",
                    "response": f"Registrando recordatorio local: {message} en {minutes} minutos."
                }
        else:
            message, minutes = reminder_match.group(1), reminder_match.group(2)
            return {
                "intent": "action",
                "action": "add_reminder",
                "action_input": f"{minutes}|{message}",
                "response": f"Registrando recordatorio local: {message} en {minutes} minutos."
            }

        alarm_match = re.search(r"alarma\s+(?:a\s+las\s+)?(\d{1,2}:\d{2})(?:\s+para\s+(.+))?", normalized)
        if alarm_match:
            time_val, message = alarm_match.group(1), alarm_match.group(2) or "Alarma"
            if len(time_val.split(":")[0]) == 1:
                time_val = f"0{time_val}"
            return {
                "intent": "action",
                "action": "add_alarm",
                "action_input": f"{time_val}|{message}",
                "response": f"Programando alarma local a las {time_val} para {message}."
            }

        if any(phrase in normalized for phrase in ["recordatorios", "alarmas"]) and any(q in normalized for q in ["que", "lista", "muestra", "ver"]):
            return {
                "intent": "action",
                "action": "list_reminders",
                "response": "Consultando sus recordatorios locales."
            }

        if any(phrase in normalized for phrase in ["cancela todos los recordatorios", "elimina mis recordatorios"]):
            return {
                "intent": "action",
                "action": "cancel_reminders",
                "response": "Cancelando todos sus recordatorios locales."
            }

        # ----------------------------------------------------
        # 6. Estadísticas y Hora
        # ----------------------------------------------------
        if any(phrase in normalized for phrase in ["estado del sistema", "ram en uso", "uso de ram", "uso de cpu", "sistema stats"]):
            return {
                "intent": "action",
                "action": "get_system_stats",
                "response": "Consultando estadísticas de hardware locales."
            }

        if any(phrase in normalized for phrase in ["que hora es", "dime la hora", "hora actual"]):
            return {"intent": "action", "action": "get_time", "response": "Consultando la hora."}
            
        if any(phrase in normalized for phrase in ["bateria", "pila"]):
            return {"intent": "action", "action": "get_battery", "response": "Consultando el estado de la batería."}

        # ----------------------------------------------------
        # 7. Aplicaciones Comunes
        # ----------------------------------------------------
        if "chrome" in normalized or "navegador" in normalized:
            return {"intent": "action", "action": "open_chrome", "response": "Abriendo Google Chrome."}
        if "visual studio code" in normalized or "vscode" in normalized or "vs code" in normalized:
            return {"intent": "action", "action": "open_vscode", "response": "Abriendo Visual Studio Code."}
        if "spotify" in normalized:
            return {"intent": "action", "action": "open_spotify", "response": "Abriendo Spotify."}
        if "github" in normalized:
            return {"intent": "action", "action": "open_github", "response": "Abriendo GitHub."}
        if "terminal" in normalized or "consola" in normalized:
            return {"intent": "action", "action": "open_terminal", "response": "Abriendo la terminal."}
        if "explorador" in normalized or "archivos" in normalized:
            return {"intent": "action", "action": "open_explorer", "response": "Abriendo el explorador de archivos."}

        # ----------------------------------------------------
        # 8. Visión
        # ----------------------------------------------------
        if "pantalla" in normalized and any(word in normalized for word in ["ves", "ver", "mira", "analiza", "observa"]):
            return {
                "intent": "action",
                "action": "analyze_screen",
                "response": "Capturando pantalla para análisis local."
            }
        if "camara" in normalized and any(word in normalized for word in ["ves", "ver", "mira", "analiza", "observa"]):
            return {
                "intent": "action",
                "action": "analyze_camera",
                "response": "Iniciando la cámara para análisis local."
            }

        return {
            "intent": "chat",
            "response": "En este momento no tengo conexión a Internet y este comando excede mis capacidades locales básicas."
        }
