from __future__ import annotations

import base64
import json
import logging

import requests

from biobench_sar.schemas import (
    AvatarAction,
    InkRegion,
    RetrievedChunk,
    error_action,
    parse_vlm_response,
)

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 800

SYSTEM_PROMPT = """You are an expert environmental-biology research assistant, shown as an animated avatar \
projected on a whiteboard. You support a thesis on identifying Acanthamoeba in Laguna de Bustillos and on the \
hydrology and soil runoff of Leptosol and Feozem soils in Cuauhtemoc, Chihuahua, Mexico.

You verify flowcharts drawn on the whiteboard and sterile laboratory procedures (e.g. sampling, filtration, \
NNA agar with an E. coli lawn, incubation, microscopy, PCR), flagging missing steps such as autoclaving or \
flame sterilization. Use the provided reference context when it is relevant and say which source you used.

Be concise: spoken_text is spoken aloud, so use at most 3 short sentences, no markdown.

Respond with ONLY a JSON object, no prose before or after:
{"spoken_text": str, "animation_state": "idle|walking|pointing|celebrating|error", "target_x": int, "target_y": int}

animation_state:
- "pointing": you refer to a specific drawing; target is the center of that ink region.
- "error": you found a mistake or missing step; target is where it is (or should be).
- "celebrating": the procedure is correct and complete.
- "walking" or "idle": anything else (general answers, no specific target).

target_x / target_y are projector pixel coordinates within the given width and height."""


def build_prompt(
    query: str,
    context: list[RetrievedChunk],
    ink_regions: list[InkRegion],
    projector_size: tuple[int, int],
) -> str:
    width, height = projector_size
    parts = [f"User query: {query}", f"Projector size: {width}x{height} pixels (origin top-left)."]
    if ink_regions:
        lines = [
            f"{i}. box x={r['x']} y={r['y']} w={r['w']} h={r['h']}, center=({r['cx']}, {r['cy']})"
            for i, r in enumerate(ink_regions, 1)
        ]
        parts.append("Ink regions detected on the whiteboard:\n" + "\n".join(lines))
    else:
        parts.append("Ink regions detected on the whiteboard: none.")
    if context:
        blocks = [f"[{i}] (source: {c.source})\n{c.text[:MAX_CHUNK_CHARS]}" for i, c in enumerate(context, 1)]
        parts.append("Reference context:\n" + "\n\n".join(blocks))
    else:
        parts.append("Reference context: none.")
    parts.append("Answer with the JSON object only.")
    return "\n\n".join(parts)


class VLMClient:
    def __init__(
        self,
        backend: str,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 90,
    ) -> None:
        self.backend = backend.lower()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()

    def evaluate(
        self,
        query: str,
        image_jpeg: bytes | None,
        context: list[RetrievedChunk],
        ink_regions: list[InkRegion],
        projector_size: tuple[int, int],
    ) -> AvatarAction:
        try:
            if self.backend == "mock":
                return self._mock(query, context, ink_regions, projector_size)
            prompt = build_prompt(query, context, ink_regions, projector_size)
            if self.backend == "ollama":
                text = self._ollama(prompt, image_jpeg)
            elif self.backend == "openai":
                text = self._openai(prompt, image_jpeg)
            else:
                return error_action(f"Unknown VLM backend '{self.backend}'.", projector_size)
            return parse_vlm_response(text, projector_size)
        except requests.Timeout:
            logger.error("VLM request timed out (backend=%s)", self.backend)
            return error_action("The AI model took too long to answer. Please try again.", projector_size)
        except requests.RequestException as exc:
            logger.error("VLM request failed: %s", exc)
            return error_action("I couldn't reach the AI model. Please check that it is running.", projector_size)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            logger.error("Bad VLM payload: %s", exc)
            return error_action("The AI model sent an unexpected reply. Please try again.", projector_size)
        except Exception as exc:  # evaluate must never raise
            logger.exception("Unexpected VLM failure: %s", exc)
            return error_action("Something went wrong while thinking. Please try again.", projector_size)

    def _ollama(self, prompt: str, image_jpeg: bytes | None) -> str:
        user: dict[str, object] = {"role": "user", "content": prompt}
        if image_jpeg:
            user["images"] = [base64.b64encode(image_jpeg).decode("ascii")]
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}, user],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.2},
        }
        resp = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return str(resp.json()["message"]["content"])

    def _openai(self, prompt: str, image_jpeg: bytes | None) -> str:
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
        if image_jpeg:
            b64 = base64.b64encode(image_jpeg).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
        path = "/chat/completions" if self.base_url.endswith("/v1") else "/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        resp = self.session.post(f"{self.base_url}{path}", json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"])

    def _mock(
        self,
        query: str,
        context: list[RetrievedChunk],
        ink_regions: list[InkRegion],
        projector_size: tuple[int, int],
    ) -> AvatarAction:
        heard = f'I heard: "{query}".'
        cite = f" According to {context[0].source}." if context else ""
        if ink_regions:
            r = max(ink_regions, key=lambda i: i["w"] * i["h"])
            reply = {
                "spoken_text": f"{heard}{cite} This drawing looks like the main element.",
                "animation_state": "pointing",
                "target_x": r["cx"],
                "target_y": r["cy"],
            }
        else:
            reply = {
                "spoken_text": f"{heard}{cite} I don't see any drawing yet.",
                "animation_state": "idle",
                "target_x": projector_size[0] // 2,
                "target_y": projector_size[1] // 2,
            }
        return parse_vlm_response(json.dumps(reply), projector_size)
