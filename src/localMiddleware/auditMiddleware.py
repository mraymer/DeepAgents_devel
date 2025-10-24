# audit_middleware.py
from __future__ import annotations
import json, time, pathlib, re, uuid
from typing import Any, Optional
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage

def _safe_file_name(name: str) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip())[:120]
    return s or f"subagent_{uuid.uuid4().hex[:8]}"

class AuditMiddleware(AgentMiddleware):
    """
    Logs only the final human-readable answer from a subagent to JSONL.
    - Skips writing if no final text found.
    - Truncates payload to `max_log` characters (default 1000), then appends
      '##Truncated, max_log=xxx, original_size=yyy]' when truncation occurs.
    - Compatible with sync (.invoke) and async (.ainvoke) runs and variant hook signatures.
    """

    def __init__(
        self,
        audit_path: str | pathlib.Path,
        subagent_name: str,
        *,
        max_log: int = 5000,
    ):
        path = pathlib.Path(audit_path)
        self.file = path if path.suffix else path / f"{_safe_file_name(subagent_name)}.jsonl"
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.subagent_name = subagent_name
        self.max_log = int(max_log)

    # ---------- internals ----------
    def _write(self, rec: dict) -> None:
        self.file.open("a", encoding="utf-8").write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _find_text(self, obj: Any) -> Optional[str]:
        """Recursively extract a concise, human-readable final answer."""
        if obj is None:
            return None
        if isinstance(obj, str):
            text = obj.strip()
            if text and len(text) < 8000:  # ignore giant blobs (likely tool dumps)
                return text
            return None
        if isinstance(obj, AIMessage):
            text = (obj.content or "").strip()
            if text and len(text) < 8000:
                return text
            return None
        if isinstance(obj, dict):
            # Try common keys first
            for k in ("content", "text", "message", "output", "messages", "result"):
                if k in obj:
                    val = self._find_text(obj[k])
                    if val:
                        return val
            # Fallback: search all values
            for v in obj.values():
                val = self._find_text(v)
                if val:
                    return val
            return None
        if isinstance(obj, (list, tuple)):
            # Search from the end (latest message last)
            for v in reversed(obj):
                val = self._find_text(v)
                if val:
                    return val
            return None
        # Last resort: stringify small, simple objects that look like natural language
        s = str(obj).strip()
        return s if (s and len(s) < 8000 and ("\n" in s or " " in s)) else None

    def _extract_final_text(self, response: Any) -> Optional[str]:
        # Try common LC/DeepAgents shapes
        for attr in ("result", "output", "content", "messages"):
            if hasattr(response, attr):
                text = self._find_text(getattr(response, attr))
                if text:
                    return text
        return self._find_text(response)

    def _truncate(self, text: str) -> str:
        if text is None:
            return text
        if len(text) <= self.max_log:
            return text
        original_size = len(text)
        return text[: self.max_log] + f"##Truncated, max_log={self.max_log}, original_size={original_size}]"

    def _record(self, response: Any):
        text = self._extract_final_text(response)
        if not text:  # skip null/empty results entirely
            return
        text = self._truncate(text)
        rec = {
            "ts": time.time(),
            "kind": "final_result",
            "subagent": self.subagent_name,
            "content": text,
        }
        self._write(rec)

    # ---------- sync (tolerant to 1-arg or 2-arg forms) ----------
    def after_model(self, *args, **kwargs):
        # Accept (response) or (request, response)
        response = args[-1] if args else kwargs.get("response")
        self._record(response)
        return response

    # ---------- async ----------
    async def aafter_model(self, *args, **kwargs):
        response = args[-1] if args else kwargs.get("response")
        self._record(response)
        return response
