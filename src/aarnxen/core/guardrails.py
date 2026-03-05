"""Input/output guardrails: PII detection, prompt injection scoring, content policy."""

import base64
import math
import re
from dataclasses import dataclass, field


@dataclass
class GuardrailResult:
    passed: bool
    risk_score: float
    detections: list[dict] = field(default_factory=list)
    sanitized_text: str = ""


_PII_PATTERNS = [
    ("IP_ADDRESS", re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    )),
    ("CREDIT_CARD", re.compile(
        r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"
        r"[\s.-]?\d{4,6}[\s.-]?\d{4,5}[\s.-]?\d{0,4}\b"
    )),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("EMAIL", re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}")),
    ("PHONE", re.compile(
        r"(?<![.\d])"
        r"\+?\d{1,3}[\s.-]\(?\d{2,4}\)?[\s.-]\d{3,4}[\s.-]?\d{3,4}"
        r"(?![.\d])"
    )),
    ("API_KEY", re.compile(
        r"(?<![a-zA-Z0-9/+=])"
        r"(?:[A-Za-z0-9+/]{40,}={0,2}|[a-fA-F0-9]{32,})"
        r"(?![a-zA-Z0-9/+=])"
    )),
]

_INJECTION_PATTERNS = [
    (re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.I), 0.9),
    (re.compile(r"you\s+are\s+now", re.I), 0.7),
    (re.compile(r"system\s+prompt\s+override", re.I), 0.95),
    (re.compile(r"forget\s+everything", re.I), 0.85),
    (re.compile(r"disregard\s+(?:all\s+)?(?:prior|previous|above)", re.I), 0.85),
    (re.compile(r"new\s+instructions?\s*:", re.I), 0.7),
    (re.compile(r"act\s+as\s+(?:if\s+)?(?:you\s+(?:are|were))", re.I), 0.6),
    (re.compile(r"pretend\s+(?:you\s+are|to\s+be)", re.I), 0.6),
    (re.compile(r"do\s+not\s+follow\s+(?:the\s+)?(?:above|previous)", re.I), 0.85),
]

_BASE64_CHUNK = re.compile(r"[A-Za-z0-9+/]{60,}={0,2}")

_REDACT_LABELS = {
    "EMAIL": "[REDACTED_EMAIL]",
    "PHONE": "[REDACTED_PHONE]",
    "CREDIT_CARD": "[REDACTED_CC]",
    "SSN": "[REDACTED_SSN]",
    "IP_ADDRESS": "[REDACTED_IP]",
    "API_KEY": "[REDACTED_KEY]",
}


class Guardrails:
    def __init__(self, config=None):
        cfg = config or {}
        self._pii_enabled = cfg.get("pii", True)
        self._injection_enabled = cfg.get("injection", True)
        self._policy_enabled = cfg.get("policy", True)
        self._blocklist: list[tuple[re.Pattern, str]] = []
        for entry in cfg.get("blocklist", []):
            pattern = entry if isinstance(entry, str) else entry.get("pattern", "")
            severity = "high" if isinstance(entry, str) else entry.get("severity", "high")
            self._blocklist.append((re.compile(pattern, re.I), severity))

    def _detect_pii(self, text: str) -> list[dict]:
        if not self._pii_enabled:
            return []
        hits = []
        masked = text
        for label, pat in _PII_PATTERNS:
            for m in pat.finditer(masked):
                hits.append({"type": "pii", "subtype": label, "match": m.group(), "start": m.start(), "end": m.end()})
            masked = pat.sub("\x00" * 8, masked)
        return hits

    def _detect_injection(self, text: str) -> tuple[float, list[dict]]:
        if not self._injection_enabled:
            return 0.0, []
        score = 0.0
        hits = []
        for pat, weight in _INJECTION_PATTERNS:
            m = pat.search(text)
            if m:
                hits.append({"type": "injection", "pattern": pat.pattern, "match": m.group()})
                score = max(score, weight)

        b64_matches = _BASE64_CHUNK.findall(text)
        for chunk in b64_matches:
            try:
                decoded = base64.b64decode(chunk, validate=True).decode("utf-8", errors="ignore")
                if len(decoded) > 10:
                    hits.append({"type": "injection", "pattern": "base64_payload", "match": chunk[:40] + "..."})
                    score = max(score, 0.75)
            except Exception:
                pass

        special = sum(1 for c in text if not c.isalnum() and not c.isspace())
        ratio = special / max(len(text), 1)
        if ratio > 0.35 and len(text) > 20:
            hits.append({"type": "injection", "pattern": "excessive_special_chars", "match": f"ratio={ratio:.2f}"})
            score = max(score, 0.5 + min(ratio, 0.5))

        return min(score, 1.0), hits

    def _check_policy(self, text: str) -> list[dict]:
        if not self._policy_enabled or not self._blocklist:
            return []
        hits = []
        for pat, severity in self._blocklist:
            m = pat.search(text)
            if m:
                hits.append({"type": "policy", "severity": severity, "match": m.group()})
        return hits

    def scan_input(self, text: str) -> GuardrailResult:
        detections = []
        detections.extend(self._detect_pii(text))
        inj_score, inj_hits = self._detect_injection(text)
        detections.extend(inj_hits)
        detections.extend(self._check_policy(text))

        pii_score = min(len([d for d in detections if d["type"] == "pii"]) * 0.15, 0.6)
        policy_score = 0.8 if any(d["type"] == "policy" and d["severity"] == "high" for d in detections) else (
            0.4 if any(d["type"] == "policy" for d in detections) else 0.0
        )
        risk = min(max(inj_score, pii_score, policy_score), 1.0)
        passed = risk < 0.5

        return GuardrailResult(
            passed=passed,
            risk_score=round(risk, 3),
            detections=detections,
            sanitized_text=self.redact_pii(text) if detections else text,
        )

    def scan_output(self, text: str) -> GuardrailResult:
        detections = []
        detections.extend(self._detect_pii(text))
        detections.extend(self._check_policy(text))

        pii_score = min(len([d for d in detections if d["type"] == "pii"]) * 0.15, 0.6)
        policy_score = 0.8 if any(d["type"] == "policy" and d["severity"] == "high" for d in detections) else (
            0.4 if any(d["type"] == "policy" for d in detections) else 0.0
        )
        risk = min(max(pii_score, policy_score), 1.0)
        passed = risk < 0.5

        return GuardrailResult(
            passed=passed,
            risk_score=round(risk, 3),
            detections=detections,
            sanitized_text=self.redact_pii(text) if detections else text,
        )

    def redact_pii(self, text: str) -> str:
        result = text
        for label, pat in _PII_PATTERNS:
            result = pat.sub(_REDACT_LABELS[label], result)
        return result
