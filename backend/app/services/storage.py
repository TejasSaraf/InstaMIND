import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings


class StorageService:
    def __init__(self) -> None:
        self.root = Path(settings.storage_root)
        self.uploads = self.root / settings.uploads_dir_name
        self.reports = self.root / settings.reports_dir_name
        self.alerts = self.root / settings.alerts_dir_name
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.reports.mkdir(parents=True, exist_ok=True)
        self.alerts.mkdir(parents=True, exist_ok=True)

    def save_upload(self, filename: str, content: bytes) -> Path:
        suffix = Path(filename).suffix or ".mp4"
        safe_name = f"{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex}{suffix}"
        file_path = self.uploads / safe_name
        file_path.write_bytes(content)
        return file_path

    def save_report(self, report_id: str, payload: dict[str, Any]) -> Path:
        out = self.reports / f"{report_id}.json"
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return out

    def save_local_alert(self, report_id: str, payload: dict[str, Any]) -> Path:
        out = self.alerts / f"{report_id}.json"
        out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return out

    def load_report(self, report_id: str) -> dict[str, Any]:
        path = self.reports / f"{report_id}.json"
        if not path.exists():
            raise FileNotFoundError(report_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def list_reports(self) -> list[dict[str, Any]]:
        reports = []
        for file_path in sorted(self.reports.glob("*.json"), reverse=True):
            reports.append(json.loads(file_path.read_text(encoding="utf-8")))
        return reports
