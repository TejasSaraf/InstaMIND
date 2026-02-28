import smtplib
from email.mime.text import MIMEText

from app.config import settings
from app.schemas import IncidentReport, IncidentType
from app.services.storage import StorageService


class AlertNotifier:
    def __init__(self, storage: StorageService) -> None:
        self.storage = storage

    def notify_if_needed(self, report: IncidentReport) -> None:
        severe = [
            i
            for i in report.incidents
            if i.incident_type in {IncidentType.fainting, IncidentType.choking, IncidentType.violent_activity}
            and i.confidence >= 0.6
        ]
        if not severe:
            return

        payload = {
            "report_id": report.report_id,
            "summary": report.summary,
            "critical_incidents": [x.model_dump() for x in severe],
        }
        self.storage.save_local_alert(report.report_id, payload)
        self._send_email_alert(payload)

    def _send_email_alert(self, payload: dict) -> None:
        if not (settings.smtp_host and settings.alert_email_to and settings.smtp_username and settings.smtp_password):
            return

        msg = MIMEText(str(payload), "plain", "utf-8")
        msg["Subject"] = f"[InstaMIND] Critical Incident {payload['report_id']}"
        msg["From"] = settings.alert_email_from
        msg["To"] = settings.alert_email_to

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_username, settings.smtp_password)
            server.sendmail(settings.alert_email_from, [settings.alert_email_to], msg.as_string())
