from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "instaMIND Backend"
    app_env: str = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    storage_root: str = "data"
    uploads_dir_name: str = "uploads"
    reports_dir_name: str = "reports"
    alerts_dir_name: str = "alerts"

    emergency_latency_target_ms: int = 100
    video_never_leaves_device: bool = True
    offline_mode: bool = True

    model_mode: str = "mock"
    gemini_api_key: str = ""
    gemini_model_name: str = "gemini-1.5-flash"
    local_gemma_model_path: str = ""
    local_gemma_endpoint: str = "http://127.0.0.1:11434/api/generate"
    local_gemma_model_name: str = "gemma3n:e4b"

    pose_event_model_path: str = "models/pose_event_detector.keras"
    pose_event_label_path: str = "models/pose_event_labels.json"

    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    alert_email_from: str = "alerts@instaMIND.local"
    alert_email_to: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
