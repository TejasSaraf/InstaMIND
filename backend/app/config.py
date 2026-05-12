from pydantic_settings import BaseSettings, SettingsConfigDict

from pathlib import Path





BACKEND_ROOT = Path(__file__).resolve().parents[1]

BACKEND_ENV_PATH = BACKEND_ROOT / ".env"





def _resolve_backend_path(raw_path: str) -> str:

    path = Path(raw_path).expanduser()

    if not path.is_absolute():

        path = BACKEND_ROOT / path

    return str(path)





class Settings(BaseSettings):

    app_name: str = "instaMIND Backend"

    app_env: str = "dev"

    app_host: str = "0.0.0.0"

    app_port: int = 8000



    storage_root: str = str(BACKEND_ROOT / "data")

    uploads_dir_name: str = "uploads"

    reports_dir_name: str = "reports"

    alerts_dir_name: str = "alerts"



    emergency_latency_target_ms: int = 100

    video_never_leaves_device: bool = True

    offline_mode: bool = True



    model_mode: str = "mock"

    upload_classifier_mode: str = "local_gemma"

    gemini_api_key: str = ""

    gemini_model_name: str = "gemini-1.5-flash"

    local_gemma_model_path: str = ""

    local_gemma_model_dir: str = str(

        BACKEND_ROOT / "models" / "gemma3-4b-instamind-mlx")

    local_gemma_endpoint: str = "http://127.0.0.1:11434/api/generate"

    local_gemma_model_name: str = "gemma3:4b"

    local_gemma_runtime: str = "mlx"

    local_gemma_timeout_seconds: int = 600

    require_local_gemma: bool = False

    local_gemma_strict_primary: bool = True

    function_gemma_enabled: bool = True

    function_gemma_model_name: str = ""

    function_gemma_endpoint: str = ""

    agentic_max_steps: int = 4



    llamacpp_vision_endpoint: str = "http://127.0.0.1:8081/v1/chat/completions"

    llamacpp_frame_interval_seconds: float = 3.0

    llamacpp_timeout_seconds: int = 30



    pose_event_model_path: str = str(

        BACKEND_ROOT / "models" / "pose_event_detector.keras")

    pose_event_label_path: str = str(

        BACKEND_ROOT / "models" / "pose_event_labels.json")



    audio_signal_extraction_enabled: bool = True



    threshold_fainting_fast: float = 0.8

    threshold_fighting_fast: float = 0.7

    threshold_shooting_fast: float = 0.7

    threshold_shoplifting_fast: float = 0.55



    guardrail_shoplifting_min: float = 0.5

    guardrail_fainting_max: float = 0.45



    smtp_host: str = ""

    smtp_port: int = 587

    smtp_username: str = ""

    smtp_password: str = ""

    alert_email_from: str = "alerts@instaMIND.local"

    alert_email_to: str = ""



    sendgrid_api_key: str = ""

    sendgrid_from_email: str = "alerts@instamind.local"

    sendgrid_to_email: str = ""

    alert_phone_to: str = ""



    twilio_account_sid: str = ""

    twilio_auth_token: str = ""

    twilio_from_number: str = ""



    supabase_url: str = ""

    supabase_service_role_key: str = ""

    supabase_users_table: str = "users"



    google_oauth_client_id: str = ""



    model_config = SettingsConfigDict(

        env_file=str(BACKEND_ENV_PATH),

        env_file_encoding="utf-8",

        extra="ignore",

    )





settings = Settings()

settings.storage_root = _resolve_backend_path(settings.storage_root)

if settings.local_gemma_model_dir:

    settings.local_gemma_model_dir = _resolve_backend_path(

        settings.local_gemma_model_dir)

if settings.local_gemma_model_path:

    settings.local_gemma_model_path = _resolve_backend_path(

        settings.local_gemma_model_path)

settings.pose_event_model_path = _resolve_backend_path(

    settings.pose_event_model_path)

settings.pose_event_label_path = _resolve_backend_path(

    settings.pose_event_label_path)
