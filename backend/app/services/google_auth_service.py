from typing import Any



from google.auth.transport import requests

from google.oauth2 import id_token



from app.config import settings





class GoogleAuthService:

    def verify_id_token(self, raw_id_token: str) -> dict[str, Any]:

        token = raw_id_token.strip()

        if not token:

            raise ValueError("Missing Google ID token.")

        if not settings.google_oauth_client_id:

            raise ValueError("Server missing GOOGLE_OAUTH_CLIENT_ID.")



        claims = id_token.verify_oauth2_token(

            token,

            requests.Request(),

            settings.google_oauth_client_id,

        )

        issuer = claims.get("iss")

        if issuer not in {"accounts.google.com", "https://accounts.google.com"}:

            raise ValueError("Invalid Google token issuer.")

        return claims
