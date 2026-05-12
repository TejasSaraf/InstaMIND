from datetime import datetime, timezone

from typing import Any



from app.config import settings



_supabase_mod = None





def _get_supabase():

    global _supabase_mod

    if _supabase_mod is None:

        import supabase as _sb

        _supabase_mod = _sb

    return _supabase_mod





class UserStore:

    def __init__(self) -> None:

        self._client = None



    def _get_client(self):

        if self._client is not None:

            return self._client

        if not settings.supabase_url or not settings.supabase_service_role_key:

            raise ValueError(

                "Missing Supabase configuration. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."

            )

        sb = _get_supabase()

        self._client = sb.create_client(

            settings.supabase_url, settings.supabase_service_role_key)

        return self._client



    def upsert_google_user(self, claims: dict[str, Any]) -> dict[str, Any]:

        client = self._get_client()

        google_sub = str(claims.get("sub") or "").strip()

        email = str(claims.get("email") or "").strip().lower()

        if not google_sub or not email:

            raise ValueError("Google token missing required subject/email.")



        now_iso = datetime.now(timezone.utc).isoformat()

        payload = {

            "google_sub": google_sub,

            "email": email,

            "name": str(claims.get("name") or ""),

            "picture": claims.get("picture"),

            "email_verified": bool(claims.get("email_verified", False)),

            "updated_at": now_iso,

            "last_login_at": now_iso,

        }



        try:

            upsert_response = (

                client.table(settings.supabase_users_table)

                .upsert(payload, on_conflict="google_sub")

                .execute()

            )

        except Exception as exc:

            print(f"[UserStore] Supabase upsert failed: {exc}")

            raise RuntimeError(

                f"Database error during user upsert: {str(exc)}") from exc



        if not upsert_response.data:

            raise RuntimeError(

                "Failed to load user after upsert (empty data returned).")

        user = upsert_response.data[0]



        return {

            "id": str(user.get("id", "")),

            "google_sub": user["google_sub"],

            "email": user["email"],

            "name": user.get("name", ""),

            "picture": user.get("picture"),

            "email_verified": bool(user.get("email_verified", False)),

            "created_at": user.get("created_at") or now_iso,

            "updated_at": user.get("updated_at") or now_iso,

            "last_login_at": user.get("last_login_at") or now_iso,

        }



    def get_user_by_id(self, user_id: str) -> dict[str, Any] | None:

        client = self._get_client()

        response = (

            client.table(settings.supabase_users_table)

            .select("*")

            .eq("id", user_id)

            .limit(1)

            .execute()

        )

        if not response.data:

            return None

        user = response.data[0]

        return {

            "id": str(user.get("id", "")),

            "google_sub": user.get("google_sub", ""),

            "email": user.get("email", ""),

            "name": user.get("name", ""),

            "picture": user.get("picture"),

            "email_verified": bool(user.get("email_verified", False)),

            "created_at": user.get("created_at"),

            "updated_at": user.get("updated_at"),

            "last_login_at": user.get("last_login_at"),

        }
