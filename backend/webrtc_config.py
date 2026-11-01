import os
import base64
import hashlib
import hmac
import time
from collections.abc import Mapping


def build_rtc_configuration(secrets: Mapping[str, object] | None = None):
    """Build ICE configuration, preferring short-lived TURN REST credentials."""
    values = secrets or {}

    def get_value(name):
        value = values.get(name)
        if value is None or str(value).strip() == "":
            value = os.getenv(name)
        return str(value).strip() if value is not None else ""

    turn_urls = [
        url.strip()
        for url in get_value("AURA_TURN_URLS").split(",")
        if url.strip()
    ]
    turn_username = get_value("AURA_TURN_USERNAME")
    turn_password = get_value("AURA_TURN_PASSWORD")
    turn_shared_secret = get_value("AURA_TURN_EPHEMERAL_SECRET")
    try:
        turn_ttl = max(60, int(get_value("AURA_TURN_CREDENTIAL_TTL") or "3600"))
    except ValueError:
        turn_ttl = 3600

    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    if turn_urls and turn_shared_secret:
        expires_at = int(time.time()) + turn_ttl
        ephemeral_username = f"aura-{expires_at}"
        digest = hmac.new(
            turn_shared_secret.encode("utf-8"),
            ephemeral_username.encode("utf-8"),
            hashlib.sha1,
        ).digest()
        ephemeral_password = base64.b64encode(digest).decode("ascii")
        ice_servers.append(
            {
                "urls": turn_urls,
                "username": ephemeral_username,
                "credential": ephemeral_password,
            }
        )
    elif (
        turn_urls
        and turn_username
        and turn_password
        and get_value("AURA_ALLOW_STATIC_TURN_CREDENTIALS").lower() in {"1", "true", "yes"}
    ):
        ice_servers.append(
            {
                "urls": turn_urls,
                "username": turn_username,
                "credential": turn_password,
            }
        )
    return {"iceServers": ice_servers}
