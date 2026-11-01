import os
from collections.abc import Mapping


def build_rtc_configuration(secrets: Mapping[str, object] | None = None):
    """Build the browser/server ICE configuration without exposing credentials."""
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

    ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
    if turn_urls and turn_username and turn_password:
        ice_servers.append(
            {
                "urls": turn_urls,
                "username": turn_username,
                "credential": turn_password,
            }
        )
    return {"iceServers": ice_servers}
