def active_alerts(alert_history, acknowledged_ids, now, interval_seconds):
    acknowledged = set(acknowledged_ids or [])
    return [
        alert for alert in (alert_history or [])
        if alert["alert_id"] not in acknowledged
        and 0 <= now - alert["created_at"] < interval_seconds
    ]


def acknowledge_alert(alert_id, acknowledged_ids):
    acknowledged = list(acknowledged_ids or [])
    if alert_id not in acknowledged:
        acknowledged.append(alert_id)
    return acknowledged