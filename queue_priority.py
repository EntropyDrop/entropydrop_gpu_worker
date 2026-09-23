"""Shared Redis priority protocol with the backend (no database dependency)."""


def effective_queue_prefix(connection, log_id: str, prefix: str) -> str:
    if prefix == "high_":
        return prefix
    promoted = connection.get(f"generation:pro_priority:{log_id}")
    return "high_" if promoted in (b"1", "1") else ""
