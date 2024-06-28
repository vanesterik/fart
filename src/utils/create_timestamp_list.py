def create_timestamp_list() -> tuple[str,str]:
    """Create a list of timestamps for the last 1000 minutes."""
    # Get current time
    now = int(time.time())

    # Create a list of timestamps for the last 1000 minutes
    timestamps = [now - x * 60 for x in range(1000)]

    return timestamps