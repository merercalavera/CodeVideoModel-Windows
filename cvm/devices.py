"""Device selection shared by inference and the lightweight readiness checker."""


def select_text_device(requested, generation_device, device_count):
    if requested != "auto":
        return requested
    # Preserve the original two-GPU placement where possible.
    return "cuda:1" if device_count > 1 else generation_device
