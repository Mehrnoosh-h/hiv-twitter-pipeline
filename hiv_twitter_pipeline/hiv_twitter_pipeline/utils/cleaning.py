"""Value-cleaning helpers."""


def clean_values(value):
    """Standardise a GPT annotation string.

    * Strips leading/trailing whitespace.
    * Capitalises the first letter.
    * Removes a trailing period.

    Non-string values are returned unchanged.
    """
    if isinstance(value, str):
        value = value.strip().capitalize()
        if value.endswith("."):
            value = value[:-1]
    return value
