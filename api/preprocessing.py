"""
Input preprocessing for API.
"""


def preprocess_text(text: str) -> str:

    if not isinstance(text, str):
        raise TypeError("News text must be a string")

    cleaned_text = text.strip()

    if not cleaned_text:
        raise ValueError("News text cannot be empty")

    return cleaned_text