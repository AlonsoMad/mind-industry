"""
Schema Mapper Module.

Converts intermediate parsed records into the application's Dataset schema
with mandatory language detection limited to EN, ES, DE, IT.
"""

import logging
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {"EN", "ES", "DE", "IT"}


def detect_language(text: str) -> str:
    """
    Detect the language of a text string.

    Only returns languages in the supported set: EN, ES, DE, IT.
    Raises ValueError if the detected language is unsupported or
    detection fails.

    Args:
        text: The text to detect the language of.

    Returns:
        Two-letter language code (e.g., "EN", "ES").

    Raises:
        ValueError: If detected language is not supported.
        ImportError: If langdetect package is not installed.
    """
    try:
        from langdetect import detect
    except ImportError:
        raise ImportError(
            "langdetect package is required for language detection. "
            "Install with: pip install langdetect"
        )

    try:
        lang = detect(text).upper()
    except Exception as e:
        raise ValueError(
            f"Language detection failed for text: '{text[:80]}...'. Error: {e}"
        )

    # langdetect returns ISO 639-1 codes
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"Unsupported language detected: '{lang}'. "
            f"This software supports: {', '.join(sorted(SUPPORTED_LANGUAGES))}. "
            f"Text sample: '{text[:80]}...'"
        )

    return lang


def records_to_dataframe(
    records: List[Dict[str, Any]],
    source_name: str,
) -> pd.DataFrame:
    """
    Convert parsed records to a DataFrame matching the Dataset schema.

    Target schema columns:
    - id_preproc (str): Unique identifier per record (e.g., "source_0")
    - text (str): The extracted textual content (≥ 100 chars)
    - lang (str): Language code: EN, ES, DE, or IT (mandatory)
    - title (str): Document or section title

    Args:
        records: List of dicts from parsers, each with at least "text".
        source_name: Base name used for generating IDs.

    Returns:
        pd.DataFrame with the target schema columns.

    Raises:
        ValueError: If any record fails language detection.
    """
    rows = []
    errors = []

    for i, record in enumerate(records):
        try:
            # Use pre-existing lang if available, otherwise detect
            lang = record.get("lang") or detect_language(record["text"])
            rows.append({
                "id_preproc": f"{source_name}_{i}",
                "text": record["text"],
                "lang": lang,
                "title": record.get("title", source_name),
            })
        except ValueError as e:
            errors.append(str(e))

    if errors:
        raise ValueError(
            f"Language detection errors in {len(errors)} record(s):\n"
            + "\n".join(errors[:5])  # Show first 5 errors max
        )

    df = pd.DataFrame(rows)

    # Drop rows with empty/whitespace-only text (safety net)
    if not df.empty:
        df = df[df["text"].str.strip().astype(bool)]
        df = df.reset_index(drop=True)

    return df
