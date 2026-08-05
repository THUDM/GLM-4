"""
Simple browser tool.

# Usage

Please start the backend browser server according to the instructions in the README.
"""

import json
import os
import re
from dataclasses import dataclass
from pprint import pprint

import requests
import streamlit as st

from .config import BROWSER_SERVER_URL
from .interface import ToolObservation


QUOTE_REGEX = re.compile(r"\[(\d+)†(.+?)\]")
BROWSER_RESPONSE_MAX_BYTES = max(1, int(os.environ.get("BROWSER_RESPONSE_MAX_BYTES", 3 * 1024 * 1024)))
BROWSER_QUOTE_LIMIT = max(1, int(os.environ.get("BROWSER_MAX_QUOTES", 20)))
BROWSER_API_KEY = os.environ.get("BROWSER_API_KEY")


@dataclass
class Quote:
    title: str
    url: str


# Quotes for displaying reference
if "quotes" not in st.session_state:
    st.session_state.quotes = {}

quotes: dict[str, Quote] = st.session_state.quotes


def map_response(response: dict) -> ToolObservation:
    # Save quotes for reference
    print("===BROWSER_RESPONSE===")
    pprint(response)
    role_metadata = response.get("roleMetadata")
    metadata = response.get("metadata")

    metadata_list = metadata.get("metadata_list", []) if isinstance(metadata, dict) else []
    quote_match = QUOTE_REGEX.search(role_metadata) if isinstance(role_metadata, str) else None
    if role_metadata and role_metadata.split()[0] == "quote_result" and quote_match and metadata_list:
        quote = metadata_list[0]
        if isinstance(quote, dict) and isinstance(quote.get("title"), str) and isinstance(quote.get("url"), str):
            quotes[quote_match.group(1)] = Quote(quote["title"], quote["url"])
    elif role_metadata == "browser_result":
        for i, quote in enumerate(metadata_list):
            if isinstance(quote, dict) and isinstance(quote.get("title"), str) and isinstance(quote.get("url"), str):
                quotes[str(i)] = Quote(quote["title"], quote["url"])
    while len(quotes) > BROWSER_QUOTE_LIMIT:
        oldest_quote_id = min(quotes, key=lambda value: int(value) if value.isdigit() else -1)
        del quotes[oldest_quote_id]

    return ToolObservation(
        content_type=response.get("contentType"),
        text=response.get("result"),
        role_metadata=role_metadata,
        metadata=metadata,
    )


def tool_call(code: str, session_id: str) -> list[ToolObservation]:
    request = {
        "session_id": session_id,
        "action": code,
    }
    headers = {"Authorization": f"Bearer {BROWSER_API_KEY}"} if BROWSER_API_KEY else None
    with requests.post(
        BROWSER_SERVER_URL, json=request, headers=headers, stream=True, timeout=(2, 20)
    ) as response:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                declared_length = int(content_length)
            except ValueError as exc:
                raise ValueError("Browser returned an invalid Content-Length") from exc
            if declared_length < 0 or declared_length > BROWSER_RESPONSE_MAX_BYTES:
                raise ValueError("Browser response is too large")
        chunks = []
        total = 0
        for chunk in response.iter_content(64 * 1024):
            total += len(chunk)
            if total > BROWSER_RESPONSE_MAX_BYTES:
                raise ValueError("Browser response is too large")
            chunks.append(chunk)
    payload = json.loads(b"".join(chunks))
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("Invalid browser response")
    return list(map(map_response, payload))
