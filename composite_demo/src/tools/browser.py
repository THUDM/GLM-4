"""
Simple browser tool.

# Usage

Please start the backend browser server according to the instructions in the README.
"""

from pprint import pprint
import re
import requests
import streamlit as st
from dataclasses import dataclass

from .config import BROWSER_SERVER_URL
from .interface import ToolObservation

QUOTE_REGEX = re.compile(r"\[(\d+)†(.+?)\]")

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
    print('===BROWSER_RESPONSE===')
    pprint(response)
    role_metadata = response.get("roleMetadata")
    metadata = response.get("metadata")
    
    if role_metadata.split()[0] == 'quote_result' and metadata:
        quote_id = QUOTE_REGEX.search(role_metadata.split()[1]).group(1)
        quote: dict[str, str] = metadata['metadata_list'][0]
        quotes[quote_id] = Quote(quote['title'], quote['url'])
    elif role_metadata == 'browser_result' and metadata:
        for i, quote in enumerate(metadata['metadata_list']):
            quotes[str(i)] = Quote(quote['title'], quote['url'])

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
    response = requests.post(BROWSER_SERVER_URL, json=request).json()
    return list(map(map_response, response))
