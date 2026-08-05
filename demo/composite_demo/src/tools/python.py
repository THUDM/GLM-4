"""Compatibility stub for the removed host-Python tool.

Executing model-generated code in the Streamlit process gives that code the
same file, network, and credential access as the application.  The tool is no
longer registered; keeping this fail-closed stub makes direct imports safe and
gives downstream users an explicit migration error.
"""

from .interface import ToolObservation


def tool_call(code: str, session_id: str) -> list[ToolObservation]:
    del code, session_id
    raise RuntimeError(
        "The host Python tool is disabled. Run code only in a separately "
        "sandboxed service with explicit resource and network policies."
    )
