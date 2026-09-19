from unittest.mock import AsyncMock, patch

import pytest
from livekit.plugins import gladia

from agent import Assistant


def test_assistant_initialization():
    """Verify Assistant initialization and system prompt content."""
    assistant = Assistant()
    assert assistant is not None
    instructions = assistant.instructions
    assert "English, Telugu, and Hindi" in instructions
    assert "You are NOT a doctor" in instructions
    assert "search_health_knowledge" in instructions
    assert (
        "search_health_knowledge, translate or format the search query into concise English keywords"
        in instructions
    )
    assert "severe difficulty breathing" in instructions


def test_gladia_stt_configuration():
    """Verify Gladia STT configuration with English, Telugu, and Hindi."""
    stt = gladia.STT(
        api_key="mock_gladia_key",
        languages=["en", "te", "hi"],
        code_switching=True,
    )
    assert stt is not None
    assert stt.provider.lower() == "gladia"


@pytest.mark.asyncio
async def test_search_health_knowledge_tool():
    """Verify search_health_knowledge tool function signature and execution."""
    assistant = Assistant()

    mock_doc = AsyncMock()
    mock_doc.id = "fever_guidance.md"
    mock_doc.score = 0.95
    mock_doc.text = "Fever is a temporary increase in body temperature."

    mock_results = AsyncMock()
    mock_results.docs = [mock_doc]

    mock_client = AsyncMock()
    mock_client.query = AsyncMock(return_value=mock_results)

    with patch.object(assistant, "_get_moss_client", return_value=mock_client):
        result = await assistant.search_health_knowledge(None, "fever warning signs")
        assert "Document ID: fever_guidance.md" in result
        assert "Fever is a temporary increase" in result
