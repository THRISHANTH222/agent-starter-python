import pytest

from agent import Assistant


@pytest.mark.asyncio
async def test_search_health_knowledge_returns_relevant_docs():
    """Verify that search_health_knowledge queries MOSS and returns relevant medical documents."""
    agent = Assistant()
    result = await agent.search_health_knowledge(
        context=None, query="I have fever and difficulty breathing"
    )

    # Check that search results contain relevant medical document IDs and text
    assert "Document ID:" in result
    assert any(
        doc_id in result
        for doc_id in ["fever", "breathing_difficulty", "emergency", "cough"]
    )
