
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from starlette.testclient import TestClient
import agent

@pytest.fixture
def client():
    """Fixture to provide a FastAPI test client for the agent app."""
    return TestClient(agent.app)

@pytest.mark.asyncio
def test_functional_successful_query_endpoint_returns_valid_queryresponse(client):
    """
    Validates that the /query endpoint returns a successful QueryResponse with expected structure
    when all dependencies are available and documents exist.
    """
    # Patch ChunkRetriever.retrieve_chunks to return synthetic chunks
    fake_chunks = [
        "Earth's equatorial diameter is 7,926 miles (12,756 km).",
        "Jupiter's equatorial diameter is 88,846 miles (142,984 km).",
        "Jupiter is much larger than Earth; over 1,300 Earths could fit inside Jupiter.",
        "Earth is 93 million miles (150 million km) from the Sun.",
        "Jupiter is 484 million miles (778 million km) from the Sun."
    ]
    # Patch LLMService.generate_response to return a plausible LLM output
    fake_llm_output = (
        "Planetary Comparison:\n"
        "- Earth: Diameter 7,926 miles (12,756 km), Distance from Sun 93 million miles (150 million km)\n"
        "- Jupiter: Diameter 88,846 miles (142,984 km), Distance from Sun 484 million miles (778 million km)\n"
        "- Jupiter is so large that over 1,300 Earths could fit inside it.\n"
        "See Earth.pdf and Jupiter.pdf for details."
    )

    # Patch the async methods used by the agent
    with patch("agent.ChunkRetriever.retrieve_chunks", new_callable=AsyncMock) as mock_retrieve_chunks, \
         patch("agent.LLMService.generate_response", new_callable=AsyncMock) as mock_generate_response:
        mock_retrieve_chunks.return_value = fake_chunks
        mock_generate_response.return_value = fake_llm_output

        response = client.post("/query")
        assert response.status_code == 200

        data = response.json()
        # Validate QueryResponse structure
        assert isinstance(data, dict)
        assert data.get("success") is True
        result = data.get("result")
        assert isinstance(result, str)
        assert len(result.strip()) > 0
        assert data.get("error") is None
        assert data.get("tips") is None