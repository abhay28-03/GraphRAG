import pytest
from core.engine import GraphRAGEngine
from fastapi.testclient import TestClient
from app import app

@pytest.fixture(scope="module")
def engine_instance():
    # Initialize engine using local in-memory fallback without requiring live Neo4j for unit tests
    return GraphRAGEngine(api_key="mock_key_for_testing", neo4j_uri="bolt://invalid_host:7687")

def test_cosine_similarity_identical_vectors():
    import numpy as np
    vec = np.array([1.0, 2.0, 3.0])
    score = GraphRAGEngine.cosine_sim(vec, vec)
    assert pytest.approx(score, 0.0001) == 1.0

def test_entity_resolution_deduplication(engine_instance):
    # Pre-seed an entity and its embedding
    canonical = "Valeron Micro"
    engine_instance.entity_embeddings[canonical] = engine_instance.get_embedding(canonical)
    
    # Test that a slight variant resolves to the canonical form
    variant = "Valeron Micro Inc."
    resolved = engine_instance.resolve_entity(variant, threshold=0.85)
    assert resolved == canonical

def test_fastapi_schema_contract():
    client = TestClient(app)
    # Validate root redirects to graph visualizer
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/graph"

    # Validate bad payload rejection on query endpoint
    empty_payload_resp = client.post("/api/v1/query/local", json={"query": ""})
    assert empty_payload_resp.status_code == 400