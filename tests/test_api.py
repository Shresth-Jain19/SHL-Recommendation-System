from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_recommend_valid_query():
    payload = {"query": "Looking for a cognitive and personality test for analysts."}
    response = client.post("/recommend?top_k=3", json=payload)
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert 1 <= len(results) <= 3
    for item in results:
        assert "name" in item
        assert "url" in item

def test_recommend_empty_query():
    payload = {"query": ""}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200 or response.status_code == 422  # Acceptable: warning or validation error