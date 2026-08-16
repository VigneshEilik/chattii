import io

from fastapi.testclient import TestClient

from engine.api.api import app

AUTH_HEADERS = {"Authorization": "Bearer chatii-admin-root-key"}


def test_rag_pdf_ingestion_and_citation():
    with TestClient(app) as client:
        content = b"Gradient descent is an optimization algorithm used to minimize loss by moving in the direction of steepest descent. Page 12."
        files = {
            "file": ("machine_learning.pdf", io.BytesIO(content), "application/pdf")
        }

        upload_resp = client.post("/v1/rag/upload", files=files, headers=AUTH_HEADERS)
        assert upload_resp.status_code == 200
        data = upload_resp.json()
        assert data["filename"] == "machine_learning.pdf"
        assert data["chunks_indexed"] >= 1

        search_resp = client.post(
            "/v1/rag/search",
            json={"query": "Explain gradient descent", "top_k": 1},
            headers=AUTH_HEADERS,
        )
        assert search_resp.status_code == 200
        results = search_resp.json()
        assert len(results) == 1
        assert results[0]["doc_name"] == "machine_learning.pdf"
        assert "gradient descent" in results[0]["content"].lower()
        assert results[0]["score"] > 0.0
