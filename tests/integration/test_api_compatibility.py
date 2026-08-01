from fastapi.testclient import TestClient

from engine.api.api import app


def test_openai_api_compatibility():
    with TestClient(app) as client:
        headers = {"Authorization": "Bearer chatii-admin-root-key"}

        # Models endpoint compatibility
        resp_models = client.get("/v1/models", headers=headers)
        assert resp_models.status_code == 200
        models_data = resp_models.json()
        assert "data" in models_data
        assert any(m["id"] == "chatii-v1" for m in models_data["data"])

        # Chat completions compatibility
        payload_chat = {
            "model": "chatii-v1",
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "temperature": 0.8,
            "max_tokens": 16,
        }
        resp_chat = client.post(
            "/v1/chat/completions", json=payload_chat, headers=headers
        )
        assert resp_chat.status_code == 200
        chat_json = resp_chat.json()
        assert "id" in chat_json
        assert chat_json["choices"][0]["message"]["content"] is not None

        # Embeddings compatibility
        payload_emb = {"model": "chatii-v1", "input": "text to embed"}
        resp_emb = client.post("/v1/embeddings", json=payload_emb, headers=headers)
        assert resp_emb.status_code == 200
        emb_json = resp_emb.json()
        assert "data" in emb_json
        assert len(emb_json["data"][0]["embedding"]) > 0
