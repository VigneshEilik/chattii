import io
import uuid

from fastapi.testclient import TestClient

from engine.api.api import app

AUTH_HEADERS = {"Authorization": "Bearer chatii-admin-root-key"}


def test_e2e_auth_flow():
    with TestClient(app) as client:
        unique_email = f"e2e_user_{uuid.uuid4().hex[:6]}@chatii.ai"
        # Register user (public route)
        reg_resp = client.post(
            "/v1/auth/register",
            json={
                "email": unique_email,
                "username": "e2e_user",
                "password": "e2e-password-123",
            },
        )
        assert reg_resp.status_code == 200
        reg_data = reg_resp.json()
        assert "access_token" in reg_data
        token = reg_data["access_token"]

        # Fetch profile (user session token)
        me_resp = client.get(
            "/v1/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert me_resp.status_code == 200
        assert me_resp.json()["email"] == unique_email


def test_e2e_conversations_flow():
    with TestClient(app) as client:
        # Create conversation
        create_resp = client.post(
            "/v1/conversations",
            json={"user_id": "usr-e2e", "title": "E2E Conv"},
            headers=AUTH_HEADERS,
        )
        assert create_resp.status_code == 200
        conv_id = create_resp.json()["id"]

        # Add message
        msg_resp = client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={"role": "user", "content": "Hello E2E"},
            headers=AUTH_HEADERS,
        )
        assert msg_resp.status_code == 200

        # Get conversation details
        get_resp = client.get(f"/v1/conversations/{conv_id}", headers=AUTH_HEADERS)
        assert get_resp.status_code == 200
        assert len(get_resp.json()["messages"]) == 1


def test_e2e_api_keys_flow():
    with TestClient(app) as client:
        # Create API key
        key_resp = client.post(
            "/v1/keys",
            json={"user_id": "usr-e2e", "name": "E2E Key"},
            headers=AUTH_HEADERS,
        )
        assert key_resp.status_code == 200
        key_id = key_resp.json()["id"]
        assert key_resp.json()["raw_key"].startswith("sk-chatii-")

        # List keys
        list_resp = client.get("/v1/keys?user_id=usr-e2e", headers=AUTH_HEADERS)
        assert list_resp.status_code == 200
        assert len(list_resp.json()) >= 1

        # Revoke key
        del_resp = client.delete(
            f"/v1/keys/{key_id}?user_id=usr-e2e", headers=AUTH_HEADERS
        )
        assert del_resp.status_code == 200


def test_e2e_rag_flow():
    with TestClient(app) as client:
        # Upload doc
        file_content = b"Chatii platform uses modular architecture for high performance LLM inference."
        files = {"file": ("test_doc.txt", io.BytesIO(file_content), "text/plain")}
        upload_resp = client.post("/v1/rag/upload", files=files, headers=AUTH_HEADERS)
        assert upload_resp.status_code == 200
        assert upload_resp.json()["status"] == "indexed"

        # Search doc
        search_resp = client.post(
            "/v1/rag/search",
            json={"query": "inference", "top_k": 1},
            headers=AUTH_HEADERS,
        )
        assert search_resp.status_code == 200
        assert len(search_resp.json()) == 1


def test_e2e_admin_dashboard_flow():
    with TestClient(app) as client:
        dash_resp = client.get("/v1/admin/dashboard", headers=AUTH_HEADERS)
        assert dash_resp.status_code == 200
        data = dash_resp.json()
        assert "system_health" in data
        assert "performance_metrics" in data
