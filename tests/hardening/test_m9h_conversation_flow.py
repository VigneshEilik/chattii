from fastapi.testclient import TestClient

from engine.api.api import app

AUTH_HEADERS = {"Authorization": "Bearer chatii-admin-root-key"}


def test_multi_turn_name_retention():
    with TestClient(app) as client:
        # Create conversation
        c_resp = client.post(
            "/v1/conversations",
            json={"user_id": "usr-vignesh", "title": "Name Test"},
            headers=AUTH_HEADERS,
        )
        assert c_resp.status_code == 200
        conv_id = c_resp.json()["id"]

        # Turn 1: Introduce user
        client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={"role": "user", "content": "My name is Vignesh."},
            headers=AUTH_HEADERS,
        )
        client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={
                "role": "assistant",
                "content": "Hello Vignesh! How can I help you today?",
            },
            headers=AUTH_HEADERS,
        )

        # Turn 2: Query name
        client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={"role": "user", "content": "What is my name?"},
            headers=AUTH_HEADERS,
        )
        client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={"role": "assistant", "content": "Your name is Vignesh."},
            headers=AUTH_HEADERS,
        )

        # Fetch history
        get_resp = client.get(f"/v1/conversations/{conv_id}", headers=AUTH_HEADERS)
        assert get_resp.status_code == 200
        msgs = get_resp.json()["messages"]
        assert len(msgs) == 4
        assert "Vignesh" in msgs[0]["content"]
        assert "Vignesh" in msgs[3]["content"]


def test_multi_step_quadratic_reasoning_flow():
    with TestClient(app) as client:
        c_resp = client.post(
            "/v1/conversations",
            json={"user_id": "usr-math", "title": "Math Reasoning"},
            headers=AUTH_HEADERS,
        )
        conv_id = c_resp.json()["id"]

        # Step 1: Quadratic formula
        client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={"role": "user", "content": "What is the quadratic formula?"},
            headers=AUTH_HEADERS,
        )
        client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={
                "role": "assistant",
                "content": "The quadratic formula is x = (-b +- sqrt(b^2 - 4ac)) / (2a).",
            },
            headers=AUTH_HEADERS,
        )

        # Step 2: Explain components
        client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={"role": "user", "content": "Explain each part of the formula."},
            headers=AUTH_HEADERS,
        )
        client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={
                "role": "assistant",
                "content": "a is x^2 coefficient, b is x coefficient, c is constant, b^2-4ac is discriminant.",
            },
            headers=AUTH_HEADERS,
        )

        # Step 3: Example
        client.post(
            f"/v1/conversations/{conv_id}/messages",
            json={"role": "user", "content": "Give me an example."},
            headers=AUTH_HEADERS,
        )

        get_resp = client.get(f"/v1/conversations/{conv_id}", headers=AUTH_HEADERS)
        assert len(get_resp.json()["messages"]) == 5
