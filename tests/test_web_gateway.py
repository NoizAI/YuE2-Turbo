import httpx
from fastapi.testclient import TestClient

from yue2.web_gateway import create_app


def replace_upstream(app, handler):
    original = app.state.client
    app.state.client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler))
    return original


def test_gateway_does_not_supply_credentials_or_expose_paths():
    app = create_app("http://127.0.0.1:8015")
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            401, json={"detail": "Invalid bearer token"})

    with TestClient(app) as client:
        original = replace_upstream(app, handler)
        page = client.get("/")
        assert page.status_code == 200
        assert "访问密钥" in page.text and "vLLM 并行推理" in page.text
        assert "/v1/plans" not in page.text and "/artifacts/" not in page.text
        assert client.get("/service.env").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/v1/jobs").status_code == 401
        assert not seen[-1].headers.get("authorization")
        assert client.get(
            "/v1/jobs",
            headers={"Authorization": "Bearer test-only"},
        ).status_code == 401
        assert seen[-1].headers["authorization"] == "Bearer test-only"
        assert client.post("/v1/jobs", content="x" * 262145).status_code == 413
        forwarded = client.post(
            "/v1/covers",
            files={"audio": ("song.mp3", b"y" * (300 * 1024), "audio/mpeg")},
            data={"style": "pop", "lyrics": "la"},
        )
        assert forwarded.status_code == 401
        assert seen[-1].url.path == "/v1/covers"
        assert len(seen[-1].content) > 256 * 1024
        assert client.post("/v1/covers", content=b"z" * (40 * 1024 * 1024 + 1)).status_code == 413
        assert page.headers["x-frame-options"] == "DENY"
        app.state.client = original


def test_gateway_reports_runtime_validates_key_and_streams_artifacts(tmp_path):
    artifact = tmp_path / "audio.flac"
    artifact.write_bytes(b"fLaC")
    app = create_app("http://127.0.0.1:8015")

    def handler(request):
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"status": "ready"})
        if request.url.path.endswith("0" * 32):
            return httpx.Response(
                404 if request.headers.get("authorization") == "Bearer valid"
                else 401,
                json={"detail": "not found"},
            )
        if request.url.path.endswith("/audio"):
            return httpx.Response(
                206,
                content=artifact.read_bytes(),
                headers={
                    "content-type": "audio/flac",
                    "content-range": "bytes 0-3/4",
                },
            )
        raise AssertionError(request.url)

    with TestClient(app) as client:
        original = replace_upstream(app, handler)
        status = client.get("/console/status").json()
        assert status["available"]
        assert status["runtime"] == {
            "backend": "vLLM",
            "ar_concurrency": 4,
            "nar_batch_size": 2,
            "ar_nar_overlap": True,
        }
        assert client.get("/console/connect").status_code == 401
        assert client.get(
            "/console/connect",
            headers={"Authorization": "Bearer invalid"},
        ).status_code == 401
        assert client.get(
            "/console/connect",
            headers={"Authorization": "Bearer valid"},
        ).json() == {"connected": True}
        response = client.get(
            "/v1/jobs/" + "1" * 32 + "/audio",
            headers={"Authorization": "Bearer valid", "Range": "bytes=0-3"},
        )
        assert response.status_code == 206
        assert response.content == b"fLaC"
        assert response.headers["content-range"] == "bytes 0-3/4"
        app.state.client = original
