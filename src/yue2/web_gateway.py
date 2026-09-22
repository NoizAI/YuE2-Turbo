"""Public same-origin gateway for the private YuE2 GPU service.

The browser supplies its own bearer credential. The gateway never embeds,
stores, logs, or injects one.
"""
from contextlib import asynccontextmanager
import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask


FORWARDED_REQUEST_HEADERS = (
    "authorization", "content-type", "idempotency-key", "range",
)
FORWARDED_RESPONSE_HEADERS = {
    "content-type", "content-disposition", "content-length", "content-range",
    "accept-ranges", "retry-after", "www-authenticate",
}
AUTH_PROBE_JOB = "0" * 32
JSON_BODY_LIMIT = 256 * 1024
COVER_BODY_LIMIT = 40 * 1024 * 1024


def create_app(upstream=None):
    upstream = (
        upstream or os.environ.get("YUE2_UPSTREAM", "http://127.0.0.1:8015")
    ).rstrip("/")

    @asynccontextmanager
    async def lifespan(app):
        timeout = httpx.Timeout(30, connect=3)
        async with httpx.AsyncClient(
                timeout=timeout, follow_redirects=False) as client:
            app.state.client = client
            yield

    app = FastAPI(
        title="YuE2 Studio gateway", docs_url=None, redoc_url=None,
        openapi_url=None, lifespan=lifespan,
    )

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = (
            "camera=(), microphone=(), geolocation=(), payment=()"
        )
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; media-src 'self' blob:; "
            "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; "
            "base-uri 'none'; form-action 'self'"
        )
        return response

    @app.get("/")
    async def page():
        return FileResponse(
            Path(__file__).with_name("studio.html"), media_type="text/html")

    @app.get("/console/status")
    async def console_status():
        maintenance = Path(os.environ.get(
            "YUE2_MAINTENANCE_FILE",
            "/opt/noiz-yue/web-maintenance.txt",
        ))
        if maintenance.exists():
            return {
                "available": False,
                "message": maintenance.read_text()[:500],
            }
        try:
            response = await app.state.client.get(upstream + "/health/ready")
            available = response.status_code == 200
            return {
                "available": available,
                "message": (
                    "模型已就绪，可并行处理创作任务"
                    if available else "模型正在加载，请稍后重试"
                ),
                "runtime": {
                    "backend": "vLLM",
                    "ar_concurrency": 4,
                    "nar_batch_size": 2,
                    "ar_nar_overlap": True,
                },
            }
        except httpx.HTTPError:
            return {
                "available": False,
                "message": "模型暂时离线，页面仍可使用，恢复后可提交任务",
            }

    @app.get("/console/connect")
    async def console_connect(request: Request):
        authorization = request.headers.get("authorization")
        if not authorization:
            return JSONResponse(
                {"detail": "Missing bearer token"}, 401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        try:
            response = await app.state.client.get(
                f"{upstream}/v1/jobs/{AUTH_PROBE_JOB}",
                headers={"authorization": authorization},
            )
        except httpx.HTTPError:
            return JSONResponse(
                {"detail": "GPU 服务暂时不可用"}, 503,
                headers={"Retry-After": "10"},
            )
        if response.status_code == 404:
            return {"connected": True}
        if response.status_code == 401:
            return JSONResponse(
                {"detail": "访问密钥无效"}, 401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return JSONResponse(
            {"detail": "GPU 服务尚未就绪"},
            503,
            headers={"Retry-After": "10"},
        )

    @app.api_route("/{path:path}", methods=["GET", "POST"])
    async def proxy(path: str, request: Request):
        if not (path.startswith("v1/") or path in {
                "health/live", "health/ready"}):
            return JSONResponse({"detail": "Not found"}, 404)
        if any(part in {".", ".."} for part in path.split("/")):
            return JSONResponse({"detail": "Invalid path"}, 400)
        cover = request.method == "POST" and path == "v1/covers"
        limit = COVER_BODY_LIMIT if cover else JSON_BODY_LIMIT
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                detail = "Request exceeds 40 MiB" if cover else "Request exceeds 256 KiB"
                return JSONResponse({"detail": detail}, 413)
        outgoing_headers = {
            key: request.headers[key] for key in FORWARDED_REQUEST_HEADERS
            if key in request.headers
        }
        outgoing_headers["accept-encoding"] = "identity"
        try:
            outgoing = app.state.client.build_request(
                request.method,
                upstream + "/" + path,
                params=request.query_params,
                headers=outgoing_headers,
                content=bytes(body),
            )
            response = await app.state.client.send(outgoing, stream=True)
        except httpx.HTTPError:
            return JSONResponse(
                {"detail": "GPU 服务暂时不可用，请稍后重试"},
                503,
                headers={"Retry-After": "10"},
            )
        selected = {
            key: value for key, value in response.headers.items()
            if key in FORWARDED_RESPONSE_HEADERS
        }
        if response.is_stream_consumed:
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=selected,
            )
        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers=selected,
            background=BackgroundTask(response.aclose),
        )

    return app


def main():
    import uvicorn

    uvicorn.run(
        create_app(),
        host="0.0.0.0",
        port=int(os.environ.get("YUE2_WEB_PORT", "8016")),
        access_log=False,
    )


if __name__ == "__main__":
    main()
