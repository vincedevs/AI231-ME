from __future__ import annotations

import asyncio
import contextlib
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import APPLICATION_ROOT, WayneManorConfig, load_config
from .models import (
    BrightnessRequest,
    ColorRequest,
    MediaStatusRequest,
    MediaVolumeRequest,
    MutationResponse,
    PowerRequest,
    TemperatureRequest,
)
from .state import MutationOutcome, WayneManorApiError, WorldStore


def _error(code: str, message: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"error": {"code": code, "message": message, "details": details or {}}}


def _request_id(value: str | None) -> str | None:
    if value is None:
        return None
    request_id = value.strip()
    if not request_id or len(request_id) > 128 or not request_id.isprintable():
        raise WayneManorApiError(422, "invalid_request_id", "X-Request-ID is invalid.")
    return request_id


def _response(outcome: MutationOutcome, request_id: str | None) -> MutationResponse:
    return MutationResponse(
        changed=outcome.changed,
        request_id=request_id,
        message=outcome.message,
        state=outcome.state,
    )


def create_app(
    config: WayneManorConfig | None = None,
    *,
    frontend_directory: Path | None = None,
    ui_mode: str = "display",
) -> FastAPI:
    if ui_mode not in {"display", "controls"}:
        raise ValueError("ui_mode must be display or controls")
    settings = config or load_config()
    world = WorldStore(settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        del application
        yield
        await world.close()

    app = FastAPI(
        title="WayneManor API",
        description=(
            "Local virtual light, thermostat, telephone, and Spotify environment for Alfred."
        ),
        lifespan=lifespan,
    )
    app.state.world = world
    app.state.settings = settings
    app.state.ui_mode = ui_mode

    @app.exception_handler(WayneManorApiError)
    async def wayne_manor_error_handler(request: Request, error: WayneManorApiError) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=error.status_code,
            content=_error(error.code, error.message, error.details),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, error: RequestValidationError
    ) -> JSONResponse:
        del request
        details = {
            "fields": [
                {"path": ".".join(str(item) for item in row["loc"]), "message": row["msg"]}
                for row in error.errors()
            ]
        }
        return JSONResponse(
            status_code=422,
            content=_error("invalid_request", "The request contains invalid values.", details),
        )

    @app.middleware("http")
    async def limit_request_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length is not None:
            try:
                too_large = int(length) > settings.server.maximum_request_bytes
            except ValueError:
                return JSONResponse(
                    status_code=400,
                    content=_error("invalid_content_length", "Content-Length is invalid."),
                )
            if too_large:
                return JSONResponse(
                    status_code=413,
                    content=_error("request_too_large", "The request body is too large."),
                )
        if request.method in {"POST", "PUT", "PATCH"}:
            body = await request.body()
            if len(body) > settings.server.maximum_request_bytes:
                return JSONResponse(
                    status_code=413,
                    content=_error("request_too_large", "The request body is too large."),
                )
        return await call_next(request)

    @app.get("/api/v1/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok"}

    @app.get("/api/v1/ui-config")
    async def ui_config() -> dict[str, str]:
        return {"mode": ui_mode}

    @app.get("/api/v1/capabilities")
    async def capabilities() -> dict[str, Any]:
        return world.capabilities()

    @app.get("/api/v1/state")
    async def state() -> dict[str, Any]:
        return await world.snapshot()

    @app.get("/api/v1/media")
    async def media() -> dict[str, Any]:
        return await world.media()

    @app.get("/api/v1/light-groups/{group_id}")
    async def light_group(group_id: str) -> dict[str, Any]:
        return await world.light(group_id)

    @app.get("/api/v1/thermostats/{thermostat_id}")
    async def thermostat(thermostat_id: str) -> dict[str, Any]:
        return await world.thermostat(thermostat_id)

    @app.put("/api/v1/light-groups/{group_id}/power", response_model=MutationResponse)
    async def set_power(
        group_id: str,
        body: PowerRequest,
        x_request_id: str | None = Header(default=None),
    ) -> MutationResponse:
        request_id = _request_id(x_request_id)
        return _response(await world.set_light_power(group_id, body.on), request_id)

    @app.put("/api/v1/light-groups/{group_id}/brightness", response_model=MutationResponse)
    async def set_brightness(
        group_id: str,
        body: BrightnessRequest,
        x_request_id: str | None = Header(default=None),
    ) -> MutationResponse:
        request_id = _request_id(x_request_id)
        return _response(await world.set_light_brightness(group_id, body.percent), request_id)

    @app.put("/api/v1/light-groups/{group_id}/color", response_model=MutationResponse)
    async def set_color(
        group_id: str,
        body: ColorRequest,
        x_request_id: str | None = Header(default=None),
    ) -> MutationResponse:
        request_id = _request_id(x_request_id)
        return _response(await world.set_light_color(group_id, body.color), request_id)

    @app.put("/api/v1/thermostats/{thermostat_id}/setpoint", response_model=MutationResponse)
    async def set_temperature(
        thermostat_id: str,
        body: TemperatureRequest,
        x_request_id: str | None = Header(default=None),
    ) -> MutationResponse:
        request_id = _request_id(x_request_id)
        return _response(
            await world.set_temperature(thermostat_id, body.degrees, body.unit), request_id
        )

    @app.post("/api/v1/telephone/calls", response_model=MutationResponse)
    async def start_telephone_call(
        x_request_id: str | None = Header(default=None),
    ) -> MutationResponse:
        request_id = _request_id(x_request_id)
        call_id = request_id or uuid.uuid4().hex
        return _response(await world.start_telephone_call(call_id), request_id)

    @app.delete("/api/v1/telephone/calls/current", response_model=MutationResponse)
    async def stop_telephone_call(
        x_request_id: str | None = Header(default=None),
    ) -> MutationResponse:
        request_id = _request_id(x_request_id)
        return _response(await world.stop_telephone_call(), request_id)

    @app.post("/api/v1/media/status", response_model=MutationResponse)
    async def set_media_status(body: MediaStatusRequest) -> MutationResponse:
        return _response(await world.set_media_status(body.model_dump()), body.command_id)

    async def media_command(action: str, x_request_id: str | None) -> MutationResponse:
        request_id = _request_id(x_request_id) or uuid.uuid4().hex
        return _response(await world.execute_media_command(action, request_id), request_id)

    @app.post("/api/v1/media/play", response_model=MutationResponse)
    async def play_media(x_request_id: str | None = Header(default=None)) -> MutationResponse:
        return await media_command("play", x_request_id)

    @app.post("/api/v1/media/pause", response_model=MutationResponse)
    async def pause_media(x_request_id: str | None = Header(default=None)) -> MutationResponse:
        return await media_command("pause", x_request_id)

    @app.post("/api/v1/media/resume", response_model=MutationResponse)
    async def resume_media(x_request_id: str | None = Header(default=None)) -> MutationResponse:
        return await media_command("resume", x_request_id)

    @app.post("/api/v1/media/stop", response_model=MutationResponse)
    async def stop_media(x_request_id: str | None = Header(default=None)) -> MutationResponse:
        return await media_command("stop", x_request_id)

    @app.post("/api/v1/media/next", response_model=MutationResponse)
    async def next_media(x_request_id: str | None = Header(default=None)) -> MutationResponse:
        return await media_command("next", x_request_id)

    @app.put("/api/v1/media/volume", response_model=MutationResponse)
    async def set_media_volume(
        body: MediaVolumeRequest,
        x_request_id: str | None = Header(default=None),
    ) -> MutationResponse:
        request_id = _request_id(x_request_id) or uuid.uuid4().hex
        outcome = await world.execute_media_command(
            "set_volume", request_id, {"percent": body.percent}
        )
        return _response(outcome, request_id)

    @app.websocket("/api/v1/events")
    async def events(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = world.subscribe()
        disconnect = asyncio.create_task(websocket.receive())
        try:
            snapshot = await world.snapshot()
            await websocket.send_json({"type": "state.snapshot", "state": snapshot})
            while True:
                state_change = asyncio.create_task(queue.get())
                completed, _ = await asyncio.wait(
                    {state_change, disconnect}, return_when=asyncio.FIRST_COMPLETED
                )
                if disconnect in completed:
                    state_change.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await state_change
                    break
                await websocket.send_json(state_change.result())
        except (WebSocketDisconnect, RuntimeError):
            pass
        except asyncio.CancelledError:
            raise
        finally:
            disconnect.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await disconnect
            world.unsubscribe(queue)

    build_directory = frontend_directory or APPLICATION_ROOT / "frontend" / "dist"
    if build_directory.is_dir():
        app.mount("/", StaticFiles(directory=build_directory, html=True), name="frontend")
    else:

        @app.get("/")
        async def development_root() -> dict[str, str]:
            return {
                "application": "Wayne Manor",
                "api": "/docs",
                "frontend": "Run the Vite development server or build frontend/dist.",
            }

    return app


app = create_app()
