import uuid, json, gzip

from typing import Any
from fastapi import Request
from starlette.responses import Response, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import iterate_in_threadpool

from core.log_context import set_request_id


def _is_json_response(response: Response) -> bool:
    return response.headers.get("Content-type", "").split(";")[0].strip() == "application/json"

class RequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        
        # produce / propagate request_id
        request_id = request.headers.get("X-Trace-Id") or str(uuid.uuid4())
        set_request_id(request_id)
        request.state.trace_id = request_id
        
        # run downstream handlers
        response: Response = await call_next(request)
        response.headers["X-Trace-Id"] = request_id
        
        # inject into JSON error bodies
        if 400 <= response.status_code < 600 and _is_json_response(response=response):
            # consume the streaming body once
            body_bytes = b""
            async for chunk in response.body_iterator:
                body_bytes += chunk
            
            # decompress if needed
            if response.headers.get("content-encoding") == "gzip":
                try:
                    body_bytes = gzip.decompress(body_bytes)
                except OSError:
                    # bad gzip payload, handle gracefully
                    response.body_iterator = iterate_in_threadpool(iter[body_bytes])
                    return response
            
            try:
                body: Any = json.loads(body_bytes)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # not valid JSON after decompression, handler gracefully, leave untouched
                response.body_iterator = iterate_in_threadpool(iter[body_bytes])
                return response
            
            if "request_id" not in body:
                body["trace_id"] = request_id
                
            # rebuild identical response (status, headers) with new body
            response = JSONResponse(
                content=body,
                status_code=response.status_code,
                headers={
                    k:v
                    for k, v in response.headers.items()
                    if k.lower() not in {"content-encoding", "content-length"}
                }
            )
            
        return response