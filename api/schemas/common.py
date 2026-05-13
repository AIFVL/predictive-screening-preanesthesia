from __future__ import annotations

from typing import Any, Generic, TypeVar
import uuid

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str
    message: str
    field: str | None = None


class ResponseMeta(BaseModel):
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    model_id: str | None = None


class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: T | None = None
    errors: list[ErrorDetail] | None = None
    meta: ResponseMeta = Field(default_factory=ResponseMeta)


def success_response(data: Any, model_id: str | None = None) -> dict:
    return ApiResponse(
        success=True,
        data=data,
        errors=None,
        meta=ResponseMeta(model_id=model_id),
    ).model_dump(mode="json")


def error_response(
    code: str,
    message: str,
    field: str | None = None,
    model_id: str | None = None,
) -> dict:
    return ApiResponse(
        success=False,
        data=None,
        errors=[ErrorDetail(code=code, message=message, field=field)],
        meta=ResponseMeta(model_id=model_id),
    ).model_dump(mode="json")
