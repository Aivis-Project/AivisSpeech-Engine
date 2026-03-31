"""OpenAPI schema の設定"""

from typing import Any

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute
from pydantic import BaseModel

from voicevox_engine.library.model import BaseLibraryInfo, VvlibManifest


def simplify_operation_ids(app: FastAPI) -> FastAPI:
    """operation ID を簡略化してAPIクライアントで生成される関数名をシンプルにする。"""
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name

    return app


def configure_openapi_schema(app: FastAPI, manage_library: bool | None) -> FastAPI:
    """自動生成された OpenAPI schema へカスタム属性を追加する。"""

    # BaseLibraryInfo/VvlibManifestモデルはAPIとして表には出ないが、エディタ側で利用したいので、手動で追加する
    # ref: https://fastapi.tiangolo.com/advanced/extending-openapi/#modify-the-openapi-schema
    def custom_openapi() -> Any:
        if app.openapi_schema:
            return app.openapi_schema
        openapi_schema = get_openapi(
            title=app.title,
            version=app.version,
            openapi_version=app.openapi_version,
            summary=app.summary,
            description=app.description,
            terms_of_service=app.terms_of_service,
            contact=app.contact,
            license_info=app.license_info,
            routes=app.routes,
            webhooks=app.webhooks.routes,
            tags=app.openapi_tags,
            servers=app.servers,
            separate_input_output_schemas=app.separate_input_output_schemas,
        )
        if manage_library:
            additional_models: list[type[BaseModel]] = [
                BaseLibraryInfo,
                VvlibManifest,
            ]
            for model in additional_models:
                # ref_templateを指定しない場合、definitionsを参照してしまうので、手動で指定する
                schema = model.model_json_schema(
                    ref_template="#/components/schemas/{model}"
                )
                # definitionsは既存のモデルを重複して定義するため、不要なので削除
                if "$defs" in schema:
                    del schema["$defs"]
                openapi_schema["components"]["schemas"][schema["title"]] = schema

        # FastAPI 0.129.0 以降のバージョンでは、UploadFile が OpenAPI 3.1 の
        # contentMediaType 表現で出力されると、openapi-generator では
        # ファイルアップロード用フォームとして認識できなくなるため、従来の表現に補正する
        _restore_binary_format_for_compatibility(openapi_schema)

        app.openapi_schema = openapi_schema
        return openapi_schema

    app.openapi = custom_openapi  # type: ignore[method-assign]

    return app


def _restore_binary_format_for_compatibility(openapi_schema: dict[str, Any]) -> None:
    """
    OpenAPI schema 内のバイナリ文字列定義を `format: binary` へ補正する。

     FastAPI 0.129.0 以降のバージョンでは、`UploadFile` を含む schema が
    `contentMediaType: application/octet-stream` として出力される場合があるが、
    openapi-generator はこれをファイルアップロードと認識できず、`string` として扱ってしまう。
    既存の openapi-generator でのクライアントコード生成との互換性を保つため、
    `type: string` かつ `contentMediaType: application/octet-stream` の定義を従来の `format: binary` 表現へ戻す。

    Args:
        openapi_schema (dict[str, Any]): FastAPI が生成した OpenAPI schema
    """

    def visit_schema_node(node: Any) -> None:
        """
        OpenAPI schema ノードを再帰的に走査し、互換補正を適用する。

        Args:
            node (Any): 走査対象の schema ノード
        """

        if isinstance(node, dict):
            if (
                node.get("type") == "string"
                and node.get("contentMediaType") == "application/octet-stream"
            ):
                del node["contentMediaType"]
                node["format"] = "binary"

            for value in node.values():
                visit_schema_node(value)
        elif isinstance(node, list):
            for value in node:
                visit_schema_node(value)

    visit_schema_node(openapi_schema)
