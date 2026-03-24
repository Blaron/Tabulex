from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Iterable, Literal, Sequence

from .models import ExtractedTable, NormalizeConfig, NormalizedTable, ValueMode
from .normalize import normalize_table

MixedMode = Literal["auto", "python", "json", "text"]
JsonOrient = Literal["records", "rows"]


def _ensure_normalized(
    table: ExtractedTable | NormalizedTable,
    config: NormalizeConfig | None,
    *,
    value_mode: ValueMode,
) -> NormalizedTable:
    if isinstance(table, NormalizedTable):
        return table
    return normalize_table(table, config, value_mode=value_mode)


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _safe_json_dumps(value: Any, *, ensure_ascii: bool = False, indent: int | None = None) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, default=_json_default)


def _is_scalar_value(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _coerce_value_for_backend(value: Any, mixed_mode: MixedMode, backend: str) -> Any:
    if value is None:
        return None
    if mixed_mode == "python":
        return value
    if mixed_mode == "text":
        if isinstance(value, dict) and value.get("text") is not None:
            return value.get("text")
        return value if isinstance(value, str) else str(value)
    if mixed_mode == "json":
        if isinstance(value, (dict, list, tuple)) or not _is_scalar_value(value):
            return _safe_json_dumps(value)
        return value
    if mixed_mode != "auto":
        raise ValueError(f"unsupported mixed mode: {mixed_mode}")

    if backend == "polars":
        if isinstance(value, (dict, list, tuple)):
            return _safe_json_dumps(value)
        if not _is_scalar_value(value):
            return str(value)
    return value


def _apply_header_rows(
    rows: list[list[Any]],
    columns: Sequence[str] | None,
    *,
    header_rows: Sequence[int] | None,
    drop_header_rows: bool,
) -> tuple[list[list[Any]], list[str] | None]:
    if drop_header_rows and header_rows:
        header_indexes = set(header_rows)
        rows = [row for index, row in enumerate(rows) if index not in header_indexes]
    return rows, None if columns is None else list(columns)


def _prepare_rows(
    table: ExtractedTable | NormalizedTable,
    config: NormalizeConfig | None,
    *,
    value_mode: ValueMode,
    mixed_mode: MixedMode,
    backend: str,
    header_rows: Sequence[int] | None,
    drop_header_rows: bool,
) -> tuple[list[list[Any]], list[str] | None]:
    normalized = _ensure_normalized(table, config, value_mode=value_mode)
    rows = [
        [_coerce_value_for_backend(value, mixed_mode, backend) for value in row]
        for row in normalized.matrix
    ]
    if header_rows is not None:
        effective_headers = tuple(header_rows)
    else:
        stored_headers = normalized.metadata.get("header_rows")
        if stored_headers is not None:
            effective_headers = tuple(stored_headers)
        elif normalized.source is not None and normalized.source.header_rows is not None:
            effective_headers = tuple(normalized.source.header_rows)
        else:
            effective_headers = ()
    rows, columns = _apply_header_rows(
        rows,
        normalized.column_names,
        header_rows=effective_headers,
        drop_header_rows=drop_header_rows,
    )
    return rows, columns


def _generated_columns(rows: Sequence[Sequence[Any]], columns: Sequence[str] | None) -> list[str]:
    if columns is not None:
        return list(columns)
    width = max((len(row) for row in rows), default=0)
    return [f"column_{index}" for index in range(width)]


def _rows_to_records(rows: Sequence[Sequence[Any]], columns: Sequence[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        records.append(
            {
                column: row[index] if index < len(row) else None
                for index, column in enumerate(columns)
            }
        )
    return records


def table_to_pandas(
    table: ExtractedTable | NormalizedTable,
    config: NormalizeConfig | None = None,
    *,
    value_mode: ValueMode = "native",
    mixed_mode: MixedMode = "python",
    header_rows: Sequence[int] | None = None,
    drop_header_rows: bool = False,
):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("pandas is required to build a pandas.DataFrame") from exc

    rows, columns = _prepare_rows(
        table,
        config,
        value_mode=value_mode,
        mixed_mode=mixed_mode,
        backend="pandas",
        header_rows=header_rows,
        drop_header_rows=drop_header_rows,
    )
    if columns is None:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows, columns=columns)


def table_to_polars(
    table: ExtractedTable | NormalizedTable,
    config: NormalizeConfig | None = None,
    *,
    value_mode: ValueMode = "native",
    mixed_mode: MixedMode = "json",
    header_rows: Sequence[int] | None = None,
    drop_header_rows: bool = False,
):
    try:
        import polars as pl
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("polars is required to build a polars.DataFrame") from exc

    rows, columns = _prepare_rows(
        table,
        config,
        value_mode=value_mode,
        mixed_mode=mixed_mode,
        backend="polars",
        header_rows=header_rows,
        drop_header_rows=drop_header_rows,
    )
    if columns is None:
        return pl.DataFrame(rows)
    if not rows:
        return pl.DataFrame({column: [] for column in columns})
    return pl.DataFrame(rows, schema=columns, orient="row")


def table_to_json(
    table: ExtractedTable | NormalizedTable,
    config: NormalizeConfig | None = None,
    *,
    value_mode: ValueMode = "native",
    mixed_mode: MixedMode = "python",
    header_rows: Sequence[int] | None = None,
    drop_header_rows: bool = False,
    orient: JsonOrient = "records",
    as_string: bool = False,
    ensure_ascii: bool = False,
    indent: int | None = 2,
):
    rows, columns = _prepare_rows(
        table,
        config,
        value_mode=value_mode,
        mixed_mode=mixed_mode,
        backend="json",
        header_rows=header_rows,
        drop_header_rows=drop_header_rows,
    )
    resolved_columns = _generated_columns(rows, columns)

    if orient == "records":
        data: Any = _rows_to_records(rows, resolved_columns)
    elif orient == "rows":
        data = rows
    else:
        raise ValueError(f"unsupported JSON orient: {orient}")

    source = table.source if isinstance(table, NormalizedTable) else table
    payload = {
        "page_number": source.page_number if source is not None else None,
        "bbox": list(source.bbox) if source is not None and source.bbox is not None else None,
        "shape": {
            "rows": len(rows),
            "cols": len(resolved_columns),
        },
        "columns": resolved_columns,
        "orient": orient,
        "data": data,
        "metadata": dict(source.metadata) if source is not None else {},
    }

    if source is not None and source.header_rows is not None:
        payload["header_rows"] = list(source.header_rows)

    if as_string:
        return _safe_json_dumps(payload, ensure_ascii=ensure_ascii, indent=indent)
    return payload


def tables_to_pandas(
    tables: Iterable[ExtractedTable | NormalizedTable],
    config: NormalizeConfig | None = None,
    **kwargs: Any,
):
    return [table_to_pandas(table, config, **kwargs) for table in tables]


def tables_to_polars(
    tables: Iterable[ExtractedTable | NormalizedTable],
    config: NormalizeConfig | None = None,
    **kwargs: Any,
):
    return [table_to_polars(table, config, **kwargs) for table in tables]


def tables_to_json(
    tables: Iterable[ExtractedTable | NormalizedTable],
    config: NormalizeConfig | None = None,
    **kwargs: Any,
):
    return [table_to_json(table, config, **kwargs) for table in tables]


__all__ = [
    "JsonOrient",
    "table_to_json",
    "table_to_pandas",
    "table_to_polars",
    "tables_to_json",
    "tables_to_pandas",
    "tables_to_polars",
]

