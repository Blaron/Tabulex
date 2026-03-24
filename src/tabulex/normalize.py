from __future__ import annotations

from typing import Any, Iterable, Sequence

from .models import CollisionPolicy, ExtractedTable, NormalizeConfig, NormalizedTable, TableCell, ValueMode

_EMPTY = object()
_UNMERGEABLE = object()


def _coerce_cells(table: ExtractedTable) -> list[TableCell]:
    return list(table.iter_cells())


def _infer_dimensions(
    table: ExtractedTable,
    cells: Sequence[TableCell],
    config: NormalizeConfig,
) -> tuple[int, int]:
    rows = max(table.n_rows, config.row_count or 0, table.max_covered_row())
    cols = max(table.n_cols, config.col_count or 0, table.max_covered_col())
    if rows == 0:
        rows = max((cell.row + cell.rowspan for cell in cells), default=0)
    if cols == 0:
        cols = max((cell.col + cell.colspan for cell in cells), default=0)
    return rows, cols


def _append_unique(target: list[Any], values: Iterable[Any]) -> None:
    for value in values:
        if any(existing == value for existing in target):
            continue
        target.append(value)


def _is_cell_payload(value: Any) -> bool:
    return isinstance(value, dict) and {"row", "col", "rowspan", "colspan"}.issubset(value)


def _payload_text_lines(value: Any) -> list[str]:
    if _is_cell_payload(value):
        lines = value.get("text_lines") or []
        if lines:
            return [str(line) for line in lines if line not in (None, "")]
        text = value.get("text")
        if text not in (None, ""):
            return [str(text)]
        return []
    if isinstance(value, str):
        return [part for part in value.splitlines() if part]
    return []


def _payload_images(value: Any) -> list[str]:
    if not _is_cell_payload(value):
        return []
    return [str(item) for item in value.get("images_base64") or [] if item not in (None, "")]


def _merge_bboxes(left: Any, right: Any) -> list[float] | None:
    if left is None and right is None:
        return None
    if left is None:
        return list(right)
    if right is None:
        return list(left)
    return [
        min(float(left[0]), float(right[0])),
        min(float(left[1]), float(right[1])),
        max(float(left[2]), float(right[2])),
        max(float(left[3]), float(right[3])),
    ]


def _merge_cell_payloads(existing: dict[str, Any], incoming: dict[str, Any], *, joiner: str) -> dict[str, Any]:
    lines: list[str] = []
    _append_unique(lines, _payload_text_lines(existing))
    _append_unique(lines, _payload_text_lines(incoming))

    images: list[str] = []
    _append_unique(images, _payload_images(existing))
    _append_unique(images, _payload_images(incoming))

    metadata: dict[str, Any] = {}
    if isinstance(existing.get("metadata"), dict):
        metadata.update(existing["metadata"])
    if isinstance(incoming.get("metadata"), dict):
        metadata.update(incoming["metadata"])

    merged: dict[str, Any] = {
        "text": joiner.join(lines) if lines else None,
        "text_lines": lines,
        "images_base64": images,
        "row": existing.get("row", incoming.get("row")),
        "col": existing.get("col", incoming.get("col")),
        "rowspan": max(int(existing.get("rowspan", 1)), int(incoming.get("rowspan", 1))),
        "colspan": max(int(existing.get("colspan", 1)), int(incoming.get("colspan", 1))),
    }

    bbox = _merge_bboxes(existing.get("bbox"), incoming.get("bbox"))
    if bbox is not None:
        merged["bbox"] = bbox
    if metadata:
        merged["metadata"] = metadata

    return merged


def _coerce_to_payload(value: Any, *, joiner: str) -> dict[str, Any] | None:
    if _is_cell_payload(value):
        return dict(value)
    if isinstance(value, str):
        lines = [part for part in value.splitlines() if part]
        return {
            "text": joiner.join(lines) if lines else value,
            "text_lines": lines or [value],
            "images_base64": [],
            "row": None,
            "col": None,
            "rowspan": 1,
            "colspan": 1,
        }
    return None


def _auto_merge_compatible_values(existing: Any, incoming: Any, *, joiner: str) -> Any:
    if existing == incoming:
        return existing
    if existing in (_EMPTY, None):
        return incoming
    if incoming in (_EMPTY, None):
        return existing

    if isinstance(existing, str) and isinstance(incoming, str):
        parts: list[str] = []
        _append_unique(parts, [part for part in existing.splitlines() if part] or [existing])
        _append_unique(parts, [part for part in incoming.splitlines() if part] or [incoming])
        return joiner.join(parts)

    existing_payload = _coerce_to_payload(existing, joiner=joiner)
    incoming_payload = _coerce_to_payload(incoming, joiner=joiner)
    if existing_payload is not None and incoming_payload is not None:
        return _merge_cell_payloads(existing_payload, incoming_payload, joiner=joiner)

    if isinstance(existing, list) or isinstance(incoming, list):
        merged: list[Any] = []
        _append_unique(merged, existing if isinstance(existing, list) else [existing])
        _append_unique(merged, incoming if isinstance(incoming, list) else [incoming])
        return merged

    return _UNMERGEABLE


def _merge_values(existing: Any, incoming: Any, policy: CollisionPolicy, *, joiner: str) -> Any:
    auto_merged = _auto_merge_compatible_values(existing, incoming, joiner=joiner)
    if auto_merged is not _UNMERGEABLE:
        return auto_merged

    if policy == "first":
        return existing
    if policy == "last":
        return incoming
    if policy == "list":
        values: list[Any] = []
        if existing not in (_EMPTY, None):
            values.extend(existing if isinstance(existing, list) else [existing])
        if incoming not in (_EMPTY, None):
            values.extend(incoming if isinstance(incoming, list) else [incoming])
        return values
    if policy == "combine":
        if existing in (_EMPTY, None):
            return incoming
        if incoming in (_EMPTY, None):
            return existing
        if isinstance(existing, str) and isinstance(incoming, str):
            return joiner.join([existing, incoming])
        if isinstance(existing, list):
            return existing + ([incoming] if not isinstance(incoming, list) else incoming)
        if isinstance(incoming, list):
            return [existing] + incoming
        return [existing, incoming]
    if policy == "raise":
        raise ValueError("collision detected while normalizing table")
    raise ValueError(f"unsupported collision policy: {policy}")


def _build_column_names(
    matrix: Sequence[Sequence[Any]],
    header_rows: Sequence[int],
    *,
    separator: str,
) -> tuple[str, ...] | None:
    if not header_rows or not matrix:
        return None
    width = max((len(row) for row in matrix), default=0)
    header_map: list[list[str]] = [[] for _ in range(width)]
    for row_index in header_rows:
        if row_index < 0 or row_index >= len(matrix):
            continue
        row = matrix[row_index]
        for col_index in range(width):
            if col_index >= len(row):
                continue
            value = row[col_index]
            if value in (None, ""):
                continue
            if isinstance(value, dict) and value.get("text") is not None:
                header_map[col_index].append(str(value["text"]))
            else:
                header_map[col_index].append(str(value))
    names = [separator.join(parts).strip() for parts in header_map]
    if not any(names):
        return None
    return tuple(name if name else f"column_{index}" for index, name in enumerate(names))


def normalize_table(
    table: ExtractedTable,
    config: NormalizeConfig | None = None,
    *,
    value_mode: ValueMode = "native",
) -> NormalizedTable:
    config = NormalizeConfig() if config is None else config
    if config.span_mode not in {"preserve", "expand"}:
        raise ValueError(f"unsupported span mode: {config.span_mode}")

    cells = _coerce_cells(table)
    rows, cols = _infer_dimensions(table, cells, config)
    matrix: list[list[Any]] = [[_EMPTY for _ in range(cols)] for _ in range(rows)]

    for cell in cells:
        payload = cell.to_value(value_mode, line_joiner=config.line_joiner)
        for row_index in range(cell.row, cell.row + cell.rowspan):
            for col_index in range(cell.col, cell.col + cell.colspan):
                if row_index >= rows or col_index >= cols:
                    raise ValueError("cell span exceeds normalized table bounds")
                if config.span_mode == "preserve" and (row_index != cell.row or col_index != cell.col):
                    continue
                current = matrix[row_index][col_index]
                if current is _EMPTY:
                    matrix[row_index][col_index] = payload
                else:
                    matrix[row_index][col_index] = _merge_values(
                        current,
                        payload,
                        config.collision_policy,
                        joiner=config.line_joiner,
                    )

    normalized_rows: list[tuple[Any, ...]] = []
    for row in matrix:
        normalized_rows.append(tuple(config.filler if value is _EMPTY else value for value in row))

    header_rows = config.header_rows if config.header_rows is not None else table.header_rows
    column_names = _build_column_names(normalized_rows, header_rows, separator=config.header_separator)

    metadata = dict(table.metadata)
    if header_rows:
        metadata["header_rows"] = tuple(header_rows)

    return NormalizedTable(
        matrix=tuple(normalized_rows),
        source=table,
        config=config,
        value_mode=value_mode,
        column_names=column_names,
        metadata=metadata,
    )


def normalize_tables(
    tables: Iterable[ExtractedTable],
    config: NormalizeConfig | None = None,
    *,
    value_mode: ValueMode = "native",
) -> list[NormalizedTable]:
    return [normalize_table(table, config, value_mode=value_mode) for table in tables]


def matrix_from_table(
    table: ExtractedTable,
    config: NormalizeConfig | None = None,
    *,
    value_mode: ValueMode = "native",
) -> list[list[Any]]:
    return normalize_table(table, config, value_mode=value_mode).as_rows()


__all__ = [
    "matrix_from_table",
    "normalize_table",
    "normalize_tables",
]
