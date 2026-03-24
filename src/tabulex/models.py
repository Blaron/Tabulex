from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Literal, Mapping, Sequence

BoundingBox = tuple[float, float, float, float]
ValueMode = Literal["native", "text", "lines", "mixed", "json", "images"]
CollisionPolicy = Literal["raise", "first", "last", "combine", "list"]
SpanMode = Literal["preserve", "expand"]


def _coerce_bbox(bbox: Sequence[float] | None) -> BoundingBox | None:
    if bbox is None:
        return None
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four values: (x0, y0, x1, y1)")
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _coerce_strings(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(value) for value in values)


@dataclass(slots=True)
class TableCell:
    row: int
    col: int
    bbox: BoundingBox | Sequence[float] | None = None
    text: str | None = None
    text_lines: Sequence[str] = field(default_factory=tuple)
    images_base64: Sequence[str] = field(default_factory=tuple)
    rowspan: int = 1
    colspan: int = 1
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.row = int(self.row)
        self.col = int(self.col)
        self.rowspan = int(self.rowspan)
        self.colspan = int(self.colspan)
        if self.row < 0 or self.col < 0:
            raise ValueError("row and col must be non-negative")
        if self.rowspan < 1 or self.colspan < 1:
            raise ValueError("rowspan and colspan must be greater than zero")
        self.bbox = _coerce_bbox(self.bbox)
        self.text_lines = _coerce_strings(self.text_lines)
        self.images_base64 = _coerce_strings(self.images_base64)
        self.metadata = dict(self.metadata)

    @property
    def has_content(self) -> bool:
        return self.text is not None or bool(self.text_lines) or bool(self.images_base64)

    @property
    def has_text(self) -> bool:
        return bool(self.text is not None and self.text != "") or bool(self.text_lines)

    @property
    def has_images(self) -> bool:
        return bool(self.images_base64)

    @property
    def is_empty(self) -> bool:
        return not self.has_content

    def copy_with(self, **changes: Any) -> TableCell:
        return replace(self, **changes)

    def to_value(self, mode: ValueMode = "native", *, line_joiner: str = "\n") -> Any:
        text = self.text
        if text is None and self.text_lines:
            text = line_joiner.join(self.text_lines)

        if mode == "native":
            if not self.has_content:
                return None
            if self.has_images:
                payload: dict[str, Any] = {
                    "text": text,
                    "text_lines": list(self.text_lines),
                    "images_base64": list(self.images_base64),
                    "row": self.row,
                    "col": self.col,
                    "rowspan": self.rowspan,
                    "colspan": self.colspan,
                }
                if self.bbox is not None:
                    payload["bbox"] = list(self.bbox)
                if self.metadata:
                    payload["metadata"] = dict(self.metadata)
                return payload
            if self.text_lines:
                return line_joiner.join(self.text_lines)
            return text

        if mode == "text":
            return text

        if mode == "lines":
            if self.text_lines:
                return list(self.text_lines)
            if text is None:
                return []
            return text.splitlines() or [text]

        if mode in {"mixed", "json"}:
            payload = {
                "text": text,
                "text_lines": list(self.text_lines),
                "images_base64": list(self.images_base64),
                "row": self.row,
                "col": self.col,
                "rowspan": self.rowspan,
                "colspan": self.colspan,
            }
            if self.bbox is not None:
                payload["bbox"] = list(self.bbox)
            if self.metadata:
                payload["metadata"] = dict(self.metadata)
            return payload

        if mode == "images":
            if not self.images_base64:
                return None
            if len(self.images_base64) == 1:
                return self.images_base64[0]
            return list(self.images_base64)

        raise ValueError(f"unsupported value mode: {mode}")


@dataclass(slots=True)
class NormalizeConfig:
    span_mode: SpanMode = "preserve"
    collision_policy: CollisionPolicy = "raise"
    row_count: int | None = None
    col_count: int | None = None
    filler: Any = None
    line_joiner: str = "\n"
    header_rows: Sequence[int] | None = None
    header_separator: str = " | "

    def __post_init__(self) -> None:
        self.row_count = None if self.row_count is None else int(self.row_count)
        self.col_count = None if self.col_count is None else int(self.col_count)
        if self.header_rows is None:
            self.header_rows = None
        else:
            self.header_rows = tuple(sorted({int(row) for row in self.header_rows}))
        self.line_joiner = str(self.line_joiner)
        self.header_separator = str(self.header_separator)

    def with_updates(self, **changes: Any) -> NormalizeConfig:
        return replace(self, **changes)


@dataclass(slots=True)
class ExtractedTable:
    page_number: int
    cells: Sequence[TableCell] = field(default_factory=tuple)
    n_rows: int = 0
    n_cols: int = 0
    bbox: BoundingBox | Sequence[float] | None = None
    header_rows: Sequence[int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.page_number = int(self.page_number)
        self.n_rows = int(self.n_rows)
        self.n_cols = int(self.n_cols)
        if self.page_number < 0:
            raise ValueError("page_number must be non-negative")
        if self.n_rows < 0 or self.n_cols < 0:
            raise ValueError("n_rows and n_cols must be non-negative")
        self.cells = tuple(self.cells)
        self.bbox = _coerce_bbox(self.bbox)
        if self.header_rows is None:
            self.header_rows = None
        else:
            self.header_rows = tuple(sorted({int(row) for row in self.header_rows}))
        self.metadata = dict(self.metadata)

    @property
    def cell_count(self) -> int:
        return len(self.cells)

    def iter_cells(self) -> tuple[TableCell, ...]:
        return tuple(sorted(self.cells, key=lambda cell: (cell.row, cell.col, cell.rowspan, cell.colspan)))

    def max_covered_row(self) -> int:
        return max((cell.row + cell.rowspan for cell in self.cells), default=0)

    def max_covered_col(self) -> int:
        return max((cell.col + cell.colspan for cell in self.cells), default=0)

    def validate(self, *, strict: bool = True) -> list[str]:
        errors: list[str] = []
        for index, cell in enumerate(self.cells):
            if cell.row < 0 or cell.col < 0:
                errors.append(f"cell {index} has negative coordinates")
            if cell.rowspan < 1 or cell.colspan < 1:
                errors.append(f"cell {index} has invalid span")
            if self.n_rows and cell.row + cell.rowspan > self.n_rows:
                errors.append(f"cell {index} exceeds declared row count")
            if self.n_cols and cell.col + cell.colspan > self.n_cols:
                errors.append(f"cell {index} exceeds declared column count")
        if strict and errors:
            raise ValueError("; ".join(errors))
        return errors

    def to_normalized(self, config: NormalizeConfig | None = None, *, value_mode: ValueMode = "native") -> NormalizedTable:
        from .normalize import normalize_table

        return normalize_table(self, config, value_mode=value_mode)

    def with_metadata(self, **changes: Any) -> ExtractedTable:
        merged = dict(self.metadata)
        merged.update(changes)
        return replace(self, metadata=merged)


@dataclass(slots=True)
class NormalizedTable:
    matrix: tuple[tuple[Any, ...], ...]
    source: ExtractedTable | None = None
    config: NormalizeConfig = field(default_factory=NormalizeConfig)
    value_mode: ValueMode = "native"
    column_names: tuple[str, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = tuple(tuple(row) for row in self.matrix)
        if self.column_names is not None:
            self.column_names = tuple(str(name) for name in self.column_names)
        self.metadata = dict(self.metadata)

    @property
    def row_count(self) -> int:
        return len(self.matrix)

    @property
    def col_count(self) -> int:
        return max((len(row) for row in self.matrix), default=0)

    def as_rows(self) -> list[list[Any]]:
        return [list(row) for row in self.matrix]

    def to_pandas(self, **kwargs: Any):
        from .adapters import table_to_pandas

        return table_to_pandas(self, **kwargs)

    def to_polars(self, **kwargs: Any):
        from .adapters import table_to_polars

        return table_to_polars(self, **kwargs)

    def with_column_names(self, column_names: Sequence[str] | None) -> NormalizedTable:
        return replace(self, column_names=None if column_names is None else tuple(column_names))


Table = ExtractedTable
Cell = TableCell

__all__ = [
    "Cell",
    "BoundingBox",
    "CollisionPolicy",
    "ExtractedTable",
    "NormalizeConfig",
    "NormalizedTable",
    "SpanMode",
    "Table",
    "TableCell",
    "ValueMode",
]
