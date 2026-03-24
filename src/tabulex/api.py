from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

from .adapters import table_to_json, table_to_pandas, table_to_polars
from .extraction import ExtractionConfig, extract_pdf_tables
from .models import ExtractedTable, NormalizeConfig, ValueMode

ReturnType = Literal["tables", "pandas", "polars", "json", "both", "all"]
EngineName = Literal["pymupdf"]


@dataclass(slots=True)
class ExtractionResult:
    tables: tuple[ExtractedTable, ...]
    pandas_tables: tuple[Any, ...] = ()
    polars_tables: tuple[Any, ...] = ()
    json_tables: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_pandas(
        self,
        *,
        config: NormalizeConfig | None = None,
        value_mode: ValueMode = "native",
        mixed_mode: str = "python",
    ) -> list[Any]:
        if config is None and value_mode == "native" and mixed_mode == "python" and self.pandas_tables:
            return list(self.pandas_tables)
        return [
            table_to_pandas(table, config=config, value_mode=value_mode, mixed_mode=mixed_mode)
            for table in self.tables
        ]

    def to_polars(
        self,
        *,
        config: NormalizeConfig | None = None,
        value_mode: ValueMode = "native",
        mixed_mode: str = "json",
    ) -> list[Any]:
        if config is None and value_mode == "native" and mixed_mode == "json" and self.polars_tables:
            return list(self.polars_tables)
        return [
            table_to_polars(table, config=config, value_mode=value_mode, mixed_mode=mixed_mode)
            for table in self.tables
        ]

    def to_json(
        self,
        *,
        config: NormalizeConfig | None = None,
        value_mode: ValueMode = "native",
        mixed_mode: str = "python",
        orient: str = "records",
        as_string: bool = False,
        ensure_ascii: bool = False,
        indent: int | None = 2,
    ) -> list[Any]:
        if (
            config is None
            and value_mode == "native"
            and mixed_mode == "python"
            and orient == "records"
            and not as_string
            and not ensure_ascii
            and indent == 2
            and self.json_tables
        ):
            return list(self.json_tables)
        return [
            table_to_json(
                table,
                config=config,
                value_mode=value_mode,
                mixed_mode=mixed_mode,
                orient=orient,
                as_string=as_string,
                ensure_ascii=ensure_ascii,
                indent=indent,
            )
            for table in self.tables
        ]


def extract_tables(
    pdf_path: str | Path,
    *,
    pages: Sequence[int] | None = None,
    return_type: ReturnType = "both",
    preserve_spans: bool = True,
    preserve_images: bool = True,
    image_encoding: str = "base64",
    engine: EngineName = "pymupdf",
    value_mode: ValueMode = "native",
    mixed_mode: str | None = None,
    min_rows: int = 2,
    min_cols: int = 2,
    use_builtin_table_finder: bool = True,
) -> ExtractionResult:
    if engine != "pymupdf":
        raise ValueError("only the 'pymupdf' engine is implemented in this version")
    if return_type not in {"tables", "pandas", "polars", "json", "both", "all"}:
        raise ValueError("return_type must be one of: 'tables', 'pandas', 'polars', 'json', 'both', 'all'")

    normalize_config = NormalizeConfig(span_mode="preserve" if preserve_spans else "expand")
    extraction_config = ExtractionConfig(
        pages=pages,
        preserve_images=preserve_images,
        image_encoding=image_encoding,
        min_rows=min_rows,
        min_cols=min_cols,
        use_builtin_table_finder=use_builtin_table_finder,
    )

    tables = tuple(extract_pdf_tables(pdf_path, config=extraction_config))
    pandas_tables: tuple[Any, ...] = ()
    polars_tables: tuple[Any, ...] = ()
    json_tables: tuple[Any, ...] = ()

    pandas_mixed_mode = mixed_mode or "python"
    polars_mixed_mode = mixed_mode or "json"
    json_mixed_mode = mixed_mode or "python"

    if return_type in {"pandas", "both", "all"}:
        pandas_tables = tuple(
            table_to_pandas(
                table,
                config=normalize_config,
                value_mode=value_mode,
                mixed_mode=pandas_mixed_mode,
            )
            for table in tables
        )

    if return_type in {"polars", "both", "all"}:
        polars_tables = tuple(
            table_to_polars(
                table,
                config=normalize_config,
                value_mode=value_mode,
                mixed_mode=polars_mixed_mode,
            )
            for table in tables
        )

    if return_type in {"json", "all"}:
        json_tables = tuple(
            table_to_json(
                table,
                config=normalize_config,
                value_mode=value_mode,
                mixed_mode=json_mixed_mode,
                orient="records",
                as_string=False,
                ensure_ascii=False,
                indent=2,
            )
            for table in tables
        )

    return ExtractionResult(
        tables=tables,
        pandas_tables=pandas_tables,
        polars_tables=polars_tables,
        json_tables=json_tables,
        metadata={
            "engine": engine,
            "source_path": str(Path(pdf_path)),
            "requested_pages": tuple(pages) if pages is not None else None,
        },
    )


def extract_tables_to_pandas(pdf_path: str | Path, **kwargs: Any) -> list[Any]:
    result = extract_tables(pdf_path, return_type="pandas", **kwargs)
    return list(result.pandas_tables)


def extract_tables_to_polars(pdf_path: str | Path, **kwargs: Any) -> list[Any]:
    result = extract_tables(pdf_path, return_type="polars", **kwargs)
    return list(result.polars_tables)


def extract_tables_to_json(pdf_path: str | Path, **kwargs: Any) -> list[Any]:
    result = extract_tables(pdf_path, return_type="json", **kwargs)
    return list(result.json_tables)


__all__ = [
    "EngineName",
    "ExtractionResult",
    "ReturnType",
    "extract_tables",
    "extract_tables_to_json",
    "extract_tables_to_pandas",
    "extract_tables_to_polars",
]
