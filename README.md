# Tabulex

English | Espanol

## English

`Tabulex` is a Python library for detecting tables in digital PDFs, reconstructing their structure, and exporting them to `pandas`, `polars`, and JSON.

### Features

- Process one-page or multi-page PDF files.
- Detect tables with `PyMuPDF` plus a text-geometry fallback heuristic.
- Preserve multiline cells by joining lines with `\n`.
- Represent merged cells with `rowspan` and `colspan` in an intermediate model.
- Store cell images as base64.
- Export tables as Pandas, Polars, or JSON.

### Installation

```bash
pip install tabulex
```

### Quick Start

```python
from tabulex import extract_tables

result = extract_tables("sample-tables.pdf", return_type="all")

print(result.pandas_tables[0])
print(result.polars_tables[0])
print(result.json_tables[0])
```

### JSON Output

```python
from tabulex import extract_tables

result = extract_tables("sample-tables.pdf", return_type="json")
print(result.json_tables[0])
```

### Main API

```python
extract_tables(
    pdf_path: str,
    pages: list[int] | None = None,
    return_type: str = "both",
    preserve_spans: bool = True,
    preserve_images: bool = True,
    image_encoding: str = "base64",
    engine: str = "pymupdf",
)
```

`extract_tables(...)` returns an `ExtractionResult` with:

- `tables`
- `pandas_tables`
- `polars_tables`
- `json_tables`
- `metadata`

### Limitations

- V1 targets digital PDFs.
- Scanned PDFs and OCR-heavy workflows are not part of the default path.
- Borderless tables still rely on heuristics.
- Tables continued across pages are handled separately.

### License

MIT. See [`LICENSE`](LICENSE).

## Espanol

`Tabulex` es una libreria Python para detectar tablas en PDFs digitales, reconstruir su estructura y exportarlas a `pandas`, `polars` y JSON.

### Caracteristicas

- Procesa PDFs de una o varias paginas.
- Detecta tablas con `PyMuPDF` y una heuristica de respaldo basada en geometria del texto.
- Conserva celdas multilinea uniendo las lineas con `\n`.
- Representa celdas fusionadas con `rowspan` y `colspan` en un modelo intermedio.
- Almacena imagenes de celdas como base64.
- Exporta tablas a Pandas, Polars o JSON.

### Instalacion

```bash
pip install tabulex
```

### Uso Rapido

```python
from tabulex import extract_tables

result = extract_tables("sample-tables.pdf", return_type="all")

print(result.pandas_tables[0])
print(result.polars_tables[0])
print(result.json_tables[0])
```

### Salida JSON

```python
from tabulex import extract_tables

result = extract_tables("sample-tables.pdf", return_type="json")
print(result.json_tables[0])
```

### API Principal

```python
extract_tables(
    pdf_path: str,
    pages: list[int] | None = None,
    return_type: str = "both",
    preserve_spans: bool = True,
    preserve_images: bool = True,
    image_encoding: str = "base64",
    engine: str = "pymupdf",
)
```

`extract_tables(...)` devuelve un `ExtractionResult` con:

- `tables`
- `pandas_tables`
- `polars_tables`
- `json_tables`
- `metadata`

### Limitaciones

- La V1 esta pensada para PDFs digitales.
- Los PDFs escaneados y los flujos con OCR no forman parte del camino principal.
- Las tablas sin bordes siguen dependiendo de heuristicas.
- Las tablas continuadas entre paginas se procesan por separado.

### Licencia

MIT. Consulta [`LICENSE`](LICENSE).
