from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterable, Sequence

from .models import BoundingBox, ExtractedTable, TableCell


@dataclass(slots=True)
class ExtractionConfig:
    pages: Sequence[int] | None = None
    preserve_images: bool = True
    image_encoding: str = "base64"
    min_rows: int = 2
    min_cols: int = 2
    line_tolerance: float = 4.0
    word_gap_tolerance: float = 10.0
    separator_tolerance: float = 12.0
    row_merge_tolerance: float = 2.0
    table_gap_tolerance: float = 18.0
    min_region_words: int = 4
    image_intersection_ratio: float = 0.1
    use_builtin_table_finder: bool = True

    def __post_init__(self) -> None:
        if self.image_encoding != "base64":
            raise ValueError("only base64 image encoding is currently supported")
        if self.min_rows < 1 or self.min_cols < 1:
            raise ValueError("min_rows and min_cols must be positive integers")


@dataclass(slots=True)
class _Word:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str

    @property
    def bbox(self) -> BoundingBox:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def center_y(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass(slots=True)
class _Segment:
    words: tuple[_Word, ...]
    text: str
    bbox: BoundingBox

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def x1(self) -> float:
        return self.bbox[2]


@dataclass(slots=True)
class _VisualLine:
    words: tuple[_Word, ...]
    segments: tuple[_Segment, ...]
    bbox: BoundingBox

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


@dataclass(slots=True)
class _LogicalRow:
    lines: tuple[_VisualLine, ...]
    bbox: BoundingBox

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def y1(self) -> float:
        return self.bbox[3]


@dataclass(slots=True)
class _PageImage:
    bbox: BoundingBox
    base64_data: str
    ext: str = "png"


@dataclass(slots=True)
class _CellBuilder:
    text_by_line: dict[int, list[str]] = field(default_factory=dict)
    images_base64: list[str] = field(default_factory=list)


def extract_pdf_tables(
    pdf_path: str | Path,
    *,
    config: ExtractionConfig | None = None,
) -> list[ExtractedTable]:
    cfg = config or ExtractionConfig()

    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("PyMuPDF is required to extract tables from PDFs") from exc

    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(path)

    document = fitz.open(path)
    try:
        tables: list[ExtractedTable] = []
        for page_index in _resolve_page_indexes(document.page_count, cfg.pages):
            page = document.load_page(page_index)
            tables.extend(_extract_tables_from_page(document, page, page_index + 1, cfg))
        return tables
    finally:
        document.close()


def _extract_tables_from_page(document, page, page_number: int, config: ExtractionConfig) -> list[ExtractedTable]:
    candidate_bboxes: list[BoundingBox] = []
    if config.use_builtin_table_finder:
        candidate_bboxes.extend(_builtin_table_bboxes(page))
    candidate_bboxes.extend(_candidate_bboxes_from_text(page, config))
    candidate_bboxes = sorted(_dedupe_bboxes(candidate_bboxes), key=lambda bbox: (bbox[1], bbox[0]))
    if not candidate_bboxes:
        return []

    page_images = _extract_page_images(document, page, None) if config.preserve_images else []
    tables: list[ExtractedTable] = []

    for table_index, bbox in enumerate(candidate_bboxes, start=1):
        table = _build_table_from_bbox(page, page_number, table_index, bbox, page_images, config)
        if table is not None:
            tables.append(table)

    return tables


def _resolve_page_indexes(page_count: int, pages: Sequence[int] | None) -> list[int]:
    if pages is None:
        return list(range(page_count))

    indexes: list[int] = []
    for page_number in pages:
        if page_number < 1 or page_number > page_count:
            raise ValueError(f"page {page_number} is out of range for a document with {page_count} pages")
        indexes.append(page_number - 1)
    return indexes


def _builtin_table_bboxes(page) -> list[BoundingBox]:
    if not hasattr(page, "find_tables"):
        return []

    try:
        finder = page.find_tables()
    except Exception:
        return []

    raw_tables = getattr(finder, "tables", finder)
    bboxes: list[BoundingBox] = []
    for raw_table in raw_tables or []:
        bbox = _coerce_bbox(getattr(raw_table, "bbox", None))
        if bbox is not None:
            bboxes.append(bbox)
    return bboxes


def _candidate_bboxes_from_text(page, config: ExtractionConfig) -> list[BoundingBox]:
    words = _extract_words(page, None)
    if not words:
        return []

    lines = _group_words_into_lines(words, config)
    candidates: list[BoundingBox] = []
    current_group: list[_VisualLine] = []

    for line in lines:
        is_table_like = len(line.segments) >= config.min_cols
        if is_table_like:
            if current_group and line.y0 - current_group[-1].y1 > config.table_gap_tolerance:
                _append_candidate_group(candidates, current_group, config)
                current_group = []
            current_group.append(line)
            continue

        _append_candidate_group(candidates, current_group, config)
        current_group = []

    _append_candidate_group(candidates, current_group, config)
    return candidates


def _append_candidate_group(
    candidates: list[BoundingBox],
    group: list[_VisualLine],
    config: ExtractionConfig,
) -> None:
    if len(group) < config.min_rows:
        return
    word_count = sum(len(line.words) for line in group)
    if word_count < config.min_region_words:
        return
    candidates.append(_pad_bbox(_bbox_union(line.bbox for line in group), 4.0))


def _build_table_from_bbox(
    page,
    page_number: int,
    table_index: int,
    bbox: BoundingBox,
    page_images: Sequence[_PageImage],
    config: ExtractionConfig,
) -> ExtractedTable | None:
    words = _extract_words(page, bbox)
    if not words:
        return None

    lines = _group_words_into_lines(words, config)
    if len(lines) < config.min_rows:
        return None

    column_edges = _infer_column_edges(lines, bbox, config)
    if len(column_edges) - 1 < config.min_cols:
        return None

    rows = _merge_lines_into_rows(lines, column_edges, config)
    if len(rows) < config.min_rows:
        return None

    images = [image for image in page_images if _intersection_ratio(image.bbox, bbox) > 0.0]
    cells = _build_cells(rows, column_edges, images, config)
    if not cells:
        return None

    header_rows = (0,) if _looks_like_header_row(rows[0], column_edges) and len(rows) > 1 else None
    return ExtractedTable(
        page_number=page_number,
        bbox=bbox,
        n_rows=len(rows),
        n_cols=len(column_edges) - 1,
        cells=cells,
        header_rows=header_rows,
        metadata={
            "engine": "pymupdf",
            "table_index": table_index,
        },
    )


def _looks_like_header_row(row: _LogicalRow, column_edges: Sequence[float]) -> bool:
    groups = _group_segments_by_cell(row, column_edges)
    if not groups:
        return False

    texts: list[str] = []
    for builder in groups.values():
        for parts in builder.text_by_line.values():
            texts.extend(parts)
    if not texts:
        return False
    return all(any(char.isalpha() for char in text) for text in texts)


def _build_cells(
    rows: Sequence[_LogicalRow],
    column_edges: Sequence[float],
    images: Sequence[_PageImage],
    config: ExtractionConfig,
) -> list[TableCell]:
    cells: list[TableCell] = []

    for row_index, row in enumerate(rows):
        grouped = _group_segments_by_cell(row, column_edges)
        _attach_images_to_groups(grouped, row, column_edges, images, config)

        for (col_index, colspan), builder in sorted(grouped.items()):
            text_lines: list[str] = []
            for line_index in sorted(builder.text_by_line):
                parts = [part.strip() for part in builder.text_by_line[line_index] if part.strip()]
                if parts:
                    text_lines.append(" ".join(parts))

            images_base64 = list(dict.fromkeys(builder.images_base64))
            if not text_lines and not images_base64:
                continue

            text = "\n".join(text_lines) if text_lines else None
            cell_bbox = (
                column_edges[col_index],
                row.y0,
                column_edges[col_index + colspan],
                row.y1,
            )
            cells.append(
                TableCell(
                    row=row_index,
                    col=col_index,
                    bbox=cell_bbox,
                    text=text,
                    text_lines=text_lines,
                    images_base64=images_base64,
                    colspan=colspan,
                )
            )

    return cells


def _group_segments_by_cell(
    row: _LogicalRow,
    column_edges: Sequence[float],
) -> dict[tuple[int, int], _CellBuilder]:
    grouped: dict[tuple[int, int], _CellBuilder] = {}

    for line_index, line in enumerate(row.lines):
        for segment in line.segments:
            covered_cols = _covered_columns(segment.bbox, column_edges)
            if not covered_cols:
                continue
            key = (covered_cols[0], len(covered_cols))
            builder = grouped.setdefault(key, _CellBuilder())
            builder.text_by_line.setdefault(line_index, []).append(segment.text)

    return grouped


def _attach_images_to_groups(
    groups: dict[tuple[int, int], _CellBuilder],
    row: _LogicalRow,
    column_edges: Sequence[float],
    images: Sequence[_PageImage],
    config: ExtractionConfig,
) -> None:
    for image in images:
        if _intersection_ratio(image.bbox, row.bbox) < config.image_intersection_ratio:
            continue

        covered_cols = _covered_columns(image.bbox, column_edges)
        if not covered_cols:
            continue

        key = (covered_cols[0], len(covered_cols))
        builder = groups.setdefault(key, _CellBuilder())
        builder.images_base64.append(image.base64_data)


def _merge_lines_into_rows(
    lines: Sequence[_VisualLine],
    column_edges: Sequence[float],
    config: ExtractionConfig,
) -> list[_LogicalRow]:
    if not lines:
        return []

    rows: list[_LogicalRow] = []
    current_lines: list[_VisualLine] = [lines[0]]

    for next_line in lines[1:]:
        if _should_merge_lines(current_lines[-1], next_line, column_edges, config):
            current_lines.append(next_line)
            continue
        rows.append(_make_logical_row(current_lines))
        current_lines = [next_line]

    rows.append(_make_logical_row(current_lines))
    return rows


def _should_merge_lines(
    current: _VisualLine,
    next_line: _VisualLine,
    column_edges: Sequence[float],
    config: ExtractionConfig,
) -> bool:
    vertical_gap = next_line.y0 - current.y1
    current_height = max(0.0, current.y1 - current.y0)
    next_height = max(0.0, next_line.y1 - next_line.y0)
    gap_limit = min(config.row_merge_tolerance, max(0.8, min(current_height, next_height) * 0.22))
    if vertical_gap > gap_limit:
        return False

    current_cols = set(_occupied_columns(current, column_edges))
    next_cols = set(_occupied_columns(next_line, column_edges))
    if not current_cols or not next_cols:
        return False

    if current_cols == next_cols:
        return False

    smaller_cols, larger_cols = (
        (current_cols, next_cols) if len(current_cols) <= len(next_cols) else (next_cols, current_cols)
    )
    if not smaller_cols.issubset(larger_cols):
        return False

    return len(smaller_cols) == 1 and len(larger_cols) > 1


def _occupied_columns(line: _VisualLine, column_edges: Sequence[float]) -> list[int]:
    occupied: list[int] = []
    for segment in line.segments:
        occupied.extend(_covered_columns(segment.bbox, column_edges))
    return sorted(set(occupied))


def _make_logical_row(lines: Sequence[_VisualLine]) -> _LogicalRow:
    return _LogicalRow(lines=tuple(lines), bbox=_bbox_union(line.bbox for line in lines))


def _infer_column_edges(
    lines: Sequence[_VisualLine],
    bbox: BoundingBox,
    config: ExtractionConfig,
) -> list[float]:
    separator_candidates: list[float] = []

    for line in lines:
        if len(line.segments) < 2:
            continue
        for left, right in zip(line.segments, line.segments[1:]):
            gap = right.x0 - left.x1
            if gap >= config.word_gap_tolerance:
                separator_candidates.append((left.x1 + right.x0) / 2)

    min_cluster_size = 2 if len(lines) >= 2 else 1
    separators = _cluster_positions(
        separator_candidates,
        tolerance=config.separator_tolerance,
        min_cluster_size=min_cluster_size,
    )

    if not separators:
        densest = max(lines, key=lambda line: len(line.segments))
        separators = [(left.x1 + right.x0) / 2 for left, right in zip(densest.segments, densest.segments[1:])]

    text_left = min((segment.bbox[0] for line in lines for segment in line.segments), default=bbox[0])
    text_right = max((segment.bbox[2] for line in lines for segment in line.segments), default=bbox[2])
    edges = [text_left, *sorted(separators), text_right]

    deduped: list[float] = []
    for edge in edges:
        if not deduped or abs(edge - deduped[-1]) > 1.0:
            deduped.append(edge)

    if len(deduped) < 2:
        return [bbox[0], bbox[2]]
    return deduped


def _horizontal_overlap_ratio(left: BoundingBox, right: BoundingBox) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    width = max(left[2] - left[0], 1e-9)
    return overlap / width


def _covered_columns(bbox: BoundingBox, column_edges: Sequence[float]) -> list[int]:
    covered: list[int] = []
    for col_index in range(len(column_edges) - 1):
        probe_bbox = (column_edges[col_index], bbox[1], column_edges[col_index + 1], bbox[3])
        if _horizontal_overlap_ratio(bbox, probe_bbox) >= 0.35:
            covered.append(col_index)

    if covered:
        return covered

    center = (bbox[0] + bbox[2]) / 2
    for col_index in range(len(column_edges) - 1):
        if column_edges[col_index] <= center <= column_edges[col_index + 1]:
            return [col_index]
    return []


def _group_words_into_lines(words: Sequence[_Word], config: ExtractionConfig) -> list[_VisualLine]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda word: (word.y0, word.x0))
    lines: list[list[_Word]] = []
    current_line: list[_Word] = [sorted_words[0]]

    for word in sorted_words[1:]:
        if _belongs_to_same_line(current_line, word, config.line_tolerance):
            current_line.append(word)
            continue
        lines.append(current_line)
        current_line = [word]

    lines.append(current_line)

    visual_lines: list[_VisualLine] = []
    for line_words in lines:
        ordered = tuple(sorted(line_words, key=lambda word: word.x0))
        segments = _segment_words(ordered, config.word_gap_tolerance)
        if not segments:
            continue
        visual_lines.append(
            _VisualLine(
                words=ordered,
                segments=segments,
                bbox=_bbox_union(word.bbox for word in ordered),
            )
        )
    return visual_lines


def _belongs_to_same_line(current_line: Sequence[_Word], word: _Word, line_tolerance: float) -> bool:
    center = sum(item.center_y for item in current_line) / len(current_line)
    current_height = median(item.height for item in current_line)
    tolerance = max(line_tolerance, current_height * 0.6)
    return abs(word.center_y - center) <= tolerance


def _segment_words(words: Sequence[_Word], word_gap_tolerance: float) -> tuple[_Segment, ...]:
    if not words:
        return ()

    gaps = [current.x0 - previous.x1 for previous, current in zip(words, words[1:]) if current.x0 - previous.x1 > 0]
    sorted_gaps = sorted(gaps)
    midpoint = len(sorted_gaps) // 2
    lower_half = sorted_gaps[:midpoint] if midpoint else sorted_gaps[:1]
    upper_half = sorted_gaps[midpoint:] if midpoint else sorted_gaps[:1]
    lower_baseline = median(lower_half) if lower_half else 0.0
    upper_baseline = median(upper_half) if upper_half else lower_baseline
    median_height = median(word.height for word in words)

    if lower_baseline > 0 and upper_baseline >= lower_baseline * 1.7:
        dynamic_gap = min(word_gap_tolerance, max(2.0, (lower_baseline + upper_baseline) / 2))
    else:
        dynamic_gap = min(word_gap_tolerance, max(3.0, median_height * 0.65))

    grouped: list[list[_Word]] = []
    current_segment: list[_Word] = [words[0]]

    for word in words[1:]:
        gap = word.x0 - current_segment[-1].x1
        if gap <= dynamic_gap:
            current_segment.append(word)
            continue
        grouped.append(current_segment)
        current_segment = [word]

    grouped.append(current_segment)

    segments: list[_Segment] = []
    for segment_words in grouped:
        text = " ".join(word.text for word in segment_words).strip()
        if not text:
            continue
        segments.append(
            _Segment(
                words=tuple(segment_words),
                text=text,
                bbox=_bbox_union(word.bbox for word in segment_words),
            )
        )
    return tuple(segments)


def _extract_words(page, clip: BoundingBox | None) -> list[_Word]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("PyMuPDF is required to extract words from PDFs") from exc

    clip_rect = fitz.Rect(*clip) if clip is not None else None
    raw_words = page.get_text("words", clip=clip_rect) if clip_rect is not None else page.get_text("words")

    words: list[_Word] = []
    for raw_word in raw_words:
        if len(raw_word) < 5:
            continue
        text = str(raw_word[4]).strip()
        if not text:
            continue
        words.append(
            _Word(
                x0=float(raw_word[0]),
                y0=float(raw_word[1]),
                x1=float(raw_word[2]),
                y1=float(raw_word[3]),
                text=text,
            )
        )
    return words


def _extract_page_images(document, page, clip: BoundingBox | None) -> list[_PageImage]:
    seen: set[tuple[int, int, int, int]] = set()
    images: list[_PageImage] = []

    for image in _iter_page_images(document, page):
        if clip is not None and _intersection_ratio(image.bbox, clip) <= 0.0:
            continue
        key = tuple(int(round(value)) for value in image.bbox)
        if key in seen:
            continue
        seen.add(key)
        images.append(image)

    return images


def _iter_page_images(document, page) -> Iterable[_PageImage]:
    try:
        infos = page.get_image_info(xrefs=True)
    except Exception:
        infos = []

    for info in infos or []:
        bbox = _coerce_bbox(info.get("bbox"))
        if bbox is None:
            continue

        image_bytes = None
        ext = str(info.get("ext") or "png")
        xref = info.get("xref")
        if xref:
            try:
                extracted = document.extract_image(xref)
            except Exception:
                extracted = None
            if extracted:
                image_bytes = extracted.get("image")
                ext = str(extracted.get("ext") or ext)

        if image_bytes is None:
            image_bytes, ext = _render_bbox_as_png(page, bbox)

        if image_bytes:
            yield _PageImage(
                bbox=bbox,
                base64_data=base64.b64encode(image_bytes).decode("ascii"),
                ext=ext,
            )

    if infos:
        return

    try:
        raw_dict = page.get_text("dict")
    except Exception:
        return

    for block in raw_dict.get("blocks", []):
        if block.get("type") != 1:
            continue
        bbox = _coerce_bbox(block.get("bbox"))
        if bbox is None:
            continue
        image_bytes = block.get("image")
        ext = str(block.get("ext") or "png")
        if image_bytes is None:
            image_bytes, ext = _render_bbox_as_png(page, bbox)
        if image_bytes:
            yield _PageImage(
                bbox=bbox,
                base64_data=base64.b64encode(image_bytes).decode("ascii"),
                ext=ext,
            )


def _render_bbox_as_png(page, bbox: BoundingBox) -> tuple[bytes | None, str]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("PyMuPDF is required to render PDF regions") from exc

    try:
        pixmap = page.get_pixmap(clip=fitz.Rect(*bbox), alpha=False)
    except Exception:
        return None, "png"

    try:
        return pixmap.tobytes("png"), "png"
    except TypeError:
        try:
            return pixmap.tobytes(output="png"), "png"
        except Exception:
            return None, "png"


def _coerce_bbox(value) -> BoundingBox | None:
    if value is None or len(value) != 4:
        return None
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _cluster_positions(
    positions: Sequence[float],
    *,
    tolerance: float,
    min_cluster_size: int = 1,
) -> list[float]:
    if not positions:
        return []

    sorted_positions = sorted(float(position) for position in positions)
    clusters: list[list[float]] = [[sorted_positions[0]]]
    for position in sorted_positions[1:]:
        if abs(position - clusters[-1][-1]) <= tolerance:
            clusters[-1].append(position)
            continue
        clusters.append([position])

    return [
        sum(cluster) / len(cluster)
        for cluster in clusters
        if len(cluster) >= min_cluster_size
    ]


def _dedupe_bboxes(bboxes: Sequence[BoundingBox]) -> list[BoundingBox]:
    deduped: list[BoundingBox] = []
    for bbox in bboxes:
        if any(_bbox_overlap_ratio(bbox, existing) >= 0.8 for existing in deduped):
            continue
        deduped.append(bbox)
    return deduped


def _bbox_overlap_ratio(left: BoundingBox, right: BoundingBox) -> float:
    intersection = _intersection_bbox(left, right)
    if intersection is None:
        return 0.0
    return _bbox_area(intersection) / max(_bbox_area(left), _bbox_area(right), 1e-9)


def _intersection_ratio(left: BoundingBox, right: BoundingBox) -> float:
    intersection = _intersection_bbox(left, right)
    if intersection is None:
        return 0.0
    return _bbox_area(intersection) / max(min(_bbox_area(left), _bbox_area(right)), 1e-9)


def _intersection_bbox(left: BoundingBox, right: BoundingBox) -> BoundingBox | None:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def _bbox_union(boxes: Iterable[BoundingBox]) -> BoundingBox:
    iterator = iter(boxes)
    first = next(iterator)
    x0, y0, x1, y1 = first
    for box in iterator:
        x0 = min(x0, box[0])
        y0 = min(y0, box[1])
        x1 = max(x1, box[2])
        y1 = max(y1, box[3])
    return (x0, y0, x1, y1)


def _pad_bbox(bbox: BoundingBox, padding: float) -> BoundingBox:
    return (
        bbox[0] - padding,
        bbox[1] - padding,
        bbox[2] + padding,
        bbox[3] + padding,
    )


def _bbox_area(bbox: BoundingBox) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


__all__ = [
    "ExtractionConfig",
    "extract_pdf_tables",
]



