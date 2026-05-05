"""Post-process Document Intelligence layout JSON into a human-readable summary.

=============================================================================
OVERVIEW
=============================================================================
Document Intelligence (formerly Form Recognizer) returns a large, geometry-rich
JSON describing every word, line, paragraph, table, selection mark (checkbox),
and handwriting style detected on a document. This module turns that raw
response into a compact, human-friendly summary you can render in a UI or
feed downstream.

Two entry points:
    summarize(layout_dict)        -> dispatches based on model used
    summarize_layout(layout_dict) -> for prebuilt-layout responses
    summarize_read(layout_dict)   -> for prebuilt-read responses

The input may be either:
    - The full REST/SDK response: { "analyzeResult": { ... } }
    - The unwrapped analyzeResult body: { "modelId": ..., "pages": [...] }
The `_ar()` helper transparently handles both shapes.

=============================================================================
LAYOUT OUTPUT SHAPE
=============================================================================
    {
      "model":        "prebuilt-layout",
      "api_version":  "2024-11-30",
      "page_count":   8,
      "handwriting":  ["Jane Doe", "01/01/2000", ...],   # styles[].isHandwritten
      "checkboxes":   [                                   # Yes/No questions
        {"page": 1, "question": "Are you homeless?",
         "answer": "no", "confidence": 0.99},
        ...
      ],
      "options":      [                                   # standalone checkboxes
        {"page": 3, "option": "SNAP",
         "selected": true, "confidence": 0.99},
        ...
      ],
      "tables":       [
        {"page": 3, "rows": 5, "columns": 5,
         "cells": [[...], ...],
         "key_values": {"first cell": "rest joined by |"}},
        ...
      ],
    }

=============================================================================
HOW THE CHECKBOX/Yes/No BINDING WORKS
=============================================================================
Document Intelligence reports each selection mark as a separate object with
a bounding polygon and a `state` of "selected" or "unselected". It does NOT
tell you which question or label the mark belongs to. We bind marks to labels
geometrically in two passes:

  Pass 1 (extract_checkbox_answers):
    - Find every "Yes" / "No" word on each page.
    - Pair each "Yes" with the nearest "No" to its right on the same row
      (within _PAIR_MAX_X_GAP). This handles both single-pair questions and
      forms with multiple Yes/No pairs side-by-side in the same row.
    - For each high-confidence mark, find the pair whose row it sits on,
      then assign it by horizontal *bracketing*:
          * mark to the right of "No" word -> belongs to No
          * mark between Yes and No        -> belongs to Yes
          * mark left of Yes               -> nearest by distance
      This matches typical form layouts:  `Yes ____ or No ____`
    - The pair's question text comes from `_question_for()` which tries the
      paragraph containing the Yes word's text span, falling back to the
      table cell label if the paragraph result is too short or just "Yes".

  Pass 2 (extract_standalone_checkboxes):
    - Any high-confidence mark NOT consumed by pass 1 is a bare checkbox
      ("check all that apply" lists, accessibility option lists, etc.).
    - Its label is the nearest line whose left edge sits to the right of the
      mark on the same row (gap <= _MAX_LABEL_X_GAP).
    - Junk filters drop labels that are too short, contain too few letters,
      or are blocklisted single words like "if" / "or".

=============================================================================
UNIVERSAL DESIGN — works on any document
=============================================================================
- Handwriting:  uses the language-agnostic `styles[].isHandwritten` array.
- Tables:       built from any tables Document Intelligence returns.
- Checkboxes:   guarded on actual mark presence — no marks => empty arrays.
- Read mode:    pure OCR text, no fabricated structure.
Documents without checkboxes (letters, contracts, receipts) yield empty
`checkboxes` and `options` lists rather than false positives.
"""

from __future__ import annotations

from typing import Any, Iterable

# ---------------------------------------------------------------------------
# TUNABLE CONSTANTS
# ---------------------------------------------------------------------------
# All distances are in the polygon's unit. For PDFs Document Intelligence
# reports inches; for images it reports pixels normalized by DPI. The defaults
# below are tuned for letter-size US government forms scanned around 200 dpi.
# Adjust these if you process forms with much tighter or looser line spacing.

_ROW_Y_TOLERANCE = 0.10      # vertical "same row" threshold for mark<->word matching
_MAX_MARK_TO_WORD = 1.0      # max horizontal distance from a mark to its Yes/No anchor
_MIN_MARK_CONFIDENCE = 0.5   # below this, a mark is treated as a printed template outline
                             #   (not an actual user check). Empirically the OCR returns
                             #   ~0.10 confidence for blank checkboxes and ~0.95+ for
                             #   real ink marks, so 0.5 cleanly separates them.
_PAIR_MAX_X_GAP = 1.5        # max gap between a Yes and its paired No on the same row
_MAX_LABEL_X_GAP = 1.5       # standalone checkbox: max gap from the mark to its label
_MIN_OPTION_CHARS = 2        # drop standalone option labels shorter than this
_MIN_OPTION_ALPHA = 2        # drop standalone options with fewer letters than this
                             #   (kills "? If" and stray punctuation glyphs)
_OPTION_BLOCKLIST = {"other", "if", "or", "and", "the", "no", "yes"}
                             # single-word labels that are almost always OCR noise
                             # rather than real options. "yes"/"no" guards against
                             # the standalone binder grabbing a Yes/No word that
                             # the pass-1 binder somehow missed.


# ---------------------------------------------------------------------------
# LOW-LEVEL HELPERS
# ---------------------------------------------------------------------------

def _ar(layout: dict) -> dict:
    """Return the analyzeResult sub-dict, accepting either shape.

    Document Intelligence REST/SDK responses have the shape
        { "status": "succeeded", "analyzeResult": { ...real data... } }
    but the SDK sometimes returns the inner dict directly. Accepting both makes
    every public function in this module robust regardless of where the caller
    got the JSON from.
    """
    return layout.get("analyzeResult", layout)


def _poly_center(polygon: list[float]) -> tuple[float, float]:
    """Return the (x, y) center of a Document Intelligence polygon.

    Polygons are flat arrays in the form [x0, y0, x1, y1, x2, y2, x3, y3]
    representing the four corners of a (usually axis-aligned) bounding box.
    Averaging x's and y's separately gives the geometric center.
    """
    xs = polygon[0::2]
    ys = polygon[1::2]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def extract_handwriting(analyze_result: dict) -> list[str]:
    """Return distinct handwritten text snippets from the document.

    Document Intelligence flags handwriting via the top-level `styles` array:
        styles: [
          { "isHandwritten": true,
            "confidence": 1.0,
            "spans": [ {"offset": 1150, "length": 8}, ... ] },
          ...
        ]
    Each `span` is a (offset, length) slice into the document's `content`
    string. We extract each slice, dedupe, and preserve insertion order so
    the result is reading-order friendly.

    This is language-agnostic — it works on any script the OCR engine handles.
    """
    content: str = analyze_result.get("content", "")
    snippets: list[str] = []
    seen: set[str] = set()
    for style in analyze_result.get("styles", []):
        # Skip non-handwriting styles (font weights, script categories, etc.)
        if not style.get("isHandwritten"):
            continue
        for span in style.get("spans", []):
            offset = span.get("offset", 0)
            length = span.get("length", 0)
            text = content[offset : offset + length].strip()
            # Dedupe: the same handwritten phrase can appear in multiple spans.
            if text and text not in seen:
                seen.add(text)
                snippets.append(text)
    return snippets


def _yes_no_words(page: dict) -> list[dict]:
    """Return every "Yes" or "No" word token on a single page.

    Each returned dict carries:
        label   : normalized "yes" or "no" (lowercased, punctuation stripped)
        cx, cy  : center coordinates of the word's bounding polygon
        polygon : the original polygon (kept for any later geometry)
        span    : the word's text span in the document content string,
                  used by `_question_for()` to look up the containing paragraph

    Note: this is the *only* place we hard-code English. Adding multilingual
    support means extending the `("yes", "no")` tuple below.
    """
    out = []
    for w in page.get("words", []):
        # Strip trailing punctuation so "Yes," and "No?" still match.
        c = w.get("content", "").strip().rstrip("?:.,").lower()
        if c in ("yes", "no") and w.get("polygon"):
            cx, cy = _poly_center(w["polygon"])
            out.append({
                "label": c,
                "cx": cx,
                "cy": cy,
                "polygon": w["polygon"],
                "span": w.get("span", {}),
            })
    return out


def _question_for(analyze_result: dict, yes_word: dict) -> str:
    """Find the descriptive question label for a Yes/No anchor word.

    Two-step strategy:
      1. Look up the *paragraph* containing the word's text span.
         For typical form prose like "Are you homeless? Yes ___ No ___",
         the entire question and the Yes/No live in one paragraph, so this
         gives a clean label.
      2. If the paragraph result is too short or just literally "Yes"/"No"
         (which happens when a Yes/No sits in its own table cell), fall back
         to `_table_cell_label_for()` which uses the table's row + column
         headers as the descriptive label, e.g. "Checking Accounts \u2014 Yes".

    The 8-character threshold for trusting the paragraph result is empirical:
    paragraphs of "Yes", "Yes No If", etc. are too generic to be useful.
    """
    span = yes_word.get("span") or {}
    offset = span.get("offset")

    # ----- Step 1: paragraph lookup --------------------------------------
    paragraph_text = ""
    if offset is not None:
        for para in analyze_result.get("paragraphs", []):
            for s in para.get("spans", []):
                # Span containment check: does this paragraph cover our word?
                if s.get("offset", 0) <= offset < s.get("offset", 0) + s.get("length", 0):
                    paragraph_text = para.get("content", "").strip()
                    # Trim the trailing " Yes ___ No ___" boilerplate so the
                    # caller sees just the question. We only trim if there's
                    # at least 10 chars before the " yes" marker, otherwise
                    # the entire paragraph would disappear.
                    lower = paragraph_text.lower()
                    idx = lower.rfind(" yes")
                    if idx > 10:
                        paragraph_text = paragraph_text[:idx]
                    paragraph_text = paragraph_text.strip().rstrip("?:").strip()
                    break
            if paragraph_text:
                break

    cleaned = _clean_question(paragraph_text)
    if cleaned and len(cleaned) >= 8 and cleaned.lower() not in ("yes", "no"):
        return cleaned

    # ----- Step 2: table-cell fallback -----------------------------------
    # Reached when the paragraph was empty, too short, or just "Yes".
    # Common case: resource grids with one Yes/No per cell and no question
    # text in the cell itself.
    cell_text = _table_cell_label_for(analyze_result, yes_word)
    if cell_text:
        cleaned_cell = _clean_question(cell_text)
        if cleaned_cell and len(cleaned_cell) >= 4:
            return cleaned_cell
    return cleaned


def _table_cell_label_for(analyze_result: dict, word: dict) -> str:
    """If `word` lives inside a table cell, return the most descriptive label.

    Picks the longer of: the row's first non-empty cell, or the column header
    (row 0 cell at the same column index).
    """
    span = word.get("span") or {}
    offset = span.get("offset")
    if offset is None:
        return ""

    for table in analyze_result.get("tables", []):
        # Find the cell containing this offset
        target_cell = None
        for cell in table.get("cells", []):
            for s in cell.get("spans", []):
                if s.get("offset", 0) <= offset < s.get("offset", 0) + s.get("length", 0):
                    target_cell = cell
                    break
            if target_cell:
                break
        if not target_cell:
            continue

        row_idx = target_cell.get("rowIndex", -1)
        col_idx = target_cell.get("columnIndex", -1)
        cells = table.get("cells", [])

        # Row's first non-empty cell
        row_label = ""
        for c in cells:
            if c.get("rowIndex") == row_idx:
                txt = (c.get("content") or "").strip()
                if txt and txt.lower() not in ("yes", "no"):
                    row_label = txt
                    break

        # Column header (row 0 at same column)
        col_label = ""
        for c in cells:
            if c.get("rowIndex") == 0 and c.get("columnIndex") == col_idx:
                col_label = (c.get("content") or "").strip()
                break

        if row_label and col_label and row_label != col_label:
            return f"{row_label} — {col_label}"
        return row_label or col_label
    return ""


# ---------------------------------------------------------------------------
# CHECKBOX BINDING (Yes/No questions)
# ---------------------------------------------------------------------------

def extract_checkbox_answers(analyze_result: dict) -> tuple[list[dict], set[int]]:
    """Bind high-confidence selection marks to Yes/No labels (pass 1 of 2).

    See the module docstring for the high-level strategy. This implementation
    is split into three logical phases per page:

        Phase A: build (Yes, No) pairs by horizontal proximity on each row.
        Phase B: assign each selection mark to a single pair using horizontal
                 bracketing (right of "No" -> No; between Yes and No -> Yes).
        Phase C: for each pair, decide the answer from its accumulated mark
                 states and emit the result with a question label.

    Returns:
        (answers, used_marks)
            answers     : list of {"page", "question", "answer", "confidence"}
            used_marks  : flat set of mark indexes consumed by this pass —
                          recomputed per-page in `_per_page_used_marks` for
                          the standalone pass to skip.

    Why a tuple? The standalone-checkbox pass needs to know which marks have
    already been claimed so it doesn't double-extract the same checkbox as
    both a Yes/No answer and a bare option.
    """
    answers: list[dict] = []
    used_marks: set[int] = set()

    for page in analyze_result.get("pages", []):
        page_num = page.get("pageNumber")
        yn = _yes_no_words(page)
        marks = [
            m for m in page.get("selectionMarks", [])
            if m.get("confidence", 0) >= _MIN_MARK_CONFIDENCE and m.get("polygon")
        ]
        if not yn or not marks:
            continue

        # ----- Phase A: pair every "yes" with its nearest right-side "no" -
        # On forms like "Did anyone X? Yes ___ No ___ ... Did anyone Y? Yes ___ No ___"
        # multiple pairs may share the same row. Greedy-nearest pairing handles
        # that correctly because the No that follows a Yes is always closer than
        # the No belonging to the *next* question on the same row.
        yes_words = [w for w in yn if w["label"] == "yes"]
        no_words = [w for w in yn if w["label"] == "no"]
        used_no: set[int] = set()
        pairs: list[dict] = []  # each pair: {"yes": word_or_None, "no": word_or_None}

        for yw in yes_words:
            best_no = None
            best_idx = None
            best_gap = _PAIR_MAX_X_GAP
            for j, nw in enumerate(no_words):
                if j in used_no:
                    continue
                if abs(nw["cy"] - yw["cy"]) > _ROW_Y_TOLERANCE:
                    continue
                gap = nw["cx"] - yw["cx"]
                if 0 < gap <= best_gap:
                    best_gap = gap
                    best_no = nw
                    best_idx = j
            if best_idx is not None:
                used_no.add(best_idx)
            pairs.append({"yes": yw, "no": best_no})
        # Lone "No"s become single-label pairs (rare; happens when the OCR
        # missed the "Yes" word or it was clipped at a page edge).
        for j, nw in enumerate(no_words):
            if j not in used_no:
                pairs.append({"yes": None, "no": nw})

        # ----- Phase B: assign each mark to a single pair ------------------
        # pair_states[pair_index] = {
        #   "yes_state": "selected"|"unselected",
        #   "yes_conf":  0.0..1.0,
        #   "no_state":  ...,
        #   "no_conf":   ...,
        # }
        # We track Yes and No independently so a row where both labels have
        # marks (the rare "both selected" case) can still be flagged.
        pair_states: dict[int, dict] = {}

        for mi, m in enumerate(marks):
            mx, my = _poly_center(m["polygon"])
            best_pair_idx = None
            best_dist = _MAX_MARK_TO_WORD
            best_target = None  # "yes" or "no" within the chosen pair

            # Try every pair on this page and keep the one whose anchor is
            # closest horizontally to the mark, subject to the row tolerance.
            for pi, pair in enumerate(pairs):
                yw, nw = pair["yes"], pair["no"]
                ref_y = yw["cy"] if yw else nw["cy"]
                if abs(my - ref_y) > _ROW_Y_TOLERANCE:
                    continue  # different row

                if yw and nw:
                    # Both Yes and No present: use horizontal bracketing.
                    # The 0.05 fudge factor handles marks that sit *exactly*
                    # on top of the label rather than slightly to its right.
                    yes_x, no_x = yw["cx"], nw["cx"]
                    if mx >= no_x - 0.05:
                        target = "no"
                        anchor_x = no_x
                    elif mx >= yes_x - 0.05:
                        target = "yes"
                        anchor_x = yes_x
                    else:
                        # Mark sits to the left of "Yes" — unusual, but
                        # assign by simple nearest-neighbor.
                        if abs(mx - yes_x) < abs(mx - no_x):
                            target, anchor_x = "yes", yes_x
                        else:
                            target, anchor_x = "no", no_x
                elif yw:
                    target, anchor_x = "yes", yw["cx"]
                else:
                    target, anchor_x = "no", nw["cx"]

                dist = abs(mx - anchor_x)
                if dist < best_dist:
                    best_dist = dist
                    best_pair_idx = pi
                    best_target = target

            if best_pair_idx is None:
                continue  # mark didn't belong to any Yes/No pair on this page

            # Record the assignment. If two marks are claimed for the same
            # label (collision), keep the one with the higher confidence.
            used_marks.add(mi)
            ps = pair_states.setdefault(best_pair_idx, {})
            key_state = f"{best_target}_state"
            key_conf = f"{best_target}_conf"
            new_conf = m.get("confidence", 0.0)
            if new_conf >= ps.get(key_conf, 0.0):
                ps[key_state] = m.get("state")
                ps[key_conf] = new_conf

        # ----- Phase C: emit one answer per pair --------------------------
        for pi, pair in enumerate(pairs):
            ps = pair_states.get(pi, {})
            yes_state, yes_conf = ps.get("yes_state"), ps.get("yes_conf", 0.0)
            no_state, no_conf = ps.get("no_state"), ps.get("no_conf", 0.0)

            # Decision matrix:
            #   Yes selected, No not        -> answer = yes
            #   No selected, Yes not        -> answer = no
            #   both selected               -> pick higher confidence + warn
            #   neither has any mark at all -> skip (no answer to report)
            if yes_state == "selected" and no_state != "selected":
                answer, confidence = "yes", yes_conf
            elif no_state == "selected" and yes_state != "selected":
                answer, confidence = "no", no_conf
            elif yes_state == "selected" and no_state == "selected":
                answer = "yes" if yes_conf >= no_conf else "no"
                answer += " (both marked)"  # the UI surfaces this as a warning
                confidence = max(yes_conf, no_conf)
            else:
                continue

            # Use the Yes word as the anchor for question lookup; if there
            # was only a No (lone-No fallback above), use that instead.
            anchor_word = pair["yes"] or pair["no"]
            question = _question_for(analyze_result, anchor_word)
            # Defensive: scrub any stray :selected: tokens that survived.
            question = _clean_question(question)
            answers.append({
                "page": page_num,
                "question": question or "(unlabeled)",
                "answer": answer,
                "confidence": round(confidence, 2),
            })

    return answers, used_marks


# ---------------------------------------------------------------------------
# STANDALONE CHECKBOX BINDING ("check all that apply" lists)
# ---------------------------------------------------------------------------

def extract_standalone_checkboxes(
    analyze_result: dict,
    used_marks_per_page: dict[int, set[int]],
) -> list[dict]:
    """Extract bare-label checkboxes (pass 2 of 2).

    Anything not already consumed by the Yes/No binder is treated as a
    standalone option. For each leftover high-confidence mark we look for the
    nearest text line whose left edge lies just to the right of the mark on
    the same row — that's the option label.

    Output items have a different shape from Yes/No answers:
        {"page": N, "option": "<label>",
         "selected": True|False, "confidence": 0.0..1.0}

    Heavy junk filtering happens here (length, alpha-count, blocklist) because
    bare checkbox detection is more error-prone than Yes/No: a stray glyph or
    underscore can look like a label otherwise.
    """
    out: list[dict] = []
    for page in analyze_result.get("pages", []):
        page_num = page.get("pageNumber")
        used = used_marks_per_page.get(page_num, set())
        marks = [
            (i, m) for i, m in enumerate(page.get("selectionMarks", []))
            if m.get("confidence", 0) >= _MIN_MARK_CONFIDENCE
            and m.get("polygon")
            and i not in used
        ]
        if not marks:
            continue

        lines = page.get("lines", [])
        for _, m in marks:
            mx, my = _poly_center(m["polygon"])
            # Right edge of the mark — labels start *after* this x-coord.
            mark_right = max(m["polygon"][0::2])

            # Primary strategy: the closest line whose left edge is
            # immediately to the right of the mark on the same row.
            best_line = None
            best_gap = _MAX_LABEL_X_GAP
            for line in lines:
                poly = line.get("polygon")
                if not poly:
                    continue
                lx, ly = _poly_center(poly)
                if abs(ly - my) > _ROW_Y_TOLERANCE:
                    continue
                left_x = min(poly[0::2])
                gap = left_x - mark_right
                if 0 <= gap <= best_gap:
                    best_gap = gap
                    best_line = line

            # Fallback: any line on the same row, even if it overlaps the
            # mark horizontally. Useful for very tight layouts.
            if not best_line:
                for line in lines:
                    poly = line.get("polygon")
                    if not poly:
                        continue
                    _, ly = _poly_center(poly)
                    if abs(ly - my) <= _ROW_Y_TOLERANCE:
                        best_line = line
                        break

            label = (best_line.get("content", "").strip() if best_line else "").rstrip("?:")
            label = _clean_question(label)

            # Junk filters. Each is documented at its constant.
            if not label or len(label) < _MIN_OPTION_CHARS:
                continue
            alpha_count = sum(1 for ch in label if ch.isalpha())
            if alpha_count < _MIN_OPTION_ALPHA:
                continue
            if label.lower().strip(".,?! ") in _OPTION_BLOCKLIST:
                continue

            out.append({
                "page": page_num,
                "option": label,
                "selected": m.get("state") == "selected",
                "confidence": round(m.get("confidence", 0.0), 2),
            })
    return out


def _clean_question(text: str) -> str:
    """Normalize question/option text by stripping selection-mark noise.

    Document Intelligence sometimes leaks `:selected:` / `:unselected:` tokens
    into the textual content (an artifact of how it serializes inline marks).
    Those tokens look ugly in the UI and confuse downstream LLMs, so we strip
    them here. We also trim trailing " Yes ... If" residue that survives the
    paragraph-truncation in `_question_for()` when the question wraps oddly.

    Idempotent: running it twice produces the same result.
    """
    if not text:
        return text
    cleaned = (
        text.replace(":selected:", " ")
            .replace(":unselected:", " ")
            .replace(":selected", " ")     # without trailing colon (rare variant)
            .replace(":unselected", " ")
    )
    # Normalize whitespace runs to single spaces.
    cleaned = " ".join(cleaned.split())
    # Drop trailing " Yes ... If" / " Yes ... No" boilerplate — only when
    # there's enough text *before* the marker to keep the question intact.
    lower = cleaned.lower()
    for marker in (" yes ", " yes,", " yes:"):
        idx = lower.rfind(marker)
        if idx > 20:
            cleaned = cleaned[:idx]
            break
    return cleaned.strip().rstrip("?:.,").strip()


# ---------------------------------------------------------------------------
# TABLE SUMMARY
# ---------------------------------------------------------------------------

def summarize_tables(analyze_result: dict) -> list[dict]:
    """Return every table as a full grid plus a key/value projection.

    For each table in the document we emit:
        - `cells`      : 2D array (rowCount × columnCount) of cell strings.
        - `key_values` : dict mapping each row's first cell to the remaining
                         cells joined with " | ". Rows with all-blank trailing
                         cells are skipped, so blank template rows are omitted.

    The key/value view is what most consumers actually want for forms:
        "Client Name"  ->  "Jane Doe"
        "DOB"          ->  "01/01/2000"
        "Cash"         ->  "$ 100 | HOUSE"

    Works on any table the OCR detects — invoices, line-item tables, resource
    grids, data dictionaries.
    """
    out: list[dict] = []
    for t in analyze_result.get("tables", []):
        rows = t.get("rowCount", 0)
        cols = t.get("columnCount", 0)
        page = (t.get("boundingRegions") or [{}])[0].get("pageNumber")
        grid: list[list[str]] = [["" for _ in range(cols)] for _ in range(rows)]
        for cell in t.get("cells", []):
            r = cell.get("rowIndex", 0)
            c = cell.get("columnIndex", 0)
            if r < rows and c < cols:
                grid[r][c] = cell.get("content", "").strip()

        key_values: dict[str, str] = {}
        for row in grid:
            if not row:
                continue
            key = row[0].strip()
            values = [v for v in (cell.strip() for cell in row[1:]) if v]
            if key and values:
                key_values[key] = " | ".join(values)

        out.append({
            "page": page,
            "rows": rows,
            "columns": cols,
            "cells": grid,
            "key_values": key_values,
        })
    return out


# ---------------------------------------------------------------------------
# TOP-LEVEL ENTRY POINTS
# ---------------------------------------------------------------------------

def summarize_layout(layout: dict) -> dict:
    """Build the full human-readable summary for a `prebuilt-layout` response.

    Orchestrates the two checkbox passes plus handwriting and table extraction.
    See the module docstring for the output shape.
    """
    ar = _ar(layout)
    checkbox_answers, used_marks = extract_checkbox_answers(ar)

    # `extract_checkbox_answers` returns a flat set of mark indexes, but the
    # standalone pass needs per-page sets. We recompute them by re-walking the
    # pages with the same membership rule. This is cheap and keeps the two
    # functions decoupled.
    used_per_page = _per_page_used_marks(ar)

    return {
        "model": ar.get("modelId"),
        "api_version": ar.get("apiVersion"),
        "page_count": len(ar.get("pages", [])),
        "handwriting": extract_handwriting(ar),
        "checkboxes": checkbox_answers,
        "options": extract_standalone_checkboxes(ar, used_per_page),
        "tables": summarize_tables(ar),
    }


def _per_page_used_marks(analyze_result: dict) -> dict[int, set[int]]:
    """Recompute which mark indexes per page were consumed by the Yes/No binder.

    This is intentionally a separate function from `extract_checkbox_answers`
    so the standalone-checkbox pass can run independently. It uses a slightly
    *broader* membership rule (any mark within row tolerance and label-distance
    of any Yes/No word counts as "consumed") which prevents the standalone
    pass from accidentally re-emitting Yes/No marks as bare options.
    """
    used_per_page: dict[int, set[int]] = {}
    for page in analyze_result.get("pages", []):
        page_num = page.get("pageNumber")
        yn = _yes_no_words(page)
        marks = [
            (i, m) for i, m in enumerate(page.get("selectionMarks", []))
            if m.get("confidence", 0) >= _MIN_MARK_CONFIDENCE and m.get("polygon")
        ]
        if not yn or not marks:
            used_per_page[page_num] = set()
            continue

        used: set[int] = set()
        # Any high-confidence mark within row tolerance and horizontal range
        # of *any* Yes/No word counts as consumed for the standalone pass.
        for i, m in marks:
            mx, my = _poly_center(m["polygon"])
            for w in yn:
                if abs(w["cy"] - my) > _ROW_Y_TOLERANCE:
                    continue
                if abs(w["cx"] - mx) <= _MAX_MARK_TO_WORD:
                    used.add(i)
                    break
        used_per_page[page_num] = used
    return used_per_page


def summarize_read(layout: dict) -> dict:
    """Compact summary for `prebuilt-read` (OCR-only) responses.

    `prebuilt-read` returns text + handwriting + (optionally) language detection,
    but no tables or selection marks. This summary is therefore much simpler:
    one entry per page giving the reading-order text, plus the same handwriting
    extraction we do for layout responses.

    Output:
        {
          "model":       "prebuilt-read",
          "api_version": "...",
          "page_count":  N,
          "handwriting": [...],
          "languages":   ["en", ...],
          "pages":       [{"page": 1, "line_count": 42,
                          "content": "line 1\\nline 2\\n..."}, ...],
          "full_text":   "entire document content as one string"
        }

    Note: there are NO `checkboxes` or `options` keys here — read mode doesn't
    detect selection marks. Callers using the dispatcher don't need to care
    about this distinction.
    """
    ar = _ar(layout)
    pages_out: list[dict] = []
    for page in ar.get("pages", []):
        lines = [l.get("content", "").strip() for l in page.get("lines", [])]
        lines = [l for l in lines if l]
        pages_out.append({
            "page": page.get("pageNumber"),
            "line_count": len(lines),
            "content": "\n".join(lines),
        })

    languages = sorted({
        loc.get("locale") for loc in ar.get("languages", []) if loc.get("locale")
    })

    return {
        "model": ar.get("modelId"),
        "api_version": ar.get("apiVersion"),
        "page_count": len(ar.get("pages", [])),
        "handwriting": extract_handwriting(ar),
        "languages": languages,
        "pages": pages_out,
        "full_text": ar.get("content", ""),
    }


def summarize(layout: dict) -> dict:
    """Dispatch to the right summarizer based on the model that was used.

    This is the recommended entry point for app code. Callers don't need to
    know whether the response came from `prebuilt-layout` or `prebuilt-read`;
    the dispatcher inspects `analyzeResult.modelId` and picks the right
    summarizer. Anything other than `prebuilt-read` is treated as layout.
    """
    ar = _ar(layout)
    model = (ar.get("modelId") or "").lower()
    if model == "prebuilt-read":
        return summarize_read(layout)
    return summarize_layout(layout)
