"""
Greek Anaphora, Epiphora & Word-Repetition Detector
=====================================================
Three detection modes:
  • Anaphora       — same phrase at the START of consecutive verse lines
  • Epiphora       — same phrase at the END of consecutive verse lines
  • Word Repetition — same phrase at the START of any clause (delimited by
                      punctuation: . · ; , and newlines), searching across a
                      configurable window of consecutive clauses

Architecture mirrors the hiatus-detector: runs inside Pyodide; options are
passed via /options.json written by app.js.

Elision fix (v2)
----------------
The elision check is now done on the RAW word BEFORE any diacritic stripping
so that apostrophe codepoints are never accidentally removed by NFD
decomposition. The stripped stem is then passed into the normal pipeline.
"""

import unicodedata
import html as html_mod
import csv
import json
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Greek stop-words
# ---------------------------------------------------------------------------

STOP_WORDS_NFC = {
    # articles
    "ὁ","ἡ","τό","οἱ","αἱ","τά",
    "τοῦ","τῆς","τοῖς","ταῖς","τούς","τάς","τῶν","τῷ","τήν","τόν",
    # conjunctions & particles
    "καί","καὶ","δέ","δὲ","γάρ","γὰρ","μέν","μὲν","ἀλλά","ἀλλὰ",
    "ἀλλ","οὖν","ἄρα","ἆρα","ἤ","ἢ","οὐδέ","οὐδὲ","μηδέ","μηδὲ",
    "τε","γε","τοί","που","πού","νυν","νῦν","αὖ","αὖτε","αὖθις",
    "εἰ","εἴ","ὡς","ὥς","ὅτι","ὅτε","ὅτ","ἐπεί","ἐπεὶ","ἐπειδή",
    "ἐπειδὴ","ὄτε","ὄτι","ἵνα","ὄφρα","ὄφρ","ὁτε","ὁτι",
    # negations
    "οὐ","οὐκ","οὐχ","οὔ","μή","μὴ","μήτε","μήτ","οὔτε","οὔτ",
    # prepositions
    "ἐν","ἐκ","ἐξ","εἰς","ἐς","πρός","πρὸς","ἀπό","ἀπὸ","ὑπό","ὑπὸ",
    "ἐπί","ἐπὶ","περί","περὶ","παρά","παρὰ","κατά","κατὰ","διά","διὰ",
    "μετά","μετὰ","ἀντί","ἀντὶ","ἀμφί","ἀμφὶ",
    # pronouns
    "αὐτός","αὐτὸς","αὐτή","αὐτὴ","αὐτό","αὐτὸ",
    "αὐτοῦ","αὐτῆς","αὐτοῖς","αὐταῖς","αὐτούς","αὐτάς","αὐτῶν","αὐτῷ",
}

# Apostrophe codepoints used in Greek texts for elision
_APOSTROPHES = {"'", "\u2019", "\u02bc", "\u0027", "\u02b9"}

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_word(word: str, opts: dict) -> str:
    """
    Return a normalised comparison key for `word`.

    Elision is handled FIRST on the raw string before NFD decomposition,
    so apostrophe codepoints are never accidentally stripped by diacritic
    removal. The elided stem is then passed through the rest of the pipeline.

    Pipeline:
      1. Elision check on raw word (strip trailing apostrophe + placeholder)
      2. NFD + lowercase
      3. Strip accents   (U+0301 acute, U+0300 grave, U+0342 circumflex)
      4. Strip breathings (U+0313 smooth, U+0314 rough)
      5. Strip iota subscript (U+0345)
      6. Nu-movable: strip final ν when preceded by ι or ε
      7. NFC
    """
    w = word

    # Step 1 — elision (must precede NFD so apostrophes survive)
    if opts.get("handle_elision", False):
        if w and w[-1] in _APOSTROPHES:
            w = w[:-1]
            # We deliberately do NOT append a placeholder vowel:
            # stripping the apostrophe lets ἀλλ' and ἀλλά both normalise to
            # "αλλ" (after accent stripping), which is the correct comparison key.

    # Step 2 — NFD + lowercase
    w = unicodedata.normalize("NFD", w.lower())

    # Step 3 — accents
    if opts.get("strip_accents", True):
        w = "".join(c for c in w if c not in {"\u0301", "\u0300", "\u0342"})

    # Step 4 — breathings
    if opts.get("strip_breathings", True):
        w = "".join(c for c in w if c not in {"\u0313", "\u0314"})

    # Step 5 — iota subscript
    if opts.get("strip_iota_subscript", False):
        w = "".join(c for c in w if c != "\u0345")

    # Step 6 — NFC
    w = unicodedata.normalize("NFC", w)

    # Step 7 — nu movable
    if opts.get("handle_nu_movable", False):
        if len(w) >= 2 and w[-1] == "ν" and w[-2] in ("ι", "ε"):
            w = w[:-1]

    return w


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

# Punctuation characters that delimit clauses (for word-repetition mode)
_CLAUSE_PUNCT = set(".,·;:\n")

# Punctuation to strip from word edges when building tokens
_WORD_PUNCT = ".,·;:!?\"''\u2019\u02bc—–\u2013\u2014-()[]⟨⟩"


def tokenize_text(text: str) -> list[dict]:
    """
    Tokenise `text` into a flat list of token dicts, each carrying:
        text        – original word string (punct stripped from edges)
        raw         – original whitespace-separated chunk
        norm        – filled in later by caller
        char_start  – byte/char offset in `text` (for highlighting)
        clause_break_before – True if this token begins a new clause
        line_num    – 1-indexed line number
    """
    tokens = []
    pos = 0
    line_num = 1
    pending_break = True   # very first token starts a clause

    for raw_chunk in re.split(r'(\s+)', text):
        if not raw_chunk:
            pos += len(raw_chunk)
            continue

        # Count newlines in whitespace chunks
        if re.match(r'^\s+$', raw_chunk):
            nl_count = raw_chunk.count('\n')
            line_num += nl_count
            if nl_count:
                pending_break = True
            pos += len(raw_chunk)
            continue

        # Strip word-level punctuation
        word = raw_chunk.strip(_WORD_PUNCT)
        if not word:
            # Check if the chunk itself ends with clause punct
            if raw_chunk and raw_chunk[-1] in _CLAUSE_PUNCT:
                pending_break = True
            pos += len(raw_chunk)
            continue

        # Did the chunk contain a clause-boundary character?
        chunk_has_clause_break = any(c in _CLAUSE_PUNCT for c in raw_chunk
                                     if c not in " \t")

        tokens.append({
            "text": word,
            "raw": raw_chunk,
            "norm": "",          # filled in by caller
            "char_start": pos,
            "clause_break_before": pending_break,
            "line_num": line_num,
        })

        # If this chunk ends with clause-punctuation, next token starts a new clause
        stripped_right = raw_chunk.rstrip()
        if stripped_right and stripped_right[-1] in _CLAUSE_PUNCT:
            pending_break = True
        else:
            pending_break = False

        pos += len(raw_chunk)

    return tokens


def tokenize_line(line: str, line_char_offset: int = 0) -> list[dict]:
    """Tokenise a single line, carrying char_start offsets (relative to full text)."""
    tokens = []
    pos = 0
    for raw in re.split(r'(\s+)', line):
        if not raw:
            pos += len(raw)
            continue
        if re.match(r'^\s+$', raw):
            pos += len(raw)
            continue
        word = raw.strip(_WORD_PUNCT)
        if word:
            # char_start: offset of the word within raw chunk + chunk offset in line
            word_offset_in_raw = raw.find(word)
            tokens.append({
                "text": word,
                "raw": raw,
                "norm": "",
                "char_start": line_char_offset + pos + word_offset_in_raw,
            })
        pos += len(raw)
    return tokens


# ---------------------------------------------------------------------------
# Options reader
# ---------------------------------------------------------------------------

def read_options() -> dict:
    defaults = {
        "detect_anaphora":       True,
        "detect_epiphora":       True,
        "detect_word_repetition": False,
        "phrase_length":         1,
        "distance_window":       2,
        "min_occurrences":       2,
        "strip_accents":         True,
        "strip_breathings":      True,
        "strip_iota_subscript":  False,
        "handle_elision":        False,
        "handle_nu_movable":     False,
        "skip_stopwords":        True,
    }
    opt_path = Path("/options.json")
    if opt_path.exists():
        try:
            data = json.loads(opt_path.read_text(encoding="utf-8"))
            for k in defaults:
                if k in data:
                    defaults[k] = data[k]
        except Exception:
            pass
    return defaults


# ---------------------------------------------------------------------------
# Stop-word helpers
# ---------------------------------------------------------------------------

def _is_stopword(norm_key: str, norm_stops: set) -> bool:
    return norm_key in norm_stops


def _skip_stops_start(toks, n, norm_stops, skip):
    """Return list of (index, token) for the first n non-stop tokens."""
    result = []
    for i, t in enumerate(toks):
        if skip and _is_stopword(t["norm"], norm_stops):
            continue
        result.append((i, t))
        if len(result) == n:
            break
    return result


def _skip_stops_end(toks, n, norm_stops, skip):
    """Return list of (index, token) for the last n non-stop tokens (in order)."""
    result = []
    for i in range(len(toks) - 1, -1, -1):
        t = toks[i]
        if skip and _is_stopword(t["norm"], norm_stops):
            continue
        result.append((i, t))
        if len(result) == n:
            break
    result.reverse()
    return result


def phrase_key(indexed_toks: list) -> str:
    return " ".join(t["norm"] for _, t in indexed_toks)


# ---------------------------------------------------------------------------
# Group-finding (shared by all three modes)
# ---------------------------------------------------------------------------

def find_groups(items, window, min_occ, kind):
    """
    items : list of (unit_index, key, indexed_phrase_toks, had_stop, line_num)
    window: max gap (in unit indices) between first and last member
    Yields group dicts.
    """
    used = set()
    groups = []
    n = len(items)

    for i in range(n):
        idx_i, key_i, phrase_i, stop_i, lnum_i = items[i]
        if not key_i or idx_i in used:
            continue

        group = [i]
        for j in range(i + 1, n):
            idx_j, key_j, phrase_j, stop_j, lnum_j = items[j]
            if idx_j - idx_i > window - 1:
                break
            if key_j == key_i:
                group.append(j)

        if len(group) < min_occ:
            continue

        groups.append({
            "kind": kind,
            "key": key_i,
            "line_nums":    [items[g][4] for g in group],
            "phrases":      [[t for _, t in items[g][2]] for g in group],
            "had_stops":    [items[g][3] for g in group],
            "unit_indices": [items[g][0] for g in group],
        })
        for g in group:
            used.add(items[g][0])

    return groups


# ---------------------------------------------------------------------------
# HTML-building helpers
# ---------------------------------------------------------------------------

def _build_line_html(raw_line: str, toks: list, highlight_indices: dict) -> str:
    """
    Rebuild a raw line as HTML, wrapping tokens whose list-index appears in
    `highlight_indices` (a dict mapping tok_index → css_class string).
    Uses find() scanning so the original spacing and punctuation are preserved.
    """
    parts = []
    remaining = raw_line
    for i, t in enumerate(toks):
        tok_text = t["text"]
        idx = remaining.find(tok_text)
        if idx == -1:
            continue
        parts.append(html_mod.escape(remaining[:idx]))
        escaped = html_mod.escape(tok_text)
        css = highlight_indices.get(i)
        if css:
            parts.append(f'<span class="{css}">{escaped}</span>')
        else:
            parts.append(escaped)
        remaining = remaining[idx + len(tok_text):]
    parts.append(html_mod.escape(remaining))
    return "".join(parts)


# ---------------------------------------------------------------------------
# Main detection entry point
# ---------------------------------------------------------------------------

def detect_repetitions(text: str):
    opts         = read_options()
    phrase_length = max(1, int(opts.get("phrase_length", 1)))
    window        = max(2, int(opts.get("distance_window", 2)))
    min_occ       = max(2, int(opts.get("min_occurrences", 2)))
    detect_ana    = opts.get("detect_anaphora", True)
    detect_epi    = opts.get("detect_epiphora", True)
    detect_wr     = opts.get("detect_word_repetition", False)
    skip          = opts.get("skip_stopwords", True)

    norm_stops = {normalize_word(w, opts) for w in STOP_WORDS_NFC}

    # ── Line-level data (for anaphora / epiphora) ───────────────────────────
    raw_lines = text.split("\n")
    line_data = []
    char_offset = 0
    for ln_idx, raw_line in enumerate(raw_lines):
        toks = tokenize_line(raw_line, line_char_offset=char_offset)
        for t in toks:
            t["norm"] = normalize_word(t["text"], opts)
        line_data.append({
            "line_num": ln_idx + 1,
            "raw": raw_line,
            "tokens": toks,
        })
        char_offset += len(raw_line) + 1  # +1 for the \n

    occurrences = []

    # ── Anaphora ────────────────────────────────────────────────────────────
    if detect_ana:
        items = []
        for i, ld in enumerate(line_data):
            toks = ld["tokens"]
            indexed = _skip_stops_start(toks, phrase_length, norm_stops, skip)
            had_stop = (len(indexed) > 0 and indexed[0][0] > 0)
            key = phrase_key(indexed) if len(indexed) == phrase_length else ""
            items.append((i, key, indexed, had_stop, ld["line_num"]))
        occurrences += find_groups(items, window, min_occ, "anaphora")

    # ── Epiphora ────────────────────────────────────────────────────────────
    if detect_epi:
        items = []
        for i, ld in enumerate(line_data):
            toks = ld["tokens"]
            indexed = _skip_stops_end(toks, phrase_length, norm_stops, skip)
            last_content_idx = indexed[-1][0] if indexed else -1
            had_stop = (last_content_idx < len(toks) - 1) if indexed else False
            key = phrase_key(indexed) if len(indexed) == phrase_length else ""
            items.append((i, key, indexed, had_stop, ld["line_num"]))
        occurrences += find_groups(items, window, min_occ, "epiphora")

    # ── Word Repetition ─────────────────────────────────────────────────────
    # Clause-level: split by punctuation (., ·, ;, ,) and newlines.
    # For each clause, extract its opening phrase (first N non-stop words)
    # and detect repetition across consecutive clauses within the window.
    wr_highlight = {}   # maps (line_num, tok_text, char_start) → css_class
    if detect_wr:
        all_toks = tokenize_text(text)
        for t in all_toks:
            t["norm"] = normalize_word(t["text"], opts)

        # Split into clauses: a new clause starts at clause_break_before=True
        clauses = []
        current = []
        for t in all_toks:
            if t["clause_break_before"] and current:
                clauses.append(current)
                current = []
            current.append(t)
        if current:
            clauses.append(current)

        # Build items list for find_groups
        items = []
        for ci, clause in enumerate(clauses):
            indexed = _skip_stops_start(clause, phrase_length, norm_stops, skip)
            had_stop = (len(indexed) > 0 and indexed[0][0] > 0)
            key = phrase_key(indexed) if len(indexed) == phrase_length else ""
            # line_num: use the line of the first token
            lnum = clause[0]["line_num"] if clause else 0
            items.append((ci, key, indexed, had_stop, lnum))

        wr_groups = find_groups(items, window, min_occ, "word_repetition")
        occurrences += wr_groups

        # Record which tokens need highlighting in each line
        for grp in wr_groups:
            for phrase_toks in grp["phrases"]:
                for t in phrase_toks:
                    key_wr = (t["line_num"], t["char_start"])
                    wr_highlight[key_wr] = "rep-wordrepeat"

    # ── Build per-line highlight index maps ─────────────────────────────────
    # Map: line_num → { tok_list_index → css_class }
    line_highlights = {}   # line_num → dict(tok_index → css)

    def add_highlight(line_num, tok_index, css):
        lh = line_highlights.setdefault(line_num, {})
        existing = lh.get(tok_index, "")
        if existing and existing != css:
            lh[tok_index] = existing + " " + css
        else:
            lh[tok_index] = css

    # Anaphora / epiphora occurrences
    for occ in occurrences:
        if occ["kind"] not in ("anaphora", "epiphora"):
            continue
        css = "rep-anaphora" if occ["kind"] == "anaphora" else "rep-epiphora"
        # unit_indices are line_data indices (0-based)
        for ui, phrase_toks in zip(occ["unit_indices"], occ["phrases"]):
            ld = line_data[ui]
            full_toks = ld["tokens"]
            # phrase_toks are token dicts; find their indices in full_toks
            phrase_texts = {id(t): t for t in phrase_toks}
            for ti, ft in enumerate(full_toks):
                if id(ft) in phrase_texts:
                    add_highlight(ld["line_num"], ti, css)

    # Word-repetition occurrences — use char_start for cross-line matching
    # We need to map back: for each line, which token char_starts are highlighted?
    wr_char_starts = {}   # line_num → set of char_start values to highlight
    for grp in occurrences:
        if grp["kind"] != "word_repetition":
            continue
        for phrase_toks in grp["phrases"]:
            for t in phrase_toks:
                wr_char_starts.setdefault(t["line_num"], set()).add(t["char_start"])

    # ── Render annotated HTML line by line ───────────────────────────────────
    html_lines = []
    for ld in line_data:
        ln = ld["line_num"]
        raw = ld["raw"]
        toks = ld["tokens"]

        # Merge anaphora/epiphora highlights
        hi = dict(line_highlights.get(ln, {}))

        # Merge word-repetition highlights (match by char_start)
        if ln in wr_char_starts:
            wr_starts = wr_char_starts[ln]
            for ti, t in enumerate(toks):
                if t.get("char_start") in wr_starts:
                    existing = hi.get(ti, "")
                    hi[ti] = (existing + " rep-wordrepeat").strip() if existing else "rep-wordrepeat"

        if not hi:
            html_lines.append(html_mod.escape(raw))
        else:
            html_lines.append(_build_line_html(raw, toks, hi))

    annotated_html = "\n".join(html_lines)

    # Number occurrences
    for n, occ in enumerate(occurrences, 1):
        occ["index"] = n

    return annotated_html, occurrences


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Greek Rhetorical Repetition Highlights</title>
<style>
body {{ font-family: serif; padding: 1.5rem; max-width: 900px; margin: auto; }}
pre.source {{ white-space: pre-wrap; font-size: 18px; line-height: 1.7; }}
.rep-anaphora   {{ background: rgba(212,160,23,0.35); border-bottom: 2px solid #b8860b; }}
.rep-epiphora   {{ background: rgba(30,160,160,0.30); border-bottom: 2px solid #1a8a8a; }}
.rep-wordrepeat {{ background: rgba(160,60,180,0.25); border-bottom: 2px solid #8a30a8; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1.5rem; font-size: 0.95rem; }}
td,th {{ border: 1px solid #bbb; padding: 6px 10px; vertical-align: top; }}
th {{ background: #f4f4f4; }}
</style>
</head>
<body>
<h1>Greek Rhetorical Repetition Highlights</h1>
<p>
  <span style="background:rgba(212,160,23,0.35);padding:2px 6px;">Gold</span> = Anaphora (A) &nbsp;
  <span style="background:rgba(30,160,160,0.30);padding:2px 6px;">Teal</span> = Epiphora (E) &nbsp;
  <span style="background:rgba(160,60,180,0.25);padding:2px 6px;">Violet</span> = Word Repetition (W)
</p>
<h2>Annotated Text</h2>
<pre class="source">{annotated}</pre>
<h2>Occurrences</h2>
<table>
<tr><th>#</th><th>Type</th><th>Repeated Phrase</th><th>Lines</th><th>Stop skipped?</th></tr>
{rows}
</table>
</body></html>
"""

_KIND_SHORT = {"anaphora": "A", "epiphora": "E", "word_repetition": "W"}


def write_outputs(annotated: str, occurrences: list, html_path, csv_path):
    import html as html_mod2
    rows = []
    csv_rows = []
    for occ in occurrences:
        n          = occ["index"]
        kind_short = _KIND_SHORT.get(occ["kind"], "?")
        lines_str  = ", ".join(str(ln) for ln in occ["line_nums"])
        phrase_display = html_mod2.escape(
            " ".join(t["text"] for t in occ["phrases"][0]) if occ["phrases"] else occ["key"]
        )
        had_stop = "yes" if any(occ["had_stops"]) else "no"
        rows.append(
            f"<tr><td>{n}</td><td>{kind_short}</td>"
            f"<td>{phrase_display}</td><td>{lines_str}</td>"
            f"<td>{had_stop}</td></tr>"
        )
        csv_rows.append({
            "index": n,
            "type": kind_short,
            "phrase_normalised": occ["key"],
            "phrase_original": " ".join(t["text"] for t in occ["phrases"][0]) if occ["phrases"] else "",
            "lines": lines_str,
            "stop_skipped": had_stop,
        })

    html_text = HTML_TEMPLATE.format(annotated=annotated, rows="\n".join(rows))
    Path(html_path).write_text(html_text, encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index","type","phrase_normalised",
                                                "phrase_original","lines","stop_skipped"])
        writer.writeheader()
        writer.writerows(csv_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process(input_path: str, html_path: str, csv_path: str):
    text = Path(input_path).read_text(encoding="utf-8")
    annotated, occurrences = detect_repetitions(text)
    write_outputs(annotated, occurrences, html_path, csv_path)
    return occurrences
