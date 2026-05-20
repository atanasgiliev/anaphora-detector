"""
Greek Anaphora & Epiphora Detector
===================================
Detects repetition of words/phrases at the beginnings (anaphora) and
ends (epiphora) of consecutive verse lines in Ancient Greek texts.

Architecture mirrors the hiatus-detector: runs inside Pyodide in the
browser; options are passed via /options.json written by app.js.
"""

import unicodedata
import html
import csv
import json
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Greek stop-words (particles, articles, conjunctions, common pronouns)
# These are "transparent" for anaphora/epiphora detection: if a line begins
# or ends with one of these, the detector looks past it for the content word.
# Stored as NFC strings; comparison uses normalised forms.
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
    # prepositions (short ones most likely to be line-initial noise)
    "ἐν","ἐκ","ἐξ","εἰς","ἐς","πρός","πρὸς","ἀπό","ἀπὸ","ὑπό","ὑπὸ",
    "ἐπί","ἐπὶ","περί","περὶ","παρά","παρὰ","κατά","κατὰ","διά","διὰ",
    "μετά","μετὰ","ἀντί","ἀντὶ","ἀμφί","ἀμφὶ",
    # pronouns
    "αὐτός","αὐτὸς","αὐτή","αὐτὴ","αὐτό","αὐτὸ",
    "αὐτοῦ","αὐτῆς","αὐτοῖς","αὐταῖς","αὐτούς","αὐτάς","αὐτῶν","αὐτῷ",
}

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _strip_combining(nfd_str, categories):
    """Remove combining characters whose unicodedata.category is in `categories`."""
    return "".join(c for c in nfd_str if unicodedata.category(c) not in categories)


def normalize_word(word: str, opts: dict) -> str:
    """
    Return a normalised comparison key for `word`.

    Pipeline (each step is opt-in via opts):
      1. NFC → NFD
      2. Lowercase
      3. Strip accents  (combining acute U+0301, grave U+0300, circumflex U+0342)
      4. Strip breathings (smooth U+0313, rough U+0314)
      5. Strip iota subscript (U+0345)
      6. Handle elision: trailing apostrophe → restore likely elided vowel
      7. Handle nu-movable: strip final ν if preceded by ι or ε
      8. Back to NFC
    """
    w = unicodedata.normalize("NFD", word.lower())

    if opts.get("strip_accents", True):
        # acute, grave, circumflex (Greek polytonic circumflex is U+0342)
        w = "".join(c for c in w if c not in {"\u0301", "\u0300", "\u0342"})

    if opts.get("strip_breathings", True):
        w = "".join(c for c in w if c not in {"\u0313", "\u0314"})

    if opts.get("strip_iota_subscript", False):
        w = "".join(c for c in w if c != "\u0345")

    # Remaining combining marks that are purely diacritical (category Mn)
    # are stripped together with the above; make sure we don't strip
    # diaeresis U+0308 here as it is phonemically significant (ϊ ≠ ι+ι).

    w = unicodedata.normalize("NFC", w)

    if opts.get("handle_elision", False):
        # Strip trailing elision markers and restore a vowel.
        # Greek elision apostrophe: U+2019 ' or plain '
        if w and w[-1] in ("'", "\u2019", "\u02bc"):
            w = w[:-1] + "α"   # placeholder final vowel for key comparison

    if opts.get("handle_nu_movable", False):
        # Strip movable ν: word ends in ν, penultimate base letter is ι or ε
        if len(w) >= 2 and w[-1] == "ν" and w[-2] in ("ι", "ε"):
            w = w[:-1]

    return w


def tokenize_line(line: str) -> list[dict]:
    """
    Split a line into word tokens. Each token is a dict:
        text  – original Unicode string
        key   – placeholder; filled in after normalization
        start – char offset within line (for highlighting, future use)
    Punctuation attached to a word is stripped for the token text.
    """
    PUNCT = set(".,·;:!?\"''\u2019\u02bc—–-()[]⟨⟩")
    tokens = []
    for raw in line.split():
        stripped = raw.strip("".join(PUNCT))
        if stripped:
            tokens.append({"text": stripped, "raw": raw})
    return tokens


# ---------------------------------------------------------------------------
# Options reader (mirrors hiatus detector pattern)
# ---------------------------------------------------------------------------

def read_options() -> dict:
    defaults = {
        # Detection scope
        "detect_anaphora": True,
        "detect_epiphora": True,
        # Phrase length: number of words to match (1–4)
        "phrase_length": 1,
        # Distance window: match within N consecutive lines
        "distance_window": 2,
        # Normalization
        "strip_accents": True,
        "strip_breathings": True,
        "strip_iota_subscript": False,
        "handle_elision": False,
        "handle_nu_movable": False,
        # Stop-word transparency
        "skip_stopwords": True,
        # Minimum number of occurrences to report
        "min_occurrences": 2,
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
# Core detection
# ---------------------------------------------------------------------------

def _is_stopword(norm_key: str, norm_stops: set) -> bool:
    return norm_key in norm_stops


def extract_phrase(tokens: list[dict], position: str, phrase_length: int,
                   opts: dict, norm_stops: set) -> tuple[list[dict], bool]:
    """
    Return (phrase_tokens, had_leading_stop) for `position` in {"start","end"}.

    If skip_stopwords is on:
      - For "start": skip leading stop-words, then take `phrase_length` tokens.
      - For "end":   skip trailing stop-words, then take `phrase_length` tokens.
    Returns the list of tokens forming the phrase, and whether a stop was skipped.
    """
    if not tokens:
        return [], False

    skip = opts.get("skip_stopwords", True)
    toks = list(tokens)  # copy

    had_stop = False
    if position == "start":
        if skip:
            while toks and _is_stopword(toks[0]["norm"], norm_stops):
                toks = toks[1:]
                had_stop = True
        phrase = toks[:phrase_length]
    else:  # "end"
        if skip:
            while toks and _is_stopword(toks[-1]["norm"], norm_stops):
                toks = toks[:-1]
                had_stop = True
        phrase = toks[-phrase_length:] if len(toks) >= phrase_length else toks

    return phrase, had_stop


def phrase_key(phrase_tokens: list[dict]) -> str:
    return " ".join(t["norm"] for t in phrase_tokens)


def detect_repetitions(text: str) -> tuple[str, list[dict]]:
    """
    Main entry point.

    Returns:
        annotated_html – the full text with <span> tags for anaphora/epiphora
        occurrences    – list of occurrence dicts for CSV/table output
    """
    opts = read_options()
    phrase_length  = max(1, int(opts.get("phrase_length", 1)))
    window         = max(2, int(opts.get("distance_window", 2)))
    min_occ        = max(2, int(opts.get("min_occurrences", 2)))
    detect_ana     = opts.get("detect_anaphora", True)
    detect_epi     = opts.get("detect_epiphora", True)

    # Pre-normalise stop-word set
    norm_stops = {
        normalize_word(w, opts) for w in STOP_WORDS_NFC
    }

    lines = text.split("\n")
    # Tokenise every line
    line_data = []
    for ln, line in enumerate(lines):
        toks = tokenize_line(line)
        for t in toks:
            t["norm"] = normalize_word(t["text"], opts)
        line_data.append({
            "line_num": ln + 1,   # 1-indexed
            "raw": line,
            "tokens": toks,
        })

    # For each line, extract start-phrase and end-phrase keys
    for ld in line_data:
        toks = ld["tokens"]
        start_phrase, start_had_stop = extract_phrase(toks, "start", phrase_length, opts, norm_stops)
        end_phrase,   end_had_stop   = extract_phrase(toks, "end",   phrase_length, opts, norm_stops)
        ld["start_phrase"]     = start_phrase
        ld["start_key"]        = phrase_key(start_phrase)
        ld["start_had_stop"]   = start_had_stop
        ld["end_phrase"]       = end_phrase
        ld["end_key"]          = phrase_key(end_phrase)
        ld["end_had_stop"]     = end_had_stop

    # -----------------------------------------------------------------------
    # Find anaphora: groups of consecutive lines (within window) sharing
    # the same start_key.
    # -----------------------------------------------------------------------
    # We collect all (line_idx, key) pairs and group by consecutive runs.

    occurrences = []   # final list

    def find_groups(line_indices_keys, kind):
        """
        Given a list of (line_idx, phrase_tokens, had_stop) tuples,
        find groups where the same key appears within `window` consecutive lines.
        Returns list of group dicts.
        """
        # Group into consecutive windows sharing the same key
        # Strategy: sliding window of size `window`; any key appearing
        # >= min_occ times within those lines constitutes a match.

        groups = []
        n = len(line_indices_keys)
        used = set()  # line indices already assigned to a group

        for i in range(n):
            idx_i, key_i, phrase_i, stop_i = line_indices_keys[i]
            if not key_i:
                continue
            if idx_i in used:
                continue

            # Collect all lines within the window that share key_i
            group_lines = [i]
            for j in range(i + 1, n):
                idx_j, key_j, phrase_j, stop_j = line_indices_keys[j]
                # Within distance window?
                if idx_j - idx_i > window - 1:
                    break
                if key_j == key_i:
                    group_lines.append(j)

            if len(group_lines) < min_occ:
                continue

            member_line_nums = [line_indices_keys[g][0] + 1 for g in group_lines]  # 1-indexed
            member_phrases   = [line_indices_keys[g][2] for g in group_lines]
            member_stops     = [line_indices_keys[g][3] for g in group_lines]
            member_raw_texts = [line_data[line_indices_keys[g][0]]["raw"] for g in group_lines]

            groups.append({
                "kind": kind,
                "key": key_i,
                "line_nums": member_line_nums,
                "phrases": member_phrases,
                "had_stops": member_stops,
                "raw_texts": member_raw_texts,
                "line_indices": [line_indices_keys[g][0] for g in group_lines],
            })
            for g in group_lines:
                used.add(line_indices_keys[g][0])

        return groups

    if detect_ana:
        ana_input = [
            (i, ld["start_key"], ld["start_phrase"], ld["start_had_stop"])
            for i, ld in enumerate(line_data)
        ]
        occurrences += find_groups(ana_input, "anaphora")

    if detect_epi:
        epi_input = [
            (i, ld["end_key"], ld["end_phrase"], ld["end_had_stop"])
            for i, ld in enumerate(line_data)
        ]
        occurrences += find_groups(epi_input, "epiphora")

    # -----------------------------------------------------------------------
    # Build set of (line_num, kind) for fast highlight lookup
    # -----------------------------------------------------------------------
    # A line can be both anaphora AND epiphora (symploce) — we mark both.
    ana_line_set = set()
    epi_line_set = set()
    for occ in occurrences:
        for ln in occ["line_nums"]:
            if occ["kind"] == "anaphora":
                ana_line_set.add(ln)
            else:
                epi_line_set.add(ln)

    # -----------------------------------------------------------------------
    # Build annotated HTML
    # For each line: highlight the start-phrase (anaphora, gold) and/or
    # end-phrase (epiphora, teal) tokens in the raw line text.
    # We do this by locating the token text within the raw line and wrapping
    # it in a <span>.
    # -----------------------------------------------------------------------

    def highlight_tokens_in_line(raw_line: str, phrase_tokens: list[dict],
                                  position: str, css_class: str,
                                  skip_stops: bool, norm_stops_set: set,
                                  line_tokens: list[dict]) -> str:
        """
        Return raw_line with the phrase tokens wrapped in <span class=css_class>.
        We locate them by scanning the raw line for the token texts.
        """
        if not phrase_tokens:
            return html.escape(raw_line)

        # Build list of (token_text, is_phrase_member) for all tokens in line
        phrase_norms = {t["norm"] for t in phrase_tokens}

        # For start: the phrase tokens are the first phrase_length non-stop tokens
        # For end:   the phrase tokens are the last phrase_length non-stop tokens
        # We need to know their positions in the original token list.

        toks = line_tokens
        if position == "start":
            candidate_toks = []
            for t in toks:
                if skip_stops and _is_stopword(t["norm"], norm_stops_set):
                    continue
                candidate_toks.append(t)
            highlight_texts = {t["text"] for t in candidate_toks[:phrase_length]}
            highlight_set_indices = []
            count = 0
            for i, t in enumerate(toks):
                if skip_stops and _is_stopword(t["norm"], norm_stops_set):
                    continue
                if count < phrase_length:
                    highlight_set_indices.append(i)
                    count += 1
                else:
                    break
        else:  # end
            candidate_toks = []
            for t in toks:
                if skip_stops and _is_stopword(t["norm"], norm_stops_set):
                    continue
                candidate_toks.append(t)
            tail = candidate_toks[-phrase_length:] if len(candidate_toks) >= phrase_length else candidate_toks
            highlight_texts_list = [t["text"] for t in tail]
            # Find their indices in original token list (from the end)
            highlight_set_indices = []
            count = 0
            needed = len(highlight_texts_list)
            for i in range(len(toks) - 1, -1, -1):
                t = toks[i]
                if skip_stops and _is_stopword(t["norm"], norm_stops_set):
                    continue
                if count < needed:
                    highlight_set_indices.append(i)
                    count += 1
                else:
                    break

        highlight_set = {toks[i]["text"] for i in highlight_set_indices}

        # Now rebuild line HTML token by token
        parts = []
        raw_remaining = raw_line
        token_idx = 0
        char_pos = 0

        for t in toks:
            # Find this token in raw_remaining
            tok_text = t["text"]
            idx = raw_remaining.find(tok_text)
            if idx == -1:
                continue
            # Everything before this token
            parts.append(html.escape(raw_remaining[:idx]))
            # The token itself
            escaped = html.escape(tok_text)
            if tok_text in highlight_set:
                parts.append(f'<span class="{css_class}">{escaped}</span>')
            else:
                parts.append(escaped)
            raw_remaining = raw_remaining[idx + len(tok_text):]

        parts.append(html.escape(raw_remaining))
        return "".join(parts)

    skip_stops = opts.get("skip_stopwords", True)
    html_lines = []
    for ld in line_data:
        ln = ld["line_num"]
        raw = ld["raw"]
        toks = ld["tokens"]

        is_ana = ln in ana_line_set
        is_epi = ln in epi_line_set

        if not is_ana and not is_epi:
            html_lines.append(html.escape(raw))
            continue

        # Start with raw line; apply anaphora highlight first, then epiphora
        line_html = raw  # we'll process token-by-token

        if is_ana and is_epi:
            # Both: highlight start with anaphora class, end with epiphora class
            # We do two passes. For simplicity, build a token-level markup.
            line_html = _highlight_both(raw, toks, ld["start_phrase"], ld["end_phrase"],
                                        phrase_length, skip_stops, norm_stops)
        elif is_ana:
            line_html = highlight_tokens_in_line(
                raw, ld["start_phrase"], "start", "rep-anaphora",
                skip_stops, norm_stops, toks)
        else:
            line_html = highlight_tokens_in_line(
                raw, ld["end_phrase"], "end", "rep-epiphora",
                skip_stops, norm_stops, toks)

        html_lines.append(line_html)

    annotated_html = "\n".join(html_lines)

    # -----------------------------------------------------------------------
    # Number occurrences sequentially
    # -----------------------------------------------------------------------
    for n, occ in enumerate(occurrences, 1):
        occ["index"] = n

    return annotated_html, occurrences


def _highlight_both(raw_line, toks, start_phrase, end_phrase,
                    phrase_length, skip_stops, norm_stops):
    """
    Highlight start-phrase tokens (anaphora) and end-phrase tokens (epiphora)
    within the same line, handling possible overlap gracefully.
    """
    if not toks:
        return html.escape(raw_line)

    # Determine which token indices get which class
    # Start phrase: first phrase_length non-stop tokens
    start_indices = set()
    count = 0
    for i, t in enumerate(toks):
        if skip_stops and t["norm"] in norm_stops:
            continue
        if count < phrase_length:
            start_indices.add(i)
            count += 1
        else:
            break

    # End phrase: last phrase_length non-stop tokens
    end_indices = set()
    count = 0
    needed = phrase_length
    for i in range(len(toks) - 1, -1, -1):
        t = toks[i]
        if skip_stops and t["norm"] in norm_stops:
            continue
        if count < needed:
            end_indices.add(i)
            count += 1
        else:
            break

    parts = []
    raw_remaining = raw_line
    for i, t in enumerate(toks):
        tok_text = t["text"]
        idx = raw_remaining.find(tok_text)
        if idx == -1:
            continue
        parts.append(html.escape(raw_remaining[:idx]))
        escaped = html.escape(tok_text)
        if i in start_indices and i in end_indices:
            parts.append(f'<span class="rep-anaphora rep-epiphora">{escaped}</span>')
        elif i in start_indices:
            parts.append(f'<span class="rep-anaphora">{escaped}</span>')
        elif i in end_indices:
            parts.append(f'<span class="rep-epiphora">{escaped}</span>')
        else:
            parts.append(escaped)
        raw_remaining = raw_remaining[idx + len(tok_text):]

    parts.append(html.escape(raw_remaining))
    return "".join(parts)


# ---------------------------------------------------------------------------
# HTML output template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>Anaphora & Epiphora Highlights</title>
<style>
body {{ font-family: serif; padding: 1.5rem; max-width: 900px; margin: auto; }}
pre.source {{ white-space: pre-wrap; font-size: 18px; line-height: 1.6; }}
.rep-anaphora  {{ background: rgba(212, 160, 23, 0.40); border-bottom: 2px solid #d4a017; }}
.rep-epiphora  {{ background: rgba(30, 160, 160, 0.35); border-bottom: 2px solid #1ea0a0; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1.5rem; font-size: 0.95rem; }}
td,th {{ border: 1px solid #bbb; padding: 6px 10px; vertical-align: top; }}
th {{ background: #f4f4f4; }}
</style>
</head>
<body>
<h1>Anaphora &amp; Epiphora Highlights</h1>
<p>
  <span style="background:rgba(212,160,23,0.4);padding:2px 6px;">Gold</span> = Anaphora (A) &nbsp;
  <span style="background:rgba(30,160,160,0.35);padding:2px 6px;">Teal</span> = Epiphora (E)
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


def write_outputs(annotated: str, occurrences: list, html_path, csv_path):
    rows = []
    csv_rows = []

    for occ in occurrences:
        n = occ["index"]
        kind_short = "A" if occ["kind"] == "anaphora" else "E"
        lines_str = ", ".join(str(ln) for ln in occ["line_nums"])
        phrase_display = html.escape(
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

    html_text = HTML_TEMPLATE.format(
        annotated=annotated,
        rows="\n".join(rows),
    )
    Path(html_path).write_text(html_text, encoding="utf-8")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["index", "type", "phrase_normalised", "phrase_original",
                      "lines", "stop_skipped"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in csv_rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# Entry point for Pyodide / GUI
# ---------------------------------------------------------------------------

def process(input_path: str, html_path: str, csv_path: str):
    """Called by app.js via Pyodide."""
    text = Path(input_path).read_text(encoding="utf-8")
    annotated, occurrences = detect_repetitions(text)
    write_outputs(annotated, occurrences, html_path, csv_path)
    return occurrences
