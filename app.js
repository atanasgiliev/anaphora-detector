/*
 * Greek Anaphora & Epiphora Detector — app.js
 * Mirrors the hiatus-detector architecture:
 *   1. Load Pyodide + detector.py into the browser
 *   2. Write /input.txt and /options.json into Pyodide FS
 *   3. Call detector.process() → read back HTML + CSV
 *   4. Render results + enable downloads
 */

"use strict";

// ── DOM refs ────────────────────────────────────────────────────────────────
const fileInput          = document.getElementById("fileInput");
const pasteArea          = document.getElementById("pasteArea");
const runBtn             = document.getElementById("runBtn");
const statusEl           = document.getElementById("status");
const outputEl           = document.getElementById("output");
const perLineTableEl     = document.getElementById("perLineTable");

const dlHtmlBtn          = document.getElementById("downloadHtmlBtn");
const dlCsvBtn           = document.getElementById("downloadCsvBtn");

// Options
const optAna             = document.getElementById("optAnaphora");
const optEpi             = document.getElementById("optEpiphora");
const optPhraseLen       = document.getElementById("optPhraseLen");
const optPhraseLenVal    = document.getElementById("optPhraseLenVal");
const optWindow          = document.getElementById("optWindow");
const optWindowVal       = document.getElementById("optWindowVal");
const optMinOcc          = document.getElementById("optMinOcc");
const optMinOccVal       = document.getElementById("optMinOccVal");
const optStopwords       = document.getElementById("optStopwords");
const optAccents         = document.getElementById("optAccents");
const optBreathings      = document.getElementById("optBreathings");
const optIotaSub         = document.getElementById("optIotaSub");
const optElision         = document.getElementById("optElision");
const optNuMovable       = document.getElementById("optNuMovable");

// ── State ───────────────────────────────────────────────────────────────────
let pyodide      = null;
let pyLoaded     = false;
let textContent  = null;   // string loaded from file or paste
let lastHtmlBlob = null;
let lastCsvBlob  = null;

// ── Slider labels ───────────────────────────────────────────────────────────
optPhraseLen.addEventListener("input", () => {
    optPhraseLenVal.textContent = optPhraseLen.value;
});
optWindow.addEventListener("input", () => {
    optWindowVal.textContent = optWindow.value;
});
optMinOcc.addEventListener("input", () => {
    optMinOccVal.textContent = optMinOcc.value;
});

// ── File / paste input ───────────────────────────────────────────────────────
fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
        textContent = e.target.result;
        pasteArea.value = "";
        updateRunBtn();
        setStatus(`Loaded "${file.name}" (${textContent.length.toLocaleString()} chars).`);
    };
    reader.readAsText(file, "utf-8");
});

pasteArea.addEventListener("input", () => {
    textContent = pasteArea.value.trim() || null;
    if (textContent) fileInput.value = "";
    updateRunBtn();
});

function updateRunBtn() {
    runBtn.disabled = !(textContent && pyLoaded);
}

// ── Pyodide loading ──────────────────────────────────────────────────────────
async function loadPyodideAndDetector() {
    setStatus("Loading Pyodide… (first load may take ~10 s)");
    try {
        pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/" });
        // Write detector.py into the Pyodide virtual FS
        const resp = await fetch("detector.py");
        const code = await resp.text();
        pyodide.FS.writeFile("/detector.py", code);
        await pyodide.runPythonAsync(`exec(open('/detector.py').read())`);
        pyLoaded = true;
        updateRunBtn();
        setStatus("Ready. Load a text file or paste Greek text above.");
    } catch (err) {
        setStatus("Error loading Pyodide: " + err.message, true);
    }
}

loadPyodideAndDetector();

// ── Run ──────────────────────────────────────────────────────────────────────
runBtn.addEventListener("click", async () => {
    if (!textContent || !pyLoaded) return;

    // Validate: at least one type checked
    if (!optAna.checked && !optEpi.checked) {
        setStatus("Please select at least one type (Anaphora or Epiphora).", true);
        return;
    }

    runBtn.disabled = true;
    dlHtmlBtn.disabled = true;
    dlCsvBtn.disabled = true;
    outputEl.innerHTML = "";
    if (perLineTableEl) perLineTableEl.innerHTML = "";
    setStatus("Analysing…");

    // Write input file
    pyodide.FS.writeFile("/input.txt", textContent);

    // Write options
    const options = {
        detect_anaphora:      optAna.checked,
        detect_epiphora:      optEpi.checked,
        phrase_length:        parseInt(optPhraseLen.value),
        distance_window:      parseInt(optWindow.value),
        min_occurrences:      parseInt(optMinOcc.value),
        skip_stopwords:       optStopwords.checked,
        strip_accents:        optAccents.checked,
        strip_breathings:     optBreathings.checked,
        strip_iota_subscript: optIotaSub.checked,
        handle_elision:       optElision.checked,
        handle_nu_movable:    optNuMovable.checked,
    };
    pyodide.FS.writeFile("/options.json", JSON.stringify(options));

    try {
        await pyodide.runPythonAsync(`
import importlib, sys
# Re-exec detector to pick up fresh options.json
exec(open('/detector.py').read())
occs = process('/input.txt', '/output.html', '/output.csv')
`);

        const htmlBytes = pyodide.FS.readFile("/output.html");
        const csvBytes  = pyodide.FS.readFile("/output.csv");
        const htmlStr   = new TextDecoder("utf-8").decode(htmlBytes);
        const csvStr    = new TextDecoder("utf-8").decode(csvBytes);

        // Count results
        const occPy = pyodide.globals.get("occs");
        const occLen = occPy && occPy.length !== undefined ? occPy.length : "?";

        // Render inline
        renderOutput(htmlStr);

        // Prepare downloads
        lastHtmlBlob = new Blob([htmlBytes], { type: "text/html;charset=utf-8" });
        lastCsvBlob  = new Blob([csvBytes],  { type: "text/csv;charset=utf-8" });
        dlHtmlBtn.disabled = false;
        dlCsvBtn.disabled  = false;

        setStatus(`Done. ${occLen} rhetorical figure${occLen === 1 ? "" : "s"} found.`);
    } catch (err) {
        setStatus("Detection error: " + err.message, true);
        console.error(err);
    } finally {
        runBtn.disabled = false;
    }
});

// ── Render output ─────────────────────────────────────────────────────────────
function renderOutput(fullHtml) {
    // Extract annotated <pre> and occurrences <table> from the full HTML
    const parser = new DOMParser();
    const doc    = parser.parseFromString(fullHtml, "text/html");

    const pre   = doc.querySelector("pre.source");
    const table = doc.querySelector("table");

    outputEl.innerHTML = "";

    if (pre) {
        const section = document.createElement("div");
        section.className = "output-section";
        const h = document.createElement("h3");
        h.textContent = "Annotated Text";
        section.appendChild(h);
        section.appendChild(pre.cloneNode(true));
        outputEl.appendChild(section);
    }

    if (table) {
        const section = document.createElement("div");
        section.className = "output-section";
        const h = document.createElement("h3");
        h.textContent = "Occurrence Table";
        section.appendChild(h);
        section.appendChild(table.cloneNode(true));
        outputEl.appendChild(section);
    }
}

// ── Downloads ─────────────────────────────────────────────────────────────────
dlHtmlBtn.addEventListener("click", () => {
    if (!lastHtmlBlob) return;
    triggerDownload(lastHtmlBlob, "anaphora_epiphora.html");
});
dlCsvBtn.addEventListener("click", () => {
    if (!lastCsvBlob) return;
    triggerDownload(lastCsvBlob, "anaphora_epiphora.csv");
});

function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a   = document.createElement("a");
    a.href    = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

// ── Status helper ─────────────────────────────────────────────────────────────
function setStatus(msg, isError = false) {
    statusEl.textContent = msg;
    statusEl.className   = isError ? "status error" : "status";
}
