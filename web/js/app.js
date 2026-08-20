// ==================== Configuration ==================== //
// Defensive number parser: localStorage values can be corrupted by
// DevTools edits, browser extensions, or partial JSON round-trips
// (e.g. ``"abc"`` or ``"3.5px"``). The naive ``parseInt(v, 10)``
// silently returns NaN for non-numeric input, and downstream
// ``setInterval(fn, NaN * 1000)`` is a no-op in every browser —
// the polling loop would just stop, the user gets no jobs
// updates, and the bug is invisible (no error message, just stale
// UI). Round 10 fix: gate every parseInt on ``Number.isFinite`` and
// fall back to the supplied default.
function _safeParseInt(value, fallback) {
    if (value == null || value === '') return fallback;
    const n = parseInt(value, 10);
    return Number.isFinite(n) ? n : fallback;
}

// Round 13: wrap every localStorage call so quota errors / Safari
// private-mode throws don't bubble up and break the surrounding click
// handler. ``getItem`` returning null when access is denied is
// indistinguishable from "key not set", so callers should treat both
// as the default — which is what ``||`` already gives us for
// ``apiBaseUrl``. The wrapper keeps the same shape but never throws.
// Round 18: M3 sometimes returns deliberate refusals for
// non-specimen figures (bar charts, tables, maps). The pipeline
// should silently skip those instead of asking the operator to
// choose a fallback for a no-op decision. This helper is the
// defensive frontend double-check (the server already marks
// these with is_non_specimen_figure, but a stale frontend mustn't
// push the popup anyway).
const _NON_SPECIMEN_REFUSAL_PATTERNS = [
    '该panel', '并非', '不涉及', '无标签', '无物种', '不可判定',
    '不是放射虫', '不是标本', '非标本', '非放射虫', '不是图版',
    '非图版', '非显微', 'bar chart', 'bar graph', '柱状图',
    '统计图', '折线图', '数量统计', 'publication count',
    'publication number', 'no specimen', 'no panel',
    'not a radiolarian', 'not a specimen', 'no radiolarian',
    'is not a radiolarian', 'is not a specimen', 'no specimen panels',
    'no panels found', 'is a chart', 'is a table', 'is a graph',
    'is a diagram', 'is a map', 'is a photo', 'is a photomicrograph',
    'is text', 'is a title page', 'is a reference',
];
function looksLikeNonSpecimenRefusal(text) {
    if (!text) return false;
    const hay = String(text).toLowerCase();
    return _NON_SPECIMEN_REFUSAL_PATTERNS.some(p => hay.includes(p.toLowerCase()));
}

function _safeStorageGet(key) {
    try { return localStorage.getItem(key); } catch (_) { return null; }
}
function _safeStorageSet(key, value) {
    try { localStorage.setItem(key, value); return true; } catch (_) { return false; }
}
function _safeStorageRemove(key) {
    try { localStorage.removeItem(key); return true; } catch (_) { return false; }
}
const CONFIG = {
    apiBaseUrl: _safeStorageGet('apiBaseUrl') || 'http://localhost:8000',
    refreshInterval: _safeParseInt(_safeStorageGet('refreshInterval'), 3),
};

let uploadedFiles = [];
let jobsData = {};
let resultsData = [];
// Set of currently-checked result row_ids in the Results tab. Persists
// across re-renders so a search/filter change doesn't silently drop
// the user's selection. Cleared after a successful delete and on
// "select all" toggle-off.
let selectedResultRowIds = new Set();
let refreshIntervalId = null;
let _notificationTimer = null;
// Auto-tab-switch timer for "job just completed" notifications. The
// loadJobs() poll sets this to a setTimeout id 1200ms after a job
// transitions to ``done``; the timer fires and switches the active
// tab to ``results``. Round 10 fix: store the id so manual tab
// switches (user clicks a tab during the 1200ms grace period) can
// cancel the timer and prevent the user's intent from being
// silently overridden. Pre-fix, switching to "settings" to check
// the API key while a job was finishing would yank the user back to
// "results" when the timer fired.
let _autoSwitchTimer = null;
// Per-page stash of full record objects. Indexed by `data-record-index`
// on each <img> / <button> in the rendered results table. Reset at the
// top of every renderResults() call to avoid stale references across
// re-renders. Replaces the previous pattern of embedding the entire
// JSON in a data-record attribute, which (a) duplicated data 25+ times
// per page, (b) was a XSS vector if records contained `&` or `<`.
let __rlpeRecords = [];

// ==================== fetchWithTimeout helper (Phase F-2 M1) ==================== //
// Wraps fetch() with an AbortController timeout so hung backends don't
// leave the UI in limbo. All 18 direct fetch() call sites have been
// converted to use this helper. AbortError is caught at the call site
// and surfaced as a user-friendly toast.
async function fetchWithTimeout(url, options = {}, timeoutMs = 30000) {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } finally {
        clearTimeout(timer);
    }
}

// ==================== Utilities ==================== //
function showNotification(message, type = 'success') {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification ${type}`;
    // Cancel any pending hide-timer from a previous notification so that
    // fast successive calls (e.g. "[1/3] uploaded" then "[2/3] uploaded")
    // don't get hidden by an earlier timer.
    if (_notificationTimer) {
        clearTimeout(_notificationTimer);
    }
    _notificationTimer = setTimeout(() => {
        notification.classList.add('hidden');
        _notificationTimer = null;
    }, 3000);
}

function formatFileSize(bytes) {
    const units = ['B', 'KB', 'MB', 'GB'];
    let size = bytes;
    let unitIndex = 0;
    while (size > 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
    }
    return `${size.toFixed(2)} ${units[unitIndex]}`;
}

function formatDate(date) {
    if (!date) return 'N/A';
    const d = new Date(date);
    if (isNaN(d.getTime())) return 'N/A';
    return d.toLocaleString('zh-CN');
}

function formatElapsed(sec) {
    if (sec == null) return 'N/A';
    sec = Math.max(0, Math.floor(sec));
    if (sec < 60) return `${sec} 秒`;
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    if (m < 60) return `${m} 分 ${s} 秒`;
    const h = Math.floor(m / 60);
    const mm = m % 60;
    return `${h} 时 ${mm} 分`;
}

// Resolve an asset path into a URL the browser can fetch.
//
// audit 2026-08-17 (WEB-B1): the previous version only handled
// ``http(s)://`` and absolute paths starting with ``/``. Real
// ``panel_path`` values from ``/results`` are filesystem-relative
// (``work/.../panel_01.png``); the original string was returned
// verbatim and produced a 404. The row's ``job_id`` is now
// threaded through so we can build the canonical
// ``GET /jobs/{job_id}/files/{rel}`` URL the server exposes.
function resolveAssetUrl(path, jobId) {
    if (!path) return '';
    if (/^https?:\/\//i.test(path)) return path;
    if (path.startsWith('/')) return `${CONFIG.apiBaseUrl}${path}`;
    // Filesystem-relative path — needs a jobId to map to the
    // /jobs/{id}/files/... endpoint. If we don't have one, fall
    // back to the raw path (which will 404) rather than crashing
    // the whole row render.
    if (jobId) {
        // Strip any leading "./" so the server's relative_to check
        // stays happy. Trailing slashes don't matter (FileResponse
        // ignores them).
        const rel = String(path).replace(/^\.\//, '');
        return `${CONFIG.apiBaseUrl}/jobs/${encodeURIComponent(jobId)}/files/${rel}`;
    }
    return path;
}

async function checkApiHealth() {
    const status = document.getElementById('api-status');
    try {
        const response = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/health`);
        if (response.ok) {
            status.textContent = '已连接';
            status.className = 'status-indicator status-connected';
            return true;
        } else {
            status.textContent = `服务异常 (${response.status})`;
            status.className = 'status-indicator status-error';
            return false;
        }
    } catch (error) {
        // Phase F-2 M1: handle AbortError (timeout) separately.
        if (error.name === 'AbortError') {
            status.textContent = '无法连接 (请求超时)';
            status.className = 'status-indicator status-error';
            return false;
        }
        // Distinguish network/CORS failures from other errors so the user
        // can tell "server is down" from "browser is blocking the request".
        if (error instanceof TypeError) {
            // Failed to fetch — most commonly a network failure or a CORS
            // rejection. Both look identical from JS, so just say so.
            status.textContent = '无法连接 (网络/CORS)';
        } else {
            status.textContent = `连接错误: ${error.message || error}`;
        }
        status.className = 'status-indicator status-error';
        return false;
    }
}

// ==================== Tab Navigation ==================== //
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const tabBtn = e.currentTarget;
        const tabName = tabBtn.dataset.tab;

        // Remove active class from all tabs
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));

        // Add active class to clicked tab
        tabBtn.classList.add('active');
        document.getElementById(`${tabName}-tab`)?.classList.add('active');

        // Cancel any pending auto-switch (a "job just completed" timer
        // scheduled by loadJobs). The user has explicitly chosen a tab;
        // we should respect that instead of yanking them away 1.2s later.
        // Round 10 fix for FM2 / FH2.
        if (_autoSwitchTimer !== null) {
            clearTimeout(_autoSwitchTimer);
            _autoSwitchTimer = null;
        }

        // Load tab-specific data
        if (tabName === 'jobs') {
            loadJobs();
        } else if (tabName === 'results') {
            loadResults();
        }
    });
});

// ==================== Upload Functionality ==================== //
const uploadArea = document.getElementById('upload-area');
const pdfInput = document.getElementById('pdf-input');

// Round 10 (FM3): reset the file input before opening the picker so
// re-selecting the same file (e.g. after "清空列表") actually fires
// the ``change`` event. Browsers do not re-fire ``change`` when the
// user picks the same FileList as before — pre-fix, a workflow like
// "upload A.pdf → clear → re-upload A.pdf" silently did nothing.
function openFilePicker() {
    pdfInput.value = '';
    pdfInput.click();
}

uploadArea.addEventListener('click', openFilePicker);

// Keyboard accessibility: the upload area is a div (not a button), so
// without this handler, Tab skips it and pressing Enter / Space does
// nothing. role="button" + tabindex="0" is set in HTML; this wires up
// the keypress.
uploadArea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openFilePicker();
    }
});

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    addFiles(files);
});

pdfInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    addFiles(files);
});

function addFiles(files) {
    // Phase F-2 M2: 256 MB per-file size limit.
    const MAX_FILE_SIZE = 256 * 1024 * 1024;
    const oversized = files.filter(f => f.size > MAX_FILE_SIZE);
    if (oversized.length > 0) {
        oversized.forEach(f =>
            showToast(`文件 '${f.name}' 超过 256 MB 限制（已跳过）`, 'warning')
        );
    }
    // Case-insensitive .pdf extension: macOS / iOS Finder and many
    // academic-paper repos serve files with upper-case `.PDF` (especially
    // when the original was scanned/OCR'd). The previous
    // ``f.name.endsWith('.pdf')`` silently dropped them.
    const pdfFiles = files.filter(
        f => (f.type === 'application/pdf' || /\.pdf$/i.test(f.name)) && f.size <= MAX_FILE_SIZE
    );
    if (pdfFiles.length === 0) {
        showNotification('请选择 PDF 文件', 'error');
        return;
    }

    uploadedFiles.push(...pdfFiles);
    renderFileList();
    document.getElementById('process-btn').disabled = uploadedFiles.length === 0;
}

function renderFileList() {
    const fileList = document.getElementById('file-list');
    if (uploadedFiles.length === 0) {
        fileList.innerHTML = '';
        return;
    }
    
    fileList.innerHTML = uploadedFiles.map((file, index) => `
        <div class="file-item">
            <div class="file-item-info">
                <div>
                    <div class="file-item-name">${escapeHtml(file.name)}</div>
                    <div class="file-item-size">${escapeHtml(formatFileSize(file.size))}</div>
                </div>
            </div>
            <button type="button" class="file-item-remove" data-file-index="${index}">删除</button>
        </div>
    `).join('');
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    renderFileList();
    document.getElementById('process-btn').disabled = uploadedFiles.length === 0;
}

// Delegated click for the per-file "删除" button. Replaces the previous
// inline onclick="removeFile(${index})" which had to be rebuilt on every
// re-render and is a (low-risk) XSS vector if index came from anywhere
// other than a JS numeric loop.
document.getElementById('file-list')?.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-file-index]');
    if (!btn) return;
    const idx = parseInt(btn.getAttribute('data-file-index'), 10);
    if (!isNaN(idx)) removeFile(idx);
});

document.getElementById('clear-btn').addEventListener('click', () => {
    uploadedFiles = [];
    renderFileList();
    document.getElementById('process-btn').disabled = true;
    pdfInput.value = '';
});

// ==================== Process Functionality ==================== //
let _processing = false;

document.getElementById('process-btn').addEventListener('click', async () => {
    if (uploadedFiles.length === 0 || _processing) return;

    _processing = true;
    const btn = document.getElementById('process-btn');
    btn.disabled = true;
    btn.classList.add('btn-loading');
    btn.innerHTML = '<span class="spinner-small"></span> 处理中...';

    const totalFiles = uploadedFiles.length;
    let uploadedCount = 0;

    try {
        // Collect LLM options from the form (returns null if "启用 LLM 增强" is unchecked,
        // throws if any field is invalid)
        let llmOptions = null;
        try {
            llmOptions = _buildLLMOptions();
        } catch (validationErr) {
            showNotification(validationErr.message, 'error');
            return;
        }
        // Collect Paleobiology Database options (returns null if checkbox is off)
        let paleodbOptions = null;
        try {
            paleodbOptions = _buildPaleodbOptions();
        } catch (validationErr) {
            showNotification(validationErr.message, 'error');
            return;
        }
        // PDF figure extractor (GROBID vs OpenDataLoader). When the user checks
        // the box, the pipeline uses OpenDataLoader-pdf in-process — no GROBID
        // server required.
        const useOpenDataLoader = document.getElementById('use-opendataloader')?.checked ?? false;

        // Core pipeline options that USED to be silently dropped: the form
        // rendered them but the previous build never read them, so the user's
        // GROBID URL / OCR engine / worker count / panel-score threshold were
        // ignored. Validate up-front so an invalid number returns a toast
        // instead of a server 400.
        let coreOpts;
        try {
            coreOpts = _buildCorePipelineOptions();
        } catch (validationErr) {
            showNotification(validationErr.message, 'error');
            return;
        }

        // Merge LLM + PBDB + extractor + core options into a single JSON body.
        // Always send the merged object so the server gets the user's full
        // configuration (the previous version skipped the request body when
        // LLM/PBDB were off, losing the core pipeline overrides too).
        const combinedOptions = {
            use_opendataloader: useOpenDataLoader,
            ...coreOpts,
            ...(llmOptions || {}),
            ...(paleodbOptions || {}),
        };

        for (const file of uploadedFiles) {
            // Update button text with progress so the user knows which
            // file is being uploaded (especially important for batch
            // uploads of 5+ PDFs where the total wait can be minutes).
            btn.innerHTML = `<span class="spinner-small"></span> 上传中 (${uploadedCount + 1}/${totalFiles})…`;

            const formData = new FormData();
            formData.append('file', file);
            // combinedOptions is always a real object now (it always at
            // least carries use_opendataloader), so unconditionally attach
            // it. The previous ``if (combinedOptions)`` guard was a relic
            // from the old code path that returned null when both LLM and
            // PBDB were off.
            formData.append('options', JSON.stringify(combinedOptions));

            const response = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/jobs/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                // Try to read the server's JSON error body (Pydantic validation
                // error messages live in `detail`).
                let errMsg = `上传失败: HTTP ${response.status} ${response.statusText}`;
                try {
                    const errBody = await response.json();
                    if (errBody && errBody.detail) {
                        errMsg = `上传失败: ${errBody.detail}`;
                    }
                } catch (_) {
                    // Body wasn't JSON; keep the status-text fallback
                }
                throw new Error(errMsg);
            }

            const data = await response.json();
            jobsData[data.job_id] = data;
            uploadedCount++;
            showNotification(`[${uploadedCount}/${totalFiles}] ${file.name} 已提交`);
        }

        uploadedFiles = [];
        renderFileList();

        // Switch to jobs tab. Use ?. so a missing button (e.g. markup
        // refactor during deploy) doesn't throw and bypass the
        // startJobPolling() call below. Without the guard, a TypeError
        // here would land in the outer catch and uploadedFiles would not
        // be cleared (handled below the click), confusing the user.
        const jobsTab = document.querySelector('[data-tab="jobs"]');
        if (jobsTab) {
            jobsTab.click();
        } else {
            // Fall back to a direct tab-pane show so the user still sees
            // the new jobs in the list.
            const jobsPane = document.getElementById('jobs-tab');
            if (jobsPane) {
                document.querySelectorAll('.tab-pane').forEach(p => p.classList.add('hidden'));
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                jobsPane.classList.remove('hidden');
            }
        }

        // Start polling
        startJobPolling();
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        _processing = false;
        // Re-enable only if there are files left to process. After a
        // successful run ``uploadedFiles`` is cleared (line 317), so the
        // button should stay disabled — clicking it with no files is a
        // no-op that confuses users.
        btn.disabled = uploadedFiles.length === 0;
        btn.classList.remove('btn-loading');
        btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> 开始提取图版';
    }
});

// ==================== Config Toggles ==================== //
function _syncLLMBackendVisibility() {
    const backend = document.getElementById('llm-backend')?.value;
    const localConfig = document.getElementById('llm-local-config');
    const MiniMaxConfig = document.getElementById('MiniMax-config');
    if (backend === 'MiniMax') {
        localConfig?.classList.add('hidden');
        MiniMaxConfig?.classList.remove('hidden');
    } else {
        localConfig?.classList.remove('hidden');
        MiniMaxConfig?.classList.add('hidden');
    }
}

document.getElementById('use-gemma4').addEventListener('change', (e) => {
    const gemmaConfig = document.getElementById('gemma-config');
    if (e.target.checked) {
        gemmaConfig.classList.remove('hidden');
        _syncLLMBackendVisibility();
    } else {
        gemmaConfig.classList.add('hidden');
    }
});

document.getElementById('llm-backend').addEventListener('change', _syncLLMBackendVisibility);

// ==================== Paleobiology Database (PBDB) options ==================== //
document.getElementById('use-paleodb').addEventListener('change', (e) => {
    const paleodbConfig = document.getElementById('paleodb-config');
    if (e.target.checked) {
        paleodbConfig.classList.remove('hidden');
    } else {
        paleodbConfig.classList.add('hidden');
    }
});

function _buildPaleodbOptions() {
    const enabled = document.getElementById('use-paleodb')?.checked ?? false;
    if (!enabled) return null;
    const opts = { use_paleodb: true };
    const maxRaw = document.getElementById('paleodb-max-occurrences')?.value?.trim() ?? '';
    const maxOcc = parseInt(maxRaw, 10);
    if (maxRaw === '' || isNaN(maxOcc) || maxOcc < 1) {
        throw new Error(`PBDB 最大出现记录数必须是 ≥1 的整数，当前值: "${maxRaw}"`);
    }
    if (maxOcc > 500) {
        throw new Error(`PBDB 最大出现记录数不能超过 500，当前值: ${maxOcc}`);
    }
    opts.paleodb_max_occurrences = maxOcc;
    const endpoint = document.getElementById('paleodb-endpoint')?.value?.trim() ?? '';
    if (endpoint) opts.paleodb_endpoint = endpoint;
    opts.paleodb_offline = document.getElementById('paleodb-offline')?.checked ?? false;
    return opts;
}

// ==================== Build core pipeline options from form ==================== //
// These fields used to be rendered in the form but never sent to the API.
// Each option is only included when the user has provided a non-empty value
// so that omissions fall back to PipelineConfig defaults on the server.
function _buildCorePipelineOptions() {
    const opts = {};

    const grobidUrl = document.getElementById('grobid-url')?.value.trim();
    if (grobidUrl) {
        // Sanity check: only http(s) URLs are accepted to avoid surprising
        // protocols (file://, javascript:) reaching the server validator.
        if (!/^https?:\/\//i.test(grobidUrl)) {
            throw new Error(`GROBID 地址必须以 http:// 或 https:// 开头，当前值: "${grobidUrl}"`);
        }
        opts.grobid_url = grobidUrl;
    }

    const ocrBackend = document.getElementById('ocr-backend')?.value;
    if (ocrBackend) opts.ocr_backend = ocrBackend;

    const workersRaw = document.getElementById('num-workers')?.value.trim();
    if (workersRaw) {
        const n = parseInt(workersRaw, 10);
        if (isNaN(n) || n < 1 || n > 32) {
            throw new Error(`并发处理数必须是 1..32 的整数，当前值: "${workersRaw}"`);
        }
        opts.num_workers = n;
    }

    const scoreRaw = document.getElementById('min-panel-score')?.value.trim();
    if (scoreRaw) {
        const f = parseFloat(scoreRaw);
        if (isNaN(f) || f < 0 || f > 1) {
            throw new Error(`Panel 分割置信度阈值必须在 [0, 1]，当前值: "${scoreRaw}"`);
        }
        opts.min_panel_score = f;
    }

    return opts;
}

// ==================== Build LLM options from form ==================== //
function _buildLLMOptions() {
    const useGemma = document.getElementById('use-gemma4')?.checked ?? false;
    if (!useGemma) return null;

    const backend = document.getElementById('llm-backend')?.value
        || _safeStorageGet(LLM_BACKEND_KEY)
        || 'MiniMax';

    // Validate conf threshold up-front so the user gets immediate feedback
    // instead of a server round-trip.
    const confRaw = document.getElementById('gemma-conf-threshold')?.value?.trim() ?? '';
    const confThreshold = parseFloat(confRaw);
    if (confRaw === '' || isNaN(confThreshold) || confThreshold < 0 || confThreshold > 1) {
        throw new Error(`LLM 置信度阈值必须在 [0, 1]，当前值: "${confRaw}"`);
    }

    const options = {
        use_gemma4: true,
        llm_backend: backend,
        gemma_conf_threshold: confThreshold,
    };

    if (backend === 'MiniMax') {
        // MiniMax M3 API path
        const apiKey = document.getElementById('MiniMax-api-key')?.value?.trim() ?? '';
        if (apiKey) options.MiniMax_api_key = apiKey;
        const endpoint = document.getElementById('MiniMax-endpoint')?.value?.trim() ?? '';
        if (endpoint) options.MiniMax_endpoint = endpoint;
        const model = document.getElementById('MiniMax-model')?.value?.trim() ?? '';
        if (model) options.MiniMax_model = model;
        options.MiniMax_enable_thinking = document.getElementById('MiniMax-enable-thinking')?.checked ?? false;

        // Validate thinking budget up-front
        const thinkingRaw = document.getElementById('MiniMax-thinking-budget')?.value?.trim() ?? '';
        const thinkingBudget = parseInt(thinkingRaw, 10);
        if (thinkingRaw === '' || isNaN(thinkingBudget) || thinkingBudget < 0) {
            throw new Error(`思考 Token 预算必须是非负整数，当前值: "${thinkingRaw}"`);
        }
        if (thinkingBudget > 32_000) {
            throw new Error(`思考 Token 预算不能超过 32000，当前值: ${thinkingBudget}`);
        }
        // If enable_thinking is true, budget must be > 0
        if (options.MiniMax_enable_thinking && thinkingBudget === 0) {
            throw new Error(`启用扩展思考时，思考 Token 预算必须 > 0`);
        }
        options.MiniMax_thinking_budget_tokens = thinkingBudget;

        options.MiniMax_fallback_default = document.getElementById('MiniMax-fallback-default')?.value ?? 'rules';
        // Web mode always uses non-interactive popup (block on event.wait)
        options.MiniMax_interactive = false;
    } else {
        // Local backend path
        const host = document.getElementById('llm-host')?.value?.trim() ?? '';
        if (host) {
            if (backend === 'llamacpp') {
                options.llama_host = host;
            } else if (backend === 'ollama') {
                options.ollama_host = host;
            }
        }
    }

    return options;
}

// ==================== Jobs Management ==================== //
// Adaptive backoff for /jobs polling. When the server returns errors or the
// network is down, the previous code kept hammering /jobs every 3 seconds,
// producing a flood of console errors and toasts. We now double the interval
// on each consecutive failure (capped at 30 s) and reset to the configured
// interval on the first successful response.
let _consecutivePollFailures = 0;
const _MAX_POLL_FAILURES = 5;          // give up the toast after this many
const _MAX_POLL_BACKOFF_SEC = 30;

// In-flight guard: setInterval fires loadJobs() every CONFIG.refreshInterval
// seconds but the request itself can take longer (slow API + many jobs).
// Without this guard, an in-flight loadJobs() that hasn't returned yet
// will be overlapped by a second invocation from the next interval tick,
// leading to race conditions on jobsData (Object.assign + delete interleaving)
// and duplicated renderJobsList() calls that flicker the UI.
let _loadJobsInFlight = false;

async function loadJobs() {
    if (_loadJobsInFlight) {
        // Skip this tick; the next one will retry. Returning silently is
        // safer than await'ing the in-flight promise (which would still
        // permit the overlap on the *next* tick).
        return;
    }
    _loadJobsInFlight = true;
    try {
        const response = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/jobs`);
        if (!response.ok) {
            _onPollFailure(`HTTP ${response.status}`);
            return;
        }

        const jobs = await response.json();

        // Detect jobs that just transitioned to "done" since the last
        // poll. If the user is watching the jobs tab, auto-switch to the
        // results tab so they see the extraction results immediately.
        // Novice users often sit on the jobs tab watching the progress
        // bar and don't realise they need to click "结果查看" to see
        // the extracted species.
        const previouslyActive = new Set(
            Object.values(jobsData)
                .filter(j => j.status === 'queued' || j.status === 'running' || j.status === 'awaiting_user_decision')
                .map(j => j.job_id)
        );
        const justCompleted = jobs.filter(
            j => previouslyActive.has(j.job_id) && j.status === 'done'
        );

        // Build a fresh map from the server's response.
        const serverJobIds = new Set();
        const freshData = {};
        for (const job of jobs) {
            freshData[job.job_id] = job;
            serverJobIds.add(job.job_id);
        }
        // Remove jobs from the local cache that no longer exist on the
        // server (they were deleted by another tab / session / CLI).
        // The previous version used ``reduce(acc[job_id]=job, jobsData)``
        // which ONLY added or updated — never deleted — so deleted jobs
        // stayed in the UI forever.
        for (const cachedId of Object.keys(jobsData)) {
            if (!serverJobIds.has(cachedId)) {
                delete jobsData[cachedId];
            }
        }
        // Merge new data into the existing cache.
        Object.assign(jobsData, freshData);

        // First successful poll after one or more failures — restore the
        // configured refresh interval and stop showing the error toast.
        if (_consecutivePollFailures > 0) {
            _consecutivePollFailures = 0;
            if (refreshIntervalId) startJobPolling();
        }

        renderJobsList();
        maybeStopPolling();
        // After we refresh job state, check whether any of them is
        // blocked on a MiniMax M3 user decision and pop the modal.
        // This is the missing piece that made the backend's
        // FallbackHandler appear silently broken from the UI side.
        checkMiniMaxFallbacks();

        // Auto-switch to results tab when a job the user was watching
        // just completed. Only switch if the user is currently on the
        // jobs tab (don't yank them away from upload/settings) and at
        // least one job transitioned to done.
        if (justCompleted.length > 0) {
            const activeTab = document.querySelector('.tab-btn.active');
            if (activeTab && activeTab.dataset.tab === 'jobs') {
                const names = justCompleted.map(j => j.filename || j.job_id.substring(0, 8)).join(', ');
                showNotification(`✅ 处理完成：${names}，正在跳转到结果…`, 'success');
                // Brief delay so the user sees the "done" status before
                // the tab switch. Round 10 (FM2 / FH2): store the timer
                // id so a manual tab click during the 1200ms grace
                // period can cancel the auto-switch and respect the
                // user's intent.
                if (_autoSwitchTimer !== null) {
                    clearTimeout(_autoSwitchTimer);
                }
                _autoSwitchTimer = setTimeout(() => {
                    _autoSwitchTimer = null;
                    document.querySelector('[data-tab="results"]')?.click();
                }, 1200);
            }
        }
    } catch (error) {
        // Phase F-2 M1: handle AbortError (timeout) explicitly.
        if (error.name === 'AbortError') {
            _onPollFailure('请求超时（30秒）');
        } else {
            _onPollFailure(error.message || String(error));
        }
    } finally {
        _loadJobsInFlight = false;
    }
}

function _onPollFailure(reason) {
    _consecutivePollFailures += 1;
    // Only show a single toast on the first failure to avoid spamming
    // the user when the server is down.
    if (_consecutivePollFailures === 1) {
        console.error('Failed to load jobs:', reason);
        showNotification(`加载任务列表失败: ${reason}`, 'error');
    } else if (_consecutivePollFailures === _MAX_POLL_FAILURES) {
        showNotification('多次连接失败，已降低刷新频率', 'error');
    }
    // Apply exponential backoff: 3s -> 6s -> 12s -> 24s -> capped at 30s.
    if (refreshIntervalId) {
        clearInterval(refreshIntervalId);
        const base = CONFIG.refreshInterval || 3;
        const backoffSec = Math.min(
            base * Math.pow(2, _consecutivePollFailures - 1),
            _MAX_POLL_BACKOFF_SEC,
        );
        refreshIntervalId = setInterval(loadJobs, backoffSec * 1000);
    }
}

// Polling should be adaptive: stop hammering /jobs once all jobs have
// settled. Without this guard, a 1-hour-old task keeps the browser
// pulling the endpoint every 3 s forever (wasted bandwidth, extra
// server load, and noisy console errors if the server is down).
function maybeStopPolling() {
    const hasActive = Object.values(jobsData).some(
        j => j.status === 'queued' || j.status === 'running' || j.status === 'awaiting_user_decision'
    );
    if (refreshIntervalId && !hasActive) {
        clearInterval(refreshIntervalId);
        refreshIntervalId = null;
        return;
    }
    // Symmetric case: a manual loadJobs() (tab switch / refresh button /
    // visibility change) discovered an active job while polling was off.
    // Without this branch, the UI would show the active job but never
    // update its progress until another full reload. Auto-restart at the
    // configured interval so the progress bar moves.
    if (!refreshIntervalId && hasActive) {
        startJobPolling();
    }
}

// Manual kill-switch for the polling loop, exposed via window so users
// can stop it from devtools if the server is broken.
window.stopJobPolling = function() {
    if (refreshIntervalId) {
        clearInterval(refreshIntervalId);
        refreshIntervalId = null;
    }
};

function renderJobsList() {
    const jobsList = document.getElementById('jobs-list');
    const searchTerm = document.getElementById('job-search')?.value.toLowerCase() || '';
    const filterStatus = document.getElementById('job-filter')?.value || '';

    const jobs = Object.values(jobsData)
        .filter(job => {
            const matchesSearch = !searchTerm || job.job_id.includes(searchTerm);
            const matchesFilter = !filterStatus || job.status === filterStatus;
            return matchesSearch && matchesFilter;
        })
        .sort((a, b) => new Date(b.created_at || 0) - new Date(a.created_at || 0));

    if (jobs.length === 0) {
        const isFiltered = searchTerm || filterStatus;
        jobsList.innerHTML = isFiltered
            ? '<div style="text-align: center; color: var(--text-muted); padding: 2rem;">没有匹配的任务，试试清除搜索/筛选条件</div>'
            : `<div style="text-align: center; color: var(--text-muted); padding: 2.5rem 1rem;">
                <p style="font-size: 1rem; margin-bottom: 0.5rem;">📋 还没有处理任务</p>
                <p style="font-size: 0.875rem;">前往「上传处理」标签页，拖入 PDF 文件即可开始</p>
                <button class="btn btn-primary btn-small" style="margin-top: 1rem;" onclick="document.querySelector('[data-tab=\\'upload\\']').click()">去上传 PDF</button>
            </div>`;
        return;
    }

    // Use escapeHtml() on every backend string; use data-action + data-job-id for
    // event delegation (replaces previous inline onclick="..." which
    // concatenated job_id and would have XSS'd if a job_id ever contained
    // a quote character). The click handler lives at module load (below).
    jobsList.innerHTML = jobs.map(job => `
        <div class="job-card" data-job-id="${escapeHtml(job.job_id)}">
            <input type="checkbox" class="job-card-checkbox" data-job-id="${escapeHtml(job.job_id)}"
                   ${selectedJobIds.has(job.job_id) ? 'checked' : ''}>
            <div class="job-header">
                <div class="job-id">ID: ${escapeHtml(job.job_id.substring(0, 12))}...</div>
                <span class="job-status status-${escapeHtml(job.status)}">${escapeHtml(getStatusLabel(job.status))}</span>
            </div>
            <div class="job-details">
                <div class="job-detail-item">
                    <span class="job-detail-label">创建时间:</span>
                    <span class="job-detail-value">${escapeHtml(formatDate(job.created_at))}</span>
                </div>
                <div class="job-detail-item">
                    <span class="job-detail-label">文件:</span>
                    <span class="job-detail-value">${escapeHtml(job.filename || 'N/A')}</span>
                </div>
                <div class="job-detail-item">
                    <span class="job-detail-label">进度:</span>
                    <span class="job-detail-value">${escapeHtml(job.progress || 0)}%</span>
                </div>
                ${job.stage ? `
                <div class="job-detail-item">
                    <span class="job-detail-label">阶段:</span>
                    <span class="job-detail-value">${escapeHtml(job.stage)}</span>
                </div>` : ''}
                ${job.elapsed_sec != null ? `
                <div class="job-detail-item">
                    <span class="job-detail-label">已用时:</span>
                    <span class="job-detail-value">${escapeHtml(formatElapsed(job.elapsed_sec))}</span>
                </div>` : ''}
                ${job.detail ? `
                <div class="job-detail-item">
                    <span class="job-detail-label">说明:</span>
                    <span class="job-detail-value">${escapeHtml(job.detail)}</span>
                </div>` : ''}
            </div>
            <div class="job-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${escapeHtml(job.progress || 0)}%"></div>
                </div>
            </div>
            <div class="job-actions">
                <button type="button" class="btn btn-small" data-action="details" data-job-id="${escapeHtml(job.job_id)}">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                    详情
                </button>
                ${job.status === 'done' ? `<button type="button" class="btn btn-small btn-primary" data-action="results" data-job-id="${escapeHtml(job.job_id)}">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
                    查看结果 →
                </button>` : ''}
                ${(job.status === 'queued' || job.status === 'running' || job.status === 'awaiting_user_decision') ? `
                <button type="button" class="btn btn-small btn-secondary" data-action="cancel" data-job-id="${escapeHtml(job.job_id)}">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    取消
                </button>
                ` : ''}
                ${(job.status === 'done' || job.status === 'failed' || job.status === 'cancelled') ? `
                <button type="button" class="btn btn-small btn-danger" data-action="delete" data-job-id="${escapeHtml(job.job_id)}" title="删除任务及文件">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/></svg>
                    删除
                </button>` : ''}
            </div>
        </div>
    `).join('');
}

// Event delegation: one listener for all action buttons in the jobs list.
// The previous design used inline onclick="..." attributes with template
// strings interpolating job_id; that pattern both fails strict-CSP and
// is a XSS vector if job_id ever contains a quote character.
document.getElementById('jobs-list')?.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action]');
    if (!btn) return;
    const jobId = btn.getAttribute('data-job-id');
    const action = btn.getAttribute('data-action');
    if (action === 'details') viewJobDetails(jobId);
    else if (action === 'results') viewJobResults(jobId);
    else if (action === 'cancel') cancelJob(jobId, btn);
    else if (action === 'delete') deleteSingleJob(jobId);
});

// Delegated change listener for the per-row checkbox.
document.getElementById('jobs-list')?.addEventListener('change', (e) => {
    if (e.target.classList && e.target.classList.contains('job-card-checkbox')) {
        onJobSelectionChange();
    }
});

function getStatusLabel(status) {
    const labels = {
        'queued': '队列中',
        'running': '处理中',
        'awaiting_user_decision': '等待用户决策',
        'done': '已完成',
        'failed': '失败',
        'cancelled': '已取消'
    };
    return labels[status] || status;
}

// ==================== MiniMax fallback popup ==================== //
// The backend pauses a job in status='awaiting_user_decision' when the
// MiniMax API errors and waits up to 5 minutes for a user decision (see
// app.py::_web_fallback_popup). The previous JS only RECOGNISED that
// status for polling — it never actually FETCHED the pending decision
// and never SHOWED a popup, so users always silently timed out and
// got the headless default. The poll loop below closes that gap.
//
// Polled jobs are tracked in a Set so we only render one modal per job
// at a time (multiple polls hitting "awaiting_user_decision" for the
// same job would otherwise stack popups).
const _MiniMaxPopupShown = new Set();

async function checkMiniMaxFallbacks() {
    const awaiting = Object.values(jobsData).filter(
        j => j.status === 'awaiting_user_decision'
    );
    for (const job of awaiting) {
        if (_MiniMaxPopupShown.has(job.job_id)) continue;
        try {
            const r = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/jobs/${job.job_id}/MiniMax-fallback`);
            if (!r.ok) continue;
            const data = await r.json();
            if (data.status !== 'awaiting_decision') continue;
            _MiniMaxPopupShown.add(job.job_id);
            showMiniMaxFallbackModal(job.job_id, data.error_info || {});
        } catch (_) { /* network blip; next poll will retry */ }
    }
    // Clear stale popups for jobs that are no longer awaiting
    for (const jid of Array.from(_MiniMaxPopupShown)) {
        const j = jobsData[jid];
        if (!j || j.status !== 'awaiting_user_decision') {
            _MiniMaxPopupShown.delete(jid);
        }
    }
}

function showMiniMaxFallbackModal(jobId, errorInfo) {
    // Round 18: M3 sometimes returns a deliberate refusal for
    // non-specimen figures (bar charts, tables, maps). The server
    // already marks these with ``is_non_specimen_figure`` and skips
    // the popup entirely; this is a defensive double-check so an
    // older server build can't push the operator into a no-op
    // decision. Pattern-matches the standard "该panel为图表…无可判定"
    // reasoning text and treats it as a silent skip.
    if (errorInfo && errorInfo.is_non_specimen_figure) {
        console.info(
            'MiniMax returned non-specimen refusal for',
            jobId,
            '— silently skipping without popup.'
        );
        return;
    }
    const msg = (errorInfo && (errorInfo.error || errorInfo.reasoning)) || '';
    if (looksLikeNonSpecimenRefusal(msg)) {
        console.info(
            'MiniMax refusal text matched non-specimen pattern for',
            jobId,
            '— silently skipping without popup.'
        );
        return;
    }
    let modal = document.getElementById('MiniMax-fallback-modal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'MiniMax-fallback-modal';
        modal.className = 'modal';
        modal.innerHTML = `
            <div class="modal-content" style="max-width: 560px;">
                <div class="modal-header">
                    <h3>⚠️ MiniMax M3 调用失败</h3>
                </div>
                <div class="modal-body">
                    <p style="color: var(--text-muted); font-size: 0.9rem;">
                        云端 LLM 后端返回错误。请选择如何继续——5 分钟内未选择将自动应用默认策略。
                    </p>
                    <div class="MiniMax-fallback-err">
                        <div><strong>任务:</strong> <code id="MiniMax-fb-job"></code></div>
                        <div><strong>类型:</strong> <span id="MiniMax-fb-type"></span></div>
                        <div><strong>错误:</strong> <span id="MiniMax-fb-msg"></span></div>
                        <div id="MiniMax-fb-ctx-row" style="display:none"><strong>上下文:</strong> <span id="MiniMax-fb-ctx"></span></div>
                    </div>
                    <div class="MiniMax-fallback-actions">
                        <button class="btn btn-primary" data-action="retry">重试</button>
                        <button class="btn btn-secondary" data-action="rules">回退到规则流水线</button>
                        <button class="btn btn-secondary" data-action="gemma4">切换本地 Gemma4</button>
                        <button class="btn btn-danger" data-action="stop">中止任务</button>
                    </div>
                </div>
            </div>`;
        document.body.appendChild(modal);
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {/* don't close on overlay; force a choice */}
        });
    }
    document.getElementById('MiniMax-fb-job').textContent = jobId.substring(0, 12) + '...';
    document.getElementById('MiniMax-fb-type').textContent = errorInfo.error_type || 'Unknown';
    document.getElementById('MiniMax-fb-msg').textContent = (errorInfo.error || '').substring(0, 240);
    if (errorInfo.context) {
        document.getElementById('MiniMax-fb-ctx-row').style.display = '';
        document.getElementById('MiniMax-fb-ctx').textContent = errorInfo.context;
    } else {
        document.getElementById('MiniMax-fb-ctx-row').style.display = 'none';
    }
    modal.classList.remove('hidden');
    // Replace action buttons each open so old listeners don't pile up
    const newActions = modal.querySelectorAll('.MiniMax-fallback-actions [data-action]');
    newActions.forEach(btn => {
        const fresh = btn.cloneNode(true);
        btn.replaceWith(fresh);
        fresh.addEventListener('click', () => submitMiniMaxFallback(jobId, fresh.dataset.action, modal));
    });
}

async function submitMiniMaxFallback(jobId, action, modal) {
    modal.querySelectorAll('button').forEach(b => b.disabled = true);
    try {
        const r = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/jobs/${jobId}/MiniMax-fallback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ job_id: jobId, action }),
        });
        if (!r.ok) {
            const errBody = await r.json().catch(() => ({}));
            showNotification(`提交失败: ${errBody.detail || r.statusText}`, 'error');
        } else {
            showNotification(`已选择: ${action}`);
            modal.classList.add('hidden');
            _MiniMaxPopupShown.delete(jobId);
            loadJobs();
        }
    } catch (err) {
        // Phase F-2 M1: handle AbortError (timeout) explicitly.
        if (err.name === 'AbortError') {
            showNotification('提交失败: 请求超时', 'error');
        } else {
            showNotification(`提交失败: ${err.message || err}`, 'error');
        }
    } finally {
        modal.querySelectorAll('button').forEach(b => b.disabled = false);
    }
}

// ==================== Delete Jobs ==================== //
// In-memory set of selected job IDs. Persists across re-renders so
// filter changes don't lose the selection.
const selectedJobIds = new Set();

function onJobSelectionChange() {
    // Sync checkboxes -> Set and visual highlight
    document.querySelectorAll('.job-card-checkbox').forEach(cb => {
        const id = cb.dataset.jobId;
        if (cb.checked) selectedJobIds.add(id);
        else selectedJobIds.delete(id);
        const card = cb.closest('.job-card');
        if (card) card.classList.toggle('selected', cb.checked);
    });
    updateDeleteSelectedButton();
    syncSelectAllCheckbox();
}

function updateDeleteSelectedButton() {
    const btn = document.getElementById('delete-selected-btn');
    const count = document.getElementById('delete-selected-count');
    if (!btn || !count) return;
    const n = selectedJobIds.size;
    btn.disabled = n === 0;
    count.textContent = `(${n})`;
}

function syncSelectAllCheckbox() {
    const allCb = document.getElementById('jobs-select-all');
    if (!allCb) return;
    const visibleCbs = Array.from(document.querySelectorAll('.job-card-checkbox'));
    if (visibleCbs.length === 0) {
        allCb.checked = false;
        allCb.indeterminate = false;
        return;
    }
    const checkedCount = visibleCbs.filter(cb => cb.checked).length;
    allCb.checked = checkedCount === visibleCbs.length;
    allCb.indeterminate = checkedCount > 0 && checkedCount < visibleCbs.length;
}

function onSelectAllToggle() {
    const allCb = document.getElementById('jobs-select-all');
    if (!allCb) return;
    document.querySelectorAll('.job-card-checkbox').forEach(cb => {
        cb.checked = allCb.checked;
    });
    onJobSelectionChange();
}

function openDeleteModalForSelection() {
    if (selectedJobIds.size === 0) return;
    const ids = Array.from(selectedJobIds);
    openDeleteModal(ids);
}

// Backend /jobs/batch-delete rejects more than 200 job_ids in one call
// (see `app.py:batch_delete_jobs`). Surface this to the user before the
// round-trip so they don't have to retry after a 400.
const BATCH_DELETE_MAX = 200;

async function openDeleteModal(jobIds) {
    const modal = document.getElementById('delete-modal');
    const summary = document.getElementById('delete-modal-summary');
    const list = document.getElementById('delete-modal-jobs');
    const confirmBtn = document.getElementById('delete-modal-confirm');

    let idsToShow = jobIds;
    let truncWarn = '';
    if (jobIds.length > BATCH_DELETE_MAX) {
        // Keep the first 200 and warn the user. They'll need to delete
        // the rest in a second pass — we don't silently drop them.
        idsToShow = jobIds.slice(0, BATCH_DELETE_MAX);
        truncWarn = `<div style="color: var(--warning-color); font-size: 0.85rem; margin-top: 0.5rem;">
            ⚠ 共 ${jobIds.length} 个任务，超过单次最大 ${BATCH_DELETE_MAX}，已截取前 ${BATCH_DELETE_MAX} 个。其余任务请分批删除。
        </div>`;
    }

    const n = idsToShow.length;
    // NOTE: there is no per-job size endpoint, so we don't pre-compute
    // bytes here. The server returns ``bytes_freed`` in the delete
    // response, which we then surface in the toast. The previous
    // ``estimateSelectedBytes`` helper was dead code (always returned 0)
    // and has been removed.
    summary.innerHTML = `将删除 <strong>${n}</strong> 个任务。此操作不可撤销。${truncWarn}`;

    // Build per-job list with filename and a remove button
    list.innerHTML = idsToShow.map(id => {
        const job = jobsData[id];
        const filename = job?.filename || '(无文件)';
        return `<div class="job-row" data-row-id="${escapeHtml(id)}">
            <span class="job-row-id">${escapeHtml(id.substring(0, 12))}...</span>
            <span class="job-row-name" title="${escapeHtml(filename)}">${escapeHtml(filename)}</span>
        </div>`;
    }).join('');

    // Reset the files checkbox and prepare confirm button
    document.getElementById('delete-modal-files').checked = true;
    confirmBtn.disabled = false;
    confirmBtn.dataset.jobIds = JSON.stringify(idsToShow);

    modal.classList.remove('hidden');
}

function closeDeleteModal() {
    const modal = document.getElementById('delete-modal');
    if (modal) modal.classList.add('hidden');
    const confirmBtn = document.getElementById('delete-modal-confirm');
    if (confirmBtn) {
        confirmBtn.disabled = false;
        confirmBtn.textContent = '确认删除';
    }
}

// Pretty-print a byte count for delete toasts (the server returns
// bytes_freed in the delete response).
function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

async function confirmDelete() {
    const confirmBtn = document.getElementById('delete-modal-confirm');
    let jobIds = [];
    try {
        jobIds = JSON.parse(confirmBtn.dataset.jobIds || '[]');
        if (!Array.isArray(jobIds)) jobIds = [];
    } catch (e) {
        // Malformed data-job-ids attribute — fall back to singular jobId if present
        console.warn('confirmDelete: failed to parse jobIds', e);
        const singular = confirmBtn.dataset.jobId;
        jobIds = singular ? [singular] : [];
    }
    const deleteFiles = document.getElementById('delete-modal-files').checked;
    if (jobIds.length === 0) return;

    confirmBtn.disabled = true;
    confirmBtn.textContent = '删除中...';

    try {
        let resp, data;
        if (jobIds.length === 1) {
            const url = `${CONFIG.apiBaseUrl}/jobs/${encodeURIComponent(jobIds[0])}?delete_files=${deleteFiles}`;
            resp = await fetchWithTimeout(url, { method: 'DELETE' });
            // Read the body only after we know the status. A non-2xx
            // response (e.g. from a proxy HTML error page) would crash
            // resp.json() with an unhandled SyntaxError.
            data = resp.ok ? await resp.json() : { detail: resp.statusText };
        } else {
            resp = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/jobs/batch-delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_ids: jobIds, delete_files: deleteFiles }),
            });
            data = resp.ok ? await resp.json() : { detail: resp.statusText };
        }

        if (!resp.ok) {
            showToast(`删除失败: ${data.detail || resp.statusText}`, 'error');
            confirmBtn.disabled = false;
            confirmBtn.textContent = '确认删除';
            return;
        }

        // For per-job response, data is a single result dict with
        // `status` ∈ {deleted, not_found, file_error, refused}. Filter
        // the IDs we actually removed from the local cache — a not_found
        // job was already gone on the server, so dropping it locally is
        // a no-op but harmless.
        const results = jobIds.length === 1
            ? [data]
            : (data.results || []);
        const actuallyRemoved = new Set(
            results.filter(r => r.status === 'deleted').map(r => r.job_id)
        );
        for (const id of jobIds) {
            selectedJobIds.delete(id);
            if (actuallyRemoved.has(id) || results.find(r => r.job_id === id)?.status === 'not_found') {
                delete jobsData[id];
            }
        }
        renderJobsList();
        // Also refresh results table (it reads from same data source)
        if (typeof loadResults === 'function') await loadResults();

        closeDeleteModal();

        // Differentiated toast. not_found / file_error / refused used to be
        // all reported as "已删除任务 (xxx)" which misled users.
        const freed = data.bytes_freed ? `，释放 ${formatBytes(data.bytes_freed)}` : '';
        if (jobIds.length === 1) {
            const r = results[0] || {};
            if (r.status === 'not_found') {
                showToast('任务已不存在（可能已被其他操作删除）', 'info');
            } else if (r.status === 'file_error') {
                showToast(`任务记录已删除，但文件清理失败: ${r.error || ''}`, 'warning');
            } else if (r.status === 'refused') {
                showToast(`拒绝删除: ${r.error || ''}`, 'error');
            } else if (r.files_skipped) {
                // CLI-loaded job: the on-disk files live under
                // APP_ROOT/work which is shared with other CLI runs;
                // we removed the in-memory record but kept the files
                // so the user can still inspect them via the CLI.
                showToast('已从列表中移除（CLI 任务文件位于共享目录，保留在磁盘上）', 'info');
            } else {
                showToast(`已删除任务${freed}`, 'success');
            }
        } else {
            const deleted = data.deleted || 0;
            const notFound = results.filter(r => r.status === 'not_found').length;
            const fileErr = results.filter(r => r.status === 'file_error').length;
            const filesSkipped = results.filter(r => r.files_skipped).length;
            // audit 2026-07-31: refused deletions were not counted —
            // batch-deleting running jobs reported "已删除 N 个任务"
            // as if everything was removed, while the running jobs
            // stayed selected and alive.
            const refused = results.filter(r => r.status === 'refused').length;
            let suffix = '';
            if (notFound) suffix += `，${notFound} 个已不存在`;
            if (fileErr) suffix += `，${fileErr} 个文件清理失败`;
            if (filesSkipped) suffix += `，${filesSkipped} 个 CLI 任务文件保留在磁盘上`;
            if (refused) suffix += `，${refused} 个被拒绝删除（运行中或刚取消）`;
            showToast(`已删除 ${deleted} 个任务${suffix}${freed}`, deleted > 0 && !refused ? 'success' : (refused ? 'warning' : 'info'));
        }
    } catch (err) {
        // Phase F-2 M1: handle AbortError (timeout) explicitly.
        console.error('Delete failed', err);
        if (err.name === 'AbortError') {
            showToast('删除请求超时，请检查网络连接', 'error');
        } else {
            showToast(`删除失败: ${err}`, 'error');
        }
        confirmBtn.disabled = false;
        confirmBtn.textContent = '确认删除';
    }
}

function showToast(message, type = 'info') {
    // Round 10 (FH3): route through showNotification so we have a single
    // toast pipeline. The previous implementation created a NEW <div>
    // for every call with its own setTimeout, leading to DOM leaks
    // (5 deletes = 5 stale toast elements if they fire close together)
    // and an inconsistent z-index (2000 here vs. CSS-defaulted for
    // #notification). showNotification reuses the #notification element
    // and cancels any pending hide-timer from a previous toast.
    showNotification(message, type);
}

function escapeHtml(s) {
    if (s == null) return '';
    return String(s)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

// NOTE: there used to be a second ``function escapeHtml(v) { return escapeHtml(v); }``
// declaration just below this one. Function declarations hoist and overwrite, so
// the duplicate replaced the real implementation with infinite self-recursion
// that blew the JS stack on every render (60+ call sites). The duplicate is
// gone — keep this comment as a tripwire so the broken version doesn't
// silently come back through a copy-paste.

function deleteSingleJob(jobId) {
    openDeleteModal([jobId]);
}

async function viewJobDetails(jobId) {
    const modal = document.getElementById('job-details-modal');
    const content = document.getElementById('job-details-content');
    const title = document.getElementById('job-details-title');

    title.textContent = `任务详情: ${jobId.substring(0, 12)}...`;
    content.innerHTML = `
        <div class="job-details-loading">
            <div class="spinner"></div>
            <span>加载中...</span>
        </div>
    `;
    modal.classList.remove('hidden');

    try {
        const response = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/jobs/${jobId}/result`);

        if (response.status === 202) {
            content.innerHTML = `
                <div class="job-detail-section">
                    <p style="text-align: center; color: var(--text-muted); padding: 2rem;">
                        任务尚未完成，请稍后再查看详情。
                    </p>
                </div>
            `;
            return;
        }

        if (response.status === 404) {
            // Job is gone on the server (deleted, server restart, or stale id).
            // Distinguish from a generic server error so the user knows the
            // job simply no longer exists, rather than blaming the server.
            content.innerHTML = `
                <div class="job-detail-section">
                    <p style="text-align: center; color: var(--text-muted); padding: 2rem;">
                        任务不存在或已被删除
                    </p>
                </div>
            `;
            return;
        }

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.json();

        const job = jobsData[jobId] || {};
        const statusClass = `status-${job.status || 'unknown'}`;

        let html = `
            <div class="job-detail-section">
                <h3>基本信息</h3>
                <div class="job-detail-grid">
                    <div class="job-detail-item">
                        <span class="label">任务ID</span>
                        <span class="value" style="font-family: monospace;">${escapeHtml(jobId)}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">状态</span>
                        <span class="value"><span class="job-status ${escapeHtml(statusClass)}">${escapeHtml(getStatusLabel(job.status))}</span></span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">文件名</span>
                        <span class="value">${escapeHtml(job.filename || 'N/A')}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">创建时间</span>
                        <span class="value">${escapeHtml(formatDate(job.created_at))}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">进度</span>
                        <span class="value">${escapeHtml(job.progress || 0)}%</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">阶段</span>
                        <span class="value">${escapeHtml(job.stage || '—')}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">已用时</span>
                        <span class="value">${escapeHtml(job.elapsed_sec != null ? formatElapsed(job.elapsed_sec) : '—')}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">说明</span>
                        <span class="value">${escapeHtml(job.detail || '无')}</span>
                    </div>
                </div>
            </div>
        `;

        if (job.status === 'failed' && job.error) {
            // Both error and error_trace come from server-side tracebacks and
            // may contain `<`, `>`, `&` (e.g. "AttributeError: '<' not supported").
            // MUST go through escapeHtml() — they are inserted into innerHTML.
            html += `
                <div class="job-detail-section">
                    <h3>错误信息</h3>
                    <div class="job-error-message">${escapeHtml(job.error)}</div>
                    ${job.error_trace ? `<div class="job-error-trace">${escapeHtml(job.error_trace)}</div>` : ''}
                </div>
            `;
        }

        if (data.result && data.result.length > 0) {
            html += `
                <div class="job-detail-section">
                    <h3>处理结果</h3>
                    <p>共生成 <strong>${data.result.length}</strong> 条匹配记录</p>
                </div>
            `;
        }

        content.innerHTML = html;
    } catch (error) {
        // error.message may contain HTML entities from the server's
        // Pydantic validation detail (e.g. "Value error, Invalid ...").
        // MUST go through escapeHtml() — it is inserted into innerHTML.
        content.innerHTML = `
            <div class="job-detail-section">
                <p style="text-align: center; color: var(--danger-color); padding: 2rem;">
                    获取详情失败: ${escapeHtml(error.message)}
                </p>
            </div>
        `;
    }
}

async function viewJobResults(jobId) {
    // Switch to results tab and filter by job. Subtle ordering matters:
    //   * If we use ``tabBtn.click()`` to switch tabs, the tab handler will
    //     ALSO call ``loadResults()`` — racing the explicit ``await
    //     loadResults()`` below (two concurrent fetches, both rebuilding
    //     the filter, with the user's just-set filter.value at risk of
    //     being clobbered by whichever finishes second).
    //   * If we set ``filter.value`` before ``loadResults()`` returns,
    //     ``populateResultFilter()`` will rebuild the <select> and the
    //     value silently disappears.
    // Solution: switch tabs MANUALLY (without firing the click handler),
    // then await a single loadResults(), then set the filter value.
    const targetBtn = document.querySelector('[data-tab="results"]');
    if (!targetBtn) return;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(pane => pane.classList.remove('active'));
    targetBtn.classList.add('active');
    document.getElementById('results-tab')?.classList.add('active');

    const filter = document.getElementById('result-filter');
    if (!filter) return;
    await loadResults();
    // After loadResults() the option for this jobId may or may not be in
    // the select (e.g. job has no rows yet). Only set if present so the
    // user's filter doesn't silently match nothing.
    const has = Array.from(filter.options).some(opt => opt.value === jobId);
    if (has) {
        filter.value = jobId;
        resultsTableState.page = 1;
        renderResults();
    } else {
        showNotification('该任务暂无结果可显示', 'info');
    }
}

async function cancelJob(jobId, cancelBtn) {
    if (!confirm('确认取消此任务？')) return;

    const originalText = cancelBtn.innerHTML;
    cancelBtn.disabled = true;
    cancelBtn.innerHTML = '<span class="spinner-small"></span> 取消中...';

    try {
        const response = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/jobs/${jobId}/cancel`, {
            method: 'POST'
        });

        if (response.ok) {
            // The server now returns {was_running, cancelled_at, ...}; surface
            // the difference so the user knows whether the cancel was a no-op
            // (job hadn't started) or actually interrupted a running pipeline.
            try {
                const data = await response.json();
                if (data.was_running) {
                    showNotification('已中断正在运行的任务', 'success');
                } else {
                    showNotification('已取消排队中的任务', 'success');
                }
            } catch (_) {
                showNotification('任务已取消');
            }
            loadJobs();
        } else {
            // Read the server's detail (e.g. "Cannot cancel a finished job")
            // so the toast tells the user what actually went wrong.
            let detail = `取消失败 (HTTP ${response.status})`;
            try {
                const errBody = await response.json();
                if (errBody && errBody.detail) detail = `取消失败: ${errBody.detail}`;
            } catch (_) { /* body wasn't JSON */ }
            throw new Error(detail);
        }
    } catch (error) {
        // Phase F-2 M1: handle AbortError (timeout) explicitly.
        const msg = error.name === 'AbortError' ? '请求超时，请检查网络连接' : error.message;
        showNotification(msg, 'error');
        cancelBtn.disabled = false;
        cancelBtn.innerHTML = originalText;
    }
}

function startJobPolling() {
    if (refreshIntervalId) clearInterval(refreshIntervalId);
    refreshIntervalId = setInterval(() => {
        loadJobs();
    }, CONFIG.refreshInterval * 1000);
}

document.getElementById('refresh-jobs-btn')?.addEventListener('click', loadJobs);

document.getElementById('job-search')?.addEventListener('input', renderJobsList);
document.getElementById('job-filter')?.addEventListener('change', renderJobsList);

// Bulk-delete wiring
document.getElementById('jobs-select-all')?.addEventListener('change', onSelectAllToggle);
document.getElementById('delete-selected-btn')?.addEventListener('click', openDeleteModalForSelection);

// Delete-modal buttons (previously bound via inline onclick="..." which
// breaks under strict-CSP). The header X, footer Cancel and footer Confirm
// are all registered by ID here.
document.getElementById('delete-modal-close-btn')?.addEventListener('click', closeDeleteModal);
document.getElementById('delete-modal-cancel-btn')?.addEventListener('click', closeDeleteModal);
document.getElementById('delete-modal-confirm')?.addEventListener('click', confirmDelete);

// ==================== Results ==================== //
async function loadResults() {
    try {
        // audit 2026-07-31: the backend returns at most 500 rows per
        // page and its contract requires the CLIENT to paginate until
        // an empty page. The old code fetched a single page, so once
        // results exceeded 500 rows the table, stats and exports all
        // silently operated on the truncated set.
        const PAGE = 500;
        let offset = 0;
        let all = [];
        for (;;) {
            const response = await fetchWithTimeout(
                `${CONFIG.apiBaseUrl}/results?limit=${PAGE}&offset=${offset}`
            );
            if (!response.ok) {
                console.error(`loadResults: HTTP ${response.status}`);
                if (offset === 0) return;
                break;
            }
            const page = await response.json();
            if (!Array.isArray(page) || page.length === 0) break;
            all = all.concat(page);
            if (page.length < PAGE) break;
            offset += PAGE;
        }

        resultsData = all;
        // Prune stale row_ids from the persistent selection set —
        // rows deleted elsewhere (CLI, another tab) would otherwise
        // sit in the set forever and accumulate.
        const liveRowIds = new Set(resultsData.map(r => r.row_id).filter(Boolean));
        for (const rid of Array.from(selectedResultRowIds)) {
            if (!liveRowIds.has(rid)) selectedResultRowIds.delete(rid);
        }
        populateResultFilter();
        renderResults();
        updateStats();
    } catch (error) {
        // Phase F-2 M1: handle AbortError (timeout) explicitly.
        // audit 2026-07-31: a failure here used to be silent —
        // the operator kept looking at stale data as if it were
        // current. Surface it like the jobs poll does.
        console.error('Failed to load results:', error);
        if (error.name === 'AbortError') {
            showToast('结果加载超时，显示的可能不是最新数据', 'warning');
        } else {
            showToast('结果加载失败，显示的可能不是最新数据', 'error');
        }
    }
}

function populateResultFilter() {
    const filter = document.getElementById('result-filter');
    if (!filter) return;

    // Get unique job IDs from results
    const jobIds = [...new Set(resultsData.map(r => r.job_id).filter(Boolean))];

    // Remember the user's previous selection so we can either restore it
    // (the corresponding job still exists) or reset to "全部论文" (the job
    // was deleted on the server). Without this, every periodic refresh
    // wiped the filter and the user's "view this paper only" intent was
    // lost on each poll.
    const prevValue = filter.value;

    // audit 2026-08-17 (WEB-B3): the previous version short-circuited
    // when ``jobIds.length <= 1``, which meant the single-job scenario
    // (the most common state for a new operator) never rebuilt the
    // ``<select>`` and stale DOM options lingered across jobs/refreshes.
    // Always rebuild regardless of count; only the option list differs.

    // Keep the first "All papers" option
    filter.innerHTML = '<option value="">全部论文</option>';
    jobIds.forEach(jobId => {
        const shortId = jobId.substring(0, 12) + '...';
        filter.innerHTML += `<option value="${escapeHtml(jobId)}">${escapeHtml(shortId)}</option>`;
    });

    // Restore the previous selection if the corresponding option is still
    // present; otherwise reset to "全部论文". (Setting .value to a missing
    // option silently leaves the field at its previous DOM value, which
    // matches no rows — surprising behaviour we explicitly normalise here.)
    if (prevValue && jobIds.includes(prevValue)) {
        filter.value = prevValue;
    } else if (prevValue && prevValue !== '' && !jobIds.includes(prevValue)) {
        filter.value = '';
    } else {
        // Default to "全部论文" so the table shows rows immediately after
        // a fresh load (avoids the "no results" surprise when the user
        // just had a previous job-id filter that no longer matches).
        filter.value = '';
    }
}

// State for the results table. Persisted across re-renders.
const resultsTableState = {
    page: 1,
    pageSize: 25,
    sortKey: 'paper_id',
    sortDir: 'asc',
    statusFilter: 'all',  // all | image_ocr | positional | no_image
};

function getRecordStatus(r) {
    const md = r.metadata || {};
    if (md.panel_id_source === 'image_ocr') return 'image_ocr';
    if (r.panel_path) return 'positional';
    return 'no_image';
}

// Single source of truth for "which rows match the current UI state?".
// audit 2026-08-17 (WEB-B4): previously ``renderResults()`` searched
// 6 fields (paper_id / species / panel_id / figure_id / geology blob /
// caption snippet) while the export path searched 3 (paper_id /
// species / panel_id) — different filters produced different "what to
// export" vs "what to display" sets. Export now uses the SAME
// function so the .xlsx file matches the visible table.
function filterRows(rows, searchTerm, filterJob) {
    const term = (searchTerm || '').toLowerCase();
    return rows.filter(r => {
        const paperId = (r.paper_id || '').toLowerCase();
        const species = (r.species || '').toLowerCase();
        const panelId = String(r.panel_id || '').toLowerCase();
        const figureId = String(r.figure_id || '').toLowerCase();
        // Round 23 audit: expand search to cover geology fields so
        // operators can search by formation / locality / age / country.
        // The strings are pulled out of the first geology link's
        // fields (most operators care about the primary fact).
        const md = r.metadata || {};
        const gl0 = Array.isArray(md.geology_links) && md.geology_links.length > 0
            ? md.geology_links[0] : {};
        const geoBlob = [
            gl0.formation || '',
            gl0.locality || '',
            gl0.country || '',
            gl0.age || '',
            gl0.chronostratigraphy || '',
            gl0.lithology || '',
        ].join(' ').toLowerCase();
        const caption = String(r.caption_snippet || '').toLowerCase();
        const matchesSearch = !term ||
            paperId.includes(term) ||
            species.includes(term) ||
            panelId.includes(term) ||
            figureId.includes(term) ||
            geoBlob.includes(term) ||
            caption.includes(term);
        const matchesFilter = !filterJob || r.job_id === filterJob;
        const matchesStatus = resultsTableState.statusFilter === 'all'
            || getRecordStatus(r) === resultsTableState.statusFilter;
        return matchesSearch && matchesFilter && matchesStatus;
    });
}

function renderResults() {
    // Reset the record stash so indices in the new render refer to fresh data.
    __rlpeRecords = [];
    const searchTerm = document.getElementById('result-search')?.value.toLowerCase() || '';
    const filterJob = document.getElementById('result-filter')?.value || '';

    const all = resultsData;
    const filtered = filterRows(all, searchTerm, filterJob);

    // Sort
    const { sortKey, sortDir } = resultsTableState;
    const dir = sortDir === 'asc' ? 1 : -1;
    // Round 22 audit: ``_geo_age`` is a CLIENT-SIDE derived field
    // (not present in the API response) so the previous sort read
    // ``undefined`` for every row and was effectively random.
    // Compute the age from ``metadata.geology_links[0]`` here so
    // the sort actually orders rows by geological age.
    const _ageOf = (r) => {
        if (sortKey !== '_geo_age') return null;
        const md = r && r.metadata;
        const gl = md && md.geology_links;
        if (!Array.isArray(gl) || gl.length === 0) return null;
        const g = gl[0] || {};
        // Prefer ``age`` (e.g. "Late Jurassic"); fall back to
        // ``chronostratigraphy`` (e.g. "Kimmeridgian"); fall back to
        // a numeric sort by ``ma_mid`` (younger to older).
        const txt = g.age || g.chronostratigraphy;
        if (typeof txt === 'string' && txt.length) return txt;
        const ma = g.ma_mid;
        if (typeof ma === 'number') return ma;
        return null;
    };
    filtered.sort((a, b) => {
        if (sortKey === '_geo_age') {
            const av = _ageOf(a);
            const bv = _ageOf(b);
            if (av == null && bv == null) return 0;
            if (av == null) return 1;
            if (bv == null) return -1;
            // Both numbers → numeric compare; both strings → string
            // compare; mixed → numbers first (numerics are rarer).
            if (typeof av === 'number' && typeof bv === 'number') {
                return (av - bv) * dir;
            }
            if (typeof av === 'number') return -1 * dir;
            if (typeof bv === 'number') return 1 * dir;
            return String(av).localeCompare(String(bv)) * dir;
        }
        const av = a[sortKey];
        const bv = b[sortKey];
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        if (typeof av === 'number' && typeof bv === 'number') return (av - bv) * dir;
        return String(av).localeCompare(String(bv)) * dir;
    });

    // Paginate
    const total = filtered.length;
    const totalPages = Math.max(1, Math.ceil(total / resultsTableState.pageSize));
    if (resultsTableState.page > totalPages) resultsTableState.page = totalPages;
    const start = (resultsTableState.page - 1) * resultsTableState.pageSize;
    const pageItems = filtered.slice(start, start + resultsTableState.pageSize);

    const tbody = document.getElementById('results-tbody');
    if (total === 0) {
        const hasAnyResults = resultsData.length > 0;
        tbody.innerHTML = hasAnyResults
            ? '<tr class="placeholder"><td colspan="10" style="text-align: center; color: var(--text-muted);">当前筛选条件下无结果，试试清除搜索或切换筛选</td></tr>'
            : `<tr class="placeholder"><td colspan="10" style="text-align: center; color: var(--text-muted); padding: 2rem;">
                🔍 还没有提取结果<br>
                <span style="font-size: 0.85rem;">完成 PDF 处理后，结果会自动显示在这里</span>
            </td></tr>`;
        renderResultsPagination(0, 0, 0);
        renderResultsStatusFilterCounts({});
        return;
    }

    tbody.innerHTML = pageItems.map((r, idx) => {
        // Stash the full record in a window-level array and reference by index.
        // The previous approach embedded the entire JSON as a data-record
        // attribute which (a) wasted ~25 KB per page, (b) was vulnerable to
        // attribute-boundary escape when records contained '&' or '<'.
        const recordIndex = __rlpeRecords.push(r) - 1;
        const paperId = escapeHtml(r.paper_id);
        const figureId = escapeHtml(r.figure_id);
        const species = escapeHtml(r.species);
        const panelPath = escapeHtml(r.panel_path);
        const panelPathEscaped = escapeHtml(resolveAssetUrl(r.panel_path || '', r.job_id));
        const md = r.metadata || {};
        const ocrSource = md.panel_id_source;
        const oldPanelId = md.v18_old_panel_id;
        const status = getRecordStatus(r);
        const ocrCell = status === 'image_ocr'
            ? `<span class="badge badge-ok" title="Re-OCR'd from panel image${oldPanelId && oldPanelId !== r.panel_id ? ' (was: ' + escapeHtml(oldPanelId) + ')' : ''}">✓ ${escapeHtml(r.panel_id) || 'N/A'}</span>`
            : (status === 'positional'
                ? `<span class="badge badge-warn" title="panel_id came from caption list (positional); image OCR did not return a usable label">⚠ ${escapeHtml(r.panel_id) || 'N/A'}</span>`
                : `<span class="badge badge-muted" title="No panel image available for OCR verification">— ${escapeHtml(r.panel_id) || 'N/A'}</span>`);
        return `
        <tr data-row-id="${escapeHtml(r.row_id || '')}">
            <td class="col-check">
                <input type="checkbox" class="results-row-check" data-row-id="${escapeHtml(r.row_id || '')}" aria-label="选中此行">
            </td>
            <td>${escapeHtml(r.paper_id)}</td>
            <td>${escapeHtml(r.figure_id)}</td>
            <td>${escapeHtml(r.panel_id) || 'N/A'}</td>
            <td>${ocrCell}</td>
            <td>${escapeHtml(r.species) || 'N/A'}</td>
            <td>
                <span class="confidence-badge ${getConfidenceClass(r.confidence || 0)}">
                    ${((r.confidence || 0) * 100).toFixed(0)}%
                </span>
            </td>
            <td class="col-geo">
                ${(() => {
                    // Compact geology summary for the table cell:
                    // show the first link's age (and Ma range when
                    // available). Keeps the cell scannable without
                    // forcing the operator to open the modal.
                    const links = ((r.metadata && r.metadata.geology_links) || []);
                    if (!links.length) return '<span class="text-muted">—</span>';
                    const g = links[0];
                    const age = g.age || g.chronostratigraphy;
                    const ma = (g.ma_top != null && g.ma_base != null)
                        ? `<div class="col-geo-ma">${(+g.ma_top).toFixed(1)}–${(+g.ma_base).toFixed(1)} Ma</div>`
                        : '';
                    return age
                        ? `<div class="col-geo-age"><strong>${escapeHtml(age)}</strong></div>${ma}`
                        : ma || '<span class="text-muted">—</span>';
                })()}
            </td>
            <td>
                ${r.panel_path ? `<img src="${panelPathEscaped}" class="thumbnail-img" data-record-index="${recordIndex}" data-species="${species}" alt="panel thumbnail">` : 'N/A'}
            </td>
            <td>
                <button class="btn btn-small" data-correct-index="${recordIndex}">纠正</button>
            </td>
        </tr>`;
    }).join('');

    // Wire up click handlers via event delegation on tbody. The
    // previous pattern used a fresh forEach + addEventListener for every
    // render — which accumulated duplicate listeners on every page
    // navigation, search input, and filter change (audit M8: memory
    // leak + duplicate handler execution). Delegation uses ONE listener
    // on tbody that lives for the document's lifetime.
    if (!tbody.__rlpeListenersWired) {
        tbody.addEventListener('click', (ev) => {
            const img = ev.target.closest('.thumbnail-img');
            if (img) {
                const idx = parseInt(img.getAttribute('data-record-index'), 10);
                const record = __rlpeRecords[idx];
                if (record) {
                    openImageModal(img.getAttribute('src'), record.species || '', record);
                }
                return;
            }
            const btn = ev.target.closest('[data-correct-index]');
            if (btn) {
                const idx = parseInt(btn.getAttribute('data-correct-index'), 10);
                const r = __rlpeRecords[idx];
                if (r) {
                    openCorrectionModal(r.paper_id, r.figure_id, r.panel_path, r);
                }
                return;
            }
        });
        // Image-error fallback (replace broken <img> with "N/A" text).
        // Delegated on ``error`` events bubbling up from .thumbnail-img
        // children. { once: true } via attribute so it doesn't loop.
        tbody.addEventListener('error', (ev) => {
            const img = ev.target.closest && ev.target.closest('.thumbnail-img');
            if (img && img.parentNode) {
                const span = document.createElement('span');
                span.className = 'text-muted';
                span.textContent = 'N/A';
                img.replaceWith(span);
            }
        }, true);  // capture: error events don't bubble otherwise
        tbody.__rlpeListenersWired = true;
    }

    renderResultsPagination(total, resultsTableState.page, totalPages);
    const counts = {
        all: all.length,
        image_ocr: all.filter(r => getRecordStatus(r) === 'image_ocr').length,
        positional: all.filter(r => getRecordStatus(r) === 'positional').length,
        no_image: all.filter(r => getRecordStatus(r) === 'no_image').length,
    };
    renderResultsStatusFilterCounts(counts);
    // Restore checkbox state from the persistent set so a search/filter
    // change doesn't silently drop the user's selection. Then refresh the
    // batch-delete button label and the select-all checkbox.
    tbody.querySelectorAll('.results-row-check').forEach(cb => {
        const rid = cb.getAttribute('data-row-id');
        if (rid && selectedResultRowIds.has(rid)) cb.checked = true;
    });
    updateResultsDeleteButton();
    syncResultsSelectAllCheckbox();
}

// ---------------------------------------------------------------------------
// Results-tab delete (Round 16 user request)
// ---------------------------------------------------------------------------

function updateResultsDeleteButton() {
    // Update the "批量删除 (N)" label + enable/disable state.
    const btn = document.getElementById('results-delete-selected-btn');
    const counter = document.getElementById('results-delete-selected-count');
    if (!btn || !counter) return;
    const n = selectedResultRowIds.size;
    counter.textContent = `(${n})`;
    btn.disabled = n === 0;
}

function syncResultsSelectAllCheckbox() {
    // The select-all checkbox reflects the visible rows on the current
    // page only (matches the page-based navigation behaviour). Three
    // states: none checked → indeterminate off; all checked → on;
    // mixed → indeterminate on.
    const selectAll = document.getElementById('results-select-all');
    if (!selectAll) return;
    const visible = Array.from(
        document.querySelectorAll('#results-tbody .results-row-check')
    );
    const checkedVisible = visible.filter(cb => cb.checked).length;
    selectAll.checked = visible.length > 0 && checkedVisible === visible.length;
    selectAll.indeterminate = checkedVisible > 0 && checkedVisible < visible.length;
}

async function deleteAllResults() {
    // One-click delete: confirm first (destructive, no undo), then
    // DELETE /results. Reload results + jobs afterwards.
    const total = (typeof resultsData !== 'undefined') ? resultsData.length : 0;
    const msg = total > 0
        ? `确认清空全部 ${total} 条结果？此操作不可撤销（仅清空结果行，任务与磁盘文件保留）。`
        : '确认清空全部结果？此操作不可撤销。';
    if (!confirm(msg)) return;
    try {
        const resp = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/results`, { method: 'DELETE' });
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || resp.statusText);
        }
        const data = await resp.json();
        selectedResultRowIds.clear();
        showToast(`已清空 ${data.removed || 0} 条结果`, 'success');
        await loadResults();
        renderResults();
    } catch (err) {
        // Phase F-2 M1: handle AbortError (timeout) explicitly.
        const msg = err.name === 'AbortError' ? '清空请求超时' : (err.message || err);
        showToast(`清空失败: ${msg}`, 'error');
    }
}

async function deleteSelectedResults() {
    // Batch delete: pull the current selection from the persistent set,
    // confirm, then DELETE /results/batch with the row_ids payload.
    if (selectedResultRowIds.size === 0) return;
    const rowIds = Array.from(selectedResultRowIds);
    if (!confirm(`确认删除选中的 ${rowIds.length} 条结果？此操作不可撤销。`)) return;
    try {
        const resp = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/results/batch`, {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ row_ids: rowIds }),
        });
        if (!resp.ok) {
            const data = await resp.json().catch(() => ({}));
            throw new Error(data.detail || resp.statusText);
        }
        const data = await resp.json();
        selectedResultRowIds.clear();
        const notFound = data.not_found || 0;
        showToast(
            `已删除 ${data.removed || 0} 条结果${notFound > 0 ? `（${notFound} 条未匹配）` : ''}`,
            'success'
        );
        await loadResults();
        renderResults();
    } catch (err) {
        // Phase F-2 M1: handle AbortError (timeout) explicitly.
        const msg = err.name === 'AbortError' ? '删除请求超时' : (err.message || err);
        showToast(`删除失败: ${msg}`, 'error');
    }
}

function initResultsDeleteButtons() {
    // Wire the four new controls. Wired once via the same
    // __rlpeXxxWired flag pattern as the rest of the module so
    // DOMContentLoaded + subsequent re-renders don't pile up handlers.
    const selectAll = document.getElementById('results-select-all');
    if (selectAll && !selectAll.__rlpeWired) {
        selectAll.addEventListener('change', () => {
            const visible = Array.from(
                document.querySelectorAll('#results-tbody .results-row-check')
            );
            if (selectAll.checked) {
                visible.forEach(cb => {
                    cb.checked = true;
                    const rid = cb.getAttribute('data-row-id');
                    if (rid) selectedResultRowIds.add(rid);
                });
            } else {
                visible.forEach(cb => {
                    cb.checked = false;
                    const rid = cb.getAttribute('data-row-id');
                    if (rid) selectedResultRowIds.delete(rid);
                });
            }
            updateResultsDeleteButton();
        });
        selectAll.__rlpeWired = true;
    }
    // Delegated listener on tbody so per-row checkbox changes update
    // the persistent set + button state. Wired once.
    const tbody = document.getElementById('results-tbody');
    if (tbody && !tbody.__rlpeCheckWired) {
        tbody.addEventListener('change', (ev) => {
            const cb = ev.target.closest('.results-row-check');
            if (!cb) return;
            const rid = cb.getAttribute('data-row-id');
            if (!rid) return;
            if (cb.checked) selectedResultRowIds.add(rid);
            else selectedResultRowIds.delete(rid);
            updateResultsDeleteButton();
            syncResultsSelectAllCheckbox();
        });
        tbody.__rlpeCheckWired = true;
    }
    const delAll = document.getElementById('results-delete-all-btn');
    if (delAll && !delAll.__rlpeWired) {
        delAll.addEventListener('click', deleteAllResults);
        delAll.__rlpeWired = true;
    }
    const delSel = document.getElementById('results-delete-selected-btn');
    if (delSel && !delSel.__rlpeWired) {
        delSel.addEventListener('click', deleteSelectedResults);
        delSel.__rlpeWired = true;
    }
}

function renderResultsStatusFilterCounts(counts) {
    const container = document.getElementById('result-status-filters');
    if (!container) return;
    const cur = resultsTableState.statusFilter;
    const buttons = [
        ['all', '全部', counts.all || 0],
        ['image_ocr', '✓ 图像 OCR', counts.image_ocr || 0],
        ['positional', '⚠ 位置回退', counts.positional || 0],
        ['no_image', '— 无图', counts.no_image || 0],
    ];
    container.innerHTML = buttons.map(([k, label, n]) =>
        `<button type="button" role="tab" class="status-filter-btn ${k === cur ? 'active' : ''}" data-status="${escapeHtml(k)}" aria-selected="${k === cur ? 'true' : 'false'}">${label} <span class="status-filter-count">${n}</span></button>`
    ).join('');
    // Round 10 (FM4): the previous code attached a fresh click listener
    // to every button on every render — duplicate handlers accumulated
    // with each search/filter/page change. Use ONE delegated listener
    // (matching the same pattern in renderResultsPagination below) so
    // the listener is wired exactly once for the container's lifetime.
    if (!container.__rlpeFilterWired) {
        container.addEventListener('click', (ev) => {
            const btn = ev.target.closest('.status-filter-btn');
            if (!btn) return;
            const next = btn.getAttribute('data-status');
            if (!next || next === resultsTableState.statusFilter) return;
            resultsTableState.statusFilter = next;
            resultsTableState.page = 1;
            renderResults();
        });
        container.__rlpeFilterWired = true;
    }
}

function renderResultsPagination(total, page, totalPages) {
    const container = document.getElementById('results-pagination');
    if (!container) return;
    if (total === 0) {
        container.innerHTML = '<span class="pagination-info">无结果</span>';
        return;
    }
    const start = (page - 1) * resultsTableState.pageSize + 1;
    const end = Math.min(page * resultsTableState.pageSize, total);
    container.innerHTML = `
        <span class="pagination-info">第 ${start}-${end} / 共 ${total} 条 · 第 ${page}/${totalPages} 页</span>
        <button class="btn btn-small" id="page-first" ${page === 1 ? 'disabled' : ''}>« 首页</button>
        <button class="btn btn-small" id="page-prev" ${page === 1 ? 'disabled' : ''}>‹ 上一页</button>
        <button class="btn btn-small" id="page-next" ${page === totalPages ? 'disabled' : ''}>下一页 ›</button>
        <button class="btn btn-small" id="page-last" ${page === totalPages ? 'disabled' : ''}>末页 »</button>
        <select id="page-size-select" class="page-size-select">
            <option value="10">10 / 页</option>
            <option value="25" selected>25 / 页</option>
            <option value="50">50 / 页</option>
            <option value="100">100 / 页</option>
        </select>`;
    // Stash the current total BEFORE wiring the listener so the
    // first click can already compute totalPages correctly. Re-stashed
    // on every subsequent render so filter / search updates are seen.
    container.__lastTotal = total;
    // Audit M9: the old code called addEventListener on freshly-
    // recreated button elements every render — duplicate listeners
    // accumulated on each page navigation. The pagination container
    // is stable across renders (only its innerHTML changes), so we
    // use event delegation: one click handler on the container
    // looks up the button id from the event target. Wired ONCE.
    if (!container.__paginationWired) {
        container.addEventListener('click', (ev) => {
            const btn = ev.target.closest('button[id^="page-"]');
            if (!btn || btn.disabled) return;
            const cur = resultsTableState.page;
            // Re-derive totalPages from the latest __lastTotal so that
            // filter changes that shrink the dataset don't allow
            // navigation past the new last page.
            const totalP = Math.max(1, Math.ceil(
                (container.__lastTotal || 0) / resultsTableState.pageSize,
            ));
            if (btn.id === 'page-first') resultsTableState.page = 1;
            else if (btn.id === 'page-prev') resultsTableState.page = Math.max(1, cur - 1);
            else if (btn.id === 'page-next') resultsTableState.page = Math.min(totalP, cur + 1);
            else if (btn.id === 'page-last') resultsTableState.page = totalP;
            else return;
            renderResults();
        });
        container.addEventListener('change', (ev) => {
            const sel = ev.target.closest('#page-size-select');
            if (!sel) return;
            resultsTableState.pageSize = parseInt(sel.value, 10) || 25;
            resultsTableState.page = 1;
            renderResults();
        });
        container.__paginationWired = true;
    }
    const ps = document.getElementById('page-size-select');
    if (ps) {
        ps.value = String(resultsTableState.pageSize);
    }
}

function getConfidenceClass(confidence) {
    const c = confidence || 0;
    if (c >= 0.8) return 'confidence-high';
    if (c >= 0.5) return 'confidence-medium';
    return 'confidence-low';
}

function updateStats() {
    const stats = {
        total: resultsData.length,
        high_confidence: resultsData.filter(r => r.confidence >= 0.8).length,
        species_matched: resultsData.filter(r => r.species).length,
        unique_species: new Set(resultsData.map(r => r.species).filter(s => s)).size,
        papers: new Set(resultsData.map(r => r.paper_id).filter(Boolean)).size,
        image_ocr: resultsData.filter(r => getRecordStatus(r) === 'image_ocr').length,
        positional: resultsData.filter(r => getRecordStatus(r) === 'positional').length,
        no_image: resultsData.filter(r => getRecordStatus(r) === 'no_image').length,
    };
    const imgOcrPct = stats.total > 0 ? (stats.image_ocr / stats.total * 100) : 0;

    const statsHtml = `
        <div class="stat-card">
            <div class="stat-label">总匹配数</div>
            <div class="stat-value">${stats.total}</div>
            <div class="stat-sub">${stats.papers} 篇论文</div>
        </div>
        <div class="stat-card secondary">
            <div class="stat-label">高置信度</div>
            <div class="stat-value">${stats.high_confidence}</div>
            <div class="stat-sub">≥ 80% 置信度</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">已识别物种</div>
            <div class="stat-value">${stats.unique_species}</div>
            <div class="stat-sub">${stats.species_matched} 个 panel 命中</div>
        </div>
        <div class="stat-card warning">
            <div class="stat-label">图像 OCR 命中</div>
            <div class="stat-value">${stats.image_ocr}</div>
            <div class="stat-sub">${imgOcrPct.toFixed(1)}% · ⚠ ${stats.positional} 位置回退</div>
        </div>
    `;

    const statsContainer = document.getElementById('results-stats');
    if (statsContainer) {
        statsContainer.innerHTML = statsHtml;
    }
}

document.getElementById('result-search')?.addEventListener('input', () => { resultsTableState.page = 1; renderResults(); });
document.getElementById('result-filter')?.addEventListener('change', () => { resultsTableState.page = 1; renderResults(); });

// Sort-by-column: clicking a th[data-sort-key] toggles asc/desc.
function updateSortIndicators() {
    document.querySelectorAll('#results-table th[data-sort-key]').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
        if (th.getAttribute('data-sort-key') === resultsTableState.sortKey) {
            th.classList.add(resultsTableState.sortDir === 'asc' ? 'sort-asc' : 'sort-desc');
        }
    });
}
document.querySelectorAll('#results-table th[data-sort-key]').forEach(th => {
    th.addEventListener('click', () => {
        const key = th.getAttribute('data-sort-key');
        if (resultsTableState.sortKey === key) {
            resultsTableState.sortDir = resultsTableState.sortDir === 'asc' ? 'desc' : 'asc';
        } else {
            resultsTableState.sortKey = key;
            resultsTableState.sortDir = 'asc';
        }
        updateSortIndicators();
        renderResults();
    });
    th.classList.add('sortable');
    th.setAttribute('title', `点击按 ${th.textContent.trim()} 排序`);
});
updateSortIndicators();

// Apply the same filter / search / statusFilter as the rendered table.
// audit 2026-08-17 (WEB-B4): now delegates to the shared ``filterRows``
// helper so the export path and the table renderer can never drift
// apart again (different filter logic used to silently export rows the
// operator had hidden).
function getFilteredResults() {
    const searchTerm = document.getElementById('result-search')?.value.toLowerCase() || '';
    const filterJob = document.getElementById('result-filter')?.value || '';
    return filterRows(resultsData, searchTerm, filterJob);
}

// CSV cell formatter: handles null/undefined, escapes embedded quotes by
// doubling them (RFC 4180), wraps in double quotes.
function csvCell(v) {
    if (v == null) return '""';
    const s = String(v);
    return `"${s.replace(/"/g, '""')}"`;
}

document.getElementById('export-btn')?.addEventListener('click', async () => {
    // Round 24: replaced client-side CSV with a backend Excel
    // (multi-sheet .xlsx via openpyxl). The endpoint
    // ``GET /jobs/{job_id}/export.xlsx`` returns a 5-sheet
    // workbook (panels / geology_contexts / localities /
    // paleo_coordinates / legend). The frontend iterates over
    // the filtered rows, groups by ``job_id`` (since each job is
    // a separate .xlsx), and downloads one file per job.
    //
    // If only one job is in the filtered set, the file is
    // downloaded directly. If multiple, the user is asked to
    // download each one.
    //
    // audit 2026-08-17 (WEB-B5): we now thread the active UI
    // filter to the server via ``paper_ids`` / ``species`` /
    // ``panel_ids`` / ``search`` query params so the downloaded
    // .xlsx mirrors the operator's visible table. Pre-fix the
    // server always exported the FULL job (ignoring the UI
    // filter), silently disagreeing with what the user could see.
    const rows = getFilteredResults();
    if (rows.length === 0) {
        showNotification('当前筛选下没有可导出的结果');
        return;
    }
    // Group rows by job_id. The frontend caches job_id on each
    // row (set in get_results()).
    const jobIds = new Set();
    for (const r of rows) {
        if (r.job_id) jobIds.add(r.job_id);
    }
    const searchTerm = document.getElementById('result-search')?.value || '';
    showNotification(`正在导出 ${rows.length} 条结果 (${jobIds.size} 个 job) ...`);
    for (const jobId of jobIds) {
        try {
            // Narrow the per-job filter to ONLY the rows that
            // belong to this job. Per-job paper/species/panel
            // lists keep the URL small even when the global
            // filter spans many jobs.
            const jobRows = rows.filter(r => r.job_id === jobId);
            const qs = new URLSearchParams();
            const paperIds = [...new Set(jobRows.map(r => r.paper_id).filter(Boolean))];
            const speciesList = [...new Set(jobRows.map(r => r.species).filter(Boolean))];
            const panelIds = [...new Set(jobRows.map(r => r.panel_id).filter(p => p != null && p !== ''))];
            if (paperIds.length) qs.set('paper_ids', paperIds.join(','));
            if (speciesList.length) qs.set('species', speciesList.join(','));
            if (panelIds.length) qs.set('panel_ids', panelIds.map(String).join(','));
            if (searchTerm) qs.set('search', searchTerm);
            const url = `${CONFIG.apiBaseUrl}/jobs/${jobId}/export.xlsx${qs.toString() ? '?' + qs.toString() : ''}`;
            // audit 2026-07-31: the URL was hard-coded relative while every
    // other request honours CONFIG.apiBaseUrl — a custom API origin
    // made export fail against the page origin.
            // Phase F-2 (M-1): route through fetchWithTimeout so a hung
            // backend can't leave the export UI hanging.
            const resp = await fetchWithTimeout(url, {
                method: 'GET',
                credentials: 'same-origin',
            });
            if (!resp.ok) {
                const err = await resp.json().catch(() => ({}));
                // audit 2026-07-31: the boolean 'true' was passed as the type
                // class — CSS has no '.true' rule so failures rendered
                // with the GREEN success style.
                showNotification(`导出 ${jobId} 失败: ${err.detail || resp.status}`, 'error');
                continue;
            }
            const blob = await resp.blob();
            // Filename comes from the Content-Disposition header
            // (the server sets ``Content-Disposition: attachment;
            // filename="rlpe_<paper_id>_<job_id>.xlsx"``). Fall
            // back to a client-side timestamp if the header is
            // missing (older server versions).
            const dispo = resp.headers.get('Content-Disposition') || '';
            const m = dispo.match(/filename="([^"]+)"/);
            const filename = m ? m[1] : `rlpe_export_${jobId}.xlsx`;
            const url2 = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url2;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            setTimeout(() => URL.revokeObjectURL(url2), 1000);
        } catch (err) {
            showNotification(`导出 ${jobId} 异常: ${err}`, 'error');
        }
    }
});

// ==================== Image Modal ==================== //
// Rich modal opener that shows all record fields. Called directly from
// the results table with the full prediction record.
function openImageModal(src, title, record) {
    const modal = document.getElementById('image-modal');
    const img = document.getElementById('modal-image');
    const info = document.getElementById('modal-info');

    img.src = src;
    img.alt = title || 'Panel image';

    if (!record) {
        info.innerHTML = `<strong>物种:</strong> ${escapeHtml(title || 'Unknown')}`;
    } else {
        const md = record.metadata || {};
        const ocrSource = md.panel_id_source;
        const oldPanelId = md.v18_old_panel_id;
        const ocrBadge = ocrSource === 'image_ocr'
            ? `<span class="badge badge-ok" title="panel_id anchored to label visible in the panel image">✓ 图像 OCR</span>`
            : (record.panel_path
                ? `<span class="badge badge-warn" title="panel_id came from caption list (positional); image OCR did not return a usable label">⚠ 位置回退</span>`
                : `<span class="badge badge-muted" title="No panel image available for verification">— 无图</span>`);
        const conf = (record.confidence || 0).toFixed(2);
        const reassignNote = (oldPanelId && oldPanelId !== record.panel_id)
            ? `<div class="modal-row"><span class="modal-label">v17 → v18:</span> <code>${escapeHtml(oldPanelId)}</code> → <code>${escapeHtml(record.panel_id)}</code></div>`
            : '';
        const captionSnippet = (record.caption_snippet || '').slice(0, 280);
        // Round 22 audit: surface paper metadata (title / authors /
        // journal) when present. Without this, the operator has to
        // open DevTools to see why a paper has ``title=None`` or
        // ``authors=[]`` (Round 20 cleanup flags).
        const paperMeta = record.paper_metadata || {};
        const paperTitle = paperMeta.title;
        const paperAuthors = Array.isArray(paperMeta.authors) ? paperMeta.authors : [];
        const paperJournal = paperMeta.journal;
        const paperYear = paperMeta.year;
        const paperReviewReasons = Array.isArray(paperMeta.review_reasons) ? paperMeta.review_reasons : [];
        // Round 22 audit: ``geology_scope`` (Round 19) tells the
        // operator whether the geology data is panel-specific
        // (extracted from this panel's caption), figure-anchor
        // (first panel inheriting figure-level data), or none
        // (no geology found). Surface as a colored badge.
        const geoScope = md.geology_scope;
        const scopeBadge = (() => {
            if (!geoScope) return '';
            const cls = ({
                'panel': 'badge-ok',
                'figure_anchor': 'badge-warn',
                'none': 'badge-muted',
            })[geoScope] || 'badge-muted';
            const labels = {
                'panel': 'Panel 专属',
                'figure_anchor': '图级锚定',
                'none': '无地质',
            };
            const titleMap = {
                'panel': '本 panel 的 caption 抽取到的地质信息',
                'figure_anchor': '第一个 panel 继承图级 caption 的地质',
                'none': '未找到该 panel 的地质数据',
            };
            return ` <span class="badge ${cls}" title="${escapeHtml(titleMap[geoScope] || geoScope)}">${escapeHtml(labels[geoScope] || geoScope)}</span>`;
        })();
        // Round 22 audit: display sample IDs (Round 21 prefix-tagged:
        // S_ legacy, B_ Boughdiri-style, R_ specimen N, N_ numeric,
        // L_ (N) numbered list, P_ pl. N). The pipeline's
        // ``metadata.geology_links`` may carry inline sample
        // references; ``samples`` is at top-level
        // ``record.samples`` only via the legacy API path — for
        // now, we surface what we have.
        const geoSampleIds = (md.geology_links || [])
            .map(g => g && g.sample_id)
            .filter(s => s && typeof s === 'string');
        const paperBlock = (paperTitle || paperAuthors.length || paperJournal || paperYear)
            ? `<div class="modal-row modal-row-wide"><span class="modal-label">论文元数据:</span>
                    <div class="modal-paper-meta">
                        ${paperTitle ? `<div><strong>${escapeHtml(paperTitle)}</strong>${paperReviewReasons.length ? ` <span class="badge badge-warn" title="${escapeHtml(paperReviewReasons.join('; '))}">⚠ ${escapeHtml(paperReviewReasons[0])}</span>` : ''}</div>` : '<div class="text-muted">(title 未抽取)</div>'}
                        ${paperAuthors.length ? `<div class="text-muted">作者: ${escapeHtml(paperAuthors.slice(0, 5).join('; '))}${paperAuthors.length > 5 ? ' …' : ''}</div>` : ''}
                        ${paperJournal ? `<div class="text-muted">期刊: ${escapeHtml(paperJournal)}${paperYear ? ` (${paperYear})` : ''}</div>` : ''}
                    </div>
                </div>`
            : '';
        const sampleBlock = geoSampleIds.length
            ? `<div class="modal-row modal-row-wide"><span class="modal-label">Sample IDs:</span> <code>${escapeHtml(geoSampleIds.join(', '))}</code></div>`
            : '';
        info.innerHTML = `
            <div class="modal-grid">
                <div class="modal-row"><span class="modal-label">论文 ID:</span> <code>${escapeHtml(record.paper_id || 'N/A')}</code></div>
                <div class="modal-row"><span class="modal-label">图版 ID:</span> <code>${escapeHtml(record.figure_id || 'N/A')}</code></div>
                <div class="modal-row"><span class="modal-label">Panel 标签:</span> <code>${escapeHtml(record.panel_id || 'N/A')}</code></div>
                <div class="modal-row"><span class="modal-label">Panel 来源:</span> ${ocrBadge}</div>
                <div class="modal-row"><span class="modal-label">物种:</span> <strong>${escapeHtml(record.species || 'N/A')}</strong></div>
                <div class="modal-row"><span class="modal-label">置信度:</span> ${((record.confidence || 0) * 100).toFixed(0)}%</div>
                ${geoScope ? `<div class="modal-row"><span class="modal-label">地质范围:</span> ${scopeBadge}</div>` : ''}
                ${reassignNote}
                ${record.bbox && Array.isArray(record.bbox) && record.bbox.some(v => v > 0) ? `<div class="modal-row"><span class="modal-label">BBox:</span> <code>[${record.bbox.map(v => escapeHtml(String(v))).join(', ')}]</code></div>` : ''}
                ${paperBlock}
                ${sampleBlock}
                ${captionSnippet ? `<div class="modal-row modal-row-wide"><span class="modal-label">Caption:</span><div class="modal-caption">${escapeHtml(captionSnippet)}${captionSnippet.length >= 280 ? '…' : ''}</div></div>` : ''}
                ${(() => {
                    // Render geology_links (age / formation / locality / Ma range /
                    // lithology / member / group / biozone) so the operator can see
                    // WHY a panel got a given species prediction. Skip silently if
                    // the metadata is missing or empty.
                    //
                    // Ma range format: "ma_top–ma_base Ma" (e.g. "251.90–254.14 Ma").
                    // Lithology / member / group / biozone are optional and
                    // rendered only when present so the existing simple captions
                    // (where only age + formation + locality are known) still look
                    // clean.
                    const links = (record.metadata && record.metadata.geology_links) || [];
                    if (!links.length) return '';
                    const fmtMa = (g) => {
                        const t = g.ma_top, b = g.ma_base;
                        if (t == null || b == null) return '';
                        return `<span class="modal-geo-ma">${(+t).toFixed(2)}–${(+b).toFixed(2)} Ma</span>`;
                    };
                    const items = links.map(g => {
                        const age = g.age || g.chronostratigraphy;
                        // Round 22 audit: use ``modern_latitude`` /
                        // ``modern_longitude`` (the schema's canonical
                        // names). The legacy ``latitude`` /
                        // ``longitude`` are also present but the
                        // frontend was reading them via
                        // ``g.latitude`` / ``g.longitude`` which were
                        // always None in API responses (the converter
                        // only emits the modern_* fields).
                        const modLat = g.modern_latitude;
                        const modLon = g.modern_longitude;
                        const paleoLat = g.paleo_latitude;
                        const paleoLon = g.paleo_longitude;
                        const head = [
                            age ? `<strong>${escapeHtml(age)}</strong>` : '',
                            fmtMa(g),
                            g.lithology ? `<span>${escapeHtml(g.lithology)}</span>` : '',
                            g.formation ? `<em>${escapeHtml(g.formation)}</em>` : '',
                            g.member ? `<span>${escapeHtml(g.member)}</span>` : '',
                            g.group ? `<span>${escapeHtml(g.group)}</span>` : '',
                            g.biozone ? `<span>${escapeHtml(g.biozone)}</span>` : '',
                            g.locality ? `<span>${escapeHtml(g.locality)}</span>` : '',
                            g.country ? `<span>${escapeHtml(g.country)}</span>` : '',
                            (modLat != null && modLon != null) ?
                                `<span class="modal-geo-coord">now ${(+modLat).toFixed(3)}, ${(+modLon).toFixed(3)}</span>` : '',
                            (paleoLat != null && paleoLon != null) ?
                                `<span class="modal-geo-paleo">@${(+paleoLat).toFixed(3)}, ${(+paleoLon).toFixed(3)}</span>` : '',
                            g.coord_source === 'country_centroid' ?
                                `<span class="modal-geo-source">[centroid]</span>` : ''
                        ].filter(Boolean).join(' · ');
                        if (!head) return '';
                        const conf = (g.confidence != null) ?
                            ` <span class="modal-geo-conf">(${(g.confidence * 100).toFixed(0)}%)</span>` : '';
                        return `<li>${head}${conf}</li>`;
                    }).filter(Boolean);
                    if (!items.length) return '';
                    // Round 18 audit: include paleo coordinates + plate
                    // ID when present. We surface them inline with the
                    // geology list and add a collapsible "evidence"
                    // block per link so the operator can see WHICH
                    // sentence the regex / vision extractor pulled
                    // the data from.
                    const extras = links.map(g => {
                        const paleo = (g.paleo_latitude != null && g.paleo_longitude != null)
                            ? `<span class="modal-geo-paleo" title="Reconstructed paleo position">~${(+g.paleo_latitude).toFixed(1)}, ${(+g.paleo_longitude).toFixed(1)} (${escapeHtml(g.plate_id || '?')} @ ${g.reconstruction_age_ma ?? '?'} Ma)</span>`
                            : '';
                        const modern = (g.modern_latitude != null && g.modern_longitude != null)
                            ? `<span class="modal-geo-modern" title="Modern coordinates">now ${(+g.modern_latitude).toFixed(2)}, ${(+g.modern_longitude).toFixed(2)}</span>`
                            : '';
                        const ev = g.evidence_text
                            ? `<details class="modal-geo-evidence"><summary>📄 提取证据</summary><pre>${escapeHtml(g.evidence_text)}</pre></details>`
                            : '';
                        return { paleo, modern, ev };
                    });
                    const extrasHtml = extras
                        .map(e => [e.paleo, e.modern, e.ev].filter(Boolean).join(' '))
                        .filter(Boolean)
                        .join('');
                    return `<div class="modal-row modal-row-wide"><span class="modal-label">地质关联:</span><ul class="modal-geo-list">${items.join('')}</ul>${extrasHtml}</div>`;
                })()}
            </div>`;
    }
    modal.classList.remove('hidden');
}

// Attach close handlers to all modal close buttons
document.querySelectorAll('.modal-close').forEach(btn => {
    btn.addEventListener('click', () => {
        btn.closest('.modal').classList.add('hidden');
    });
});

// Close modals when clicking on overlay background
['image-modal', 'job-details-modal', 'correction-modal'].forEach(modalId => {
    document.getElementById(modalId)?.addEventListener('click', (e) => {
        if (e.target === e.currentTarget) {
            e.currentTarget.classList.add('hidden');
        }
    });
});

// Round 10 (FM1): Escape key closes any open modal. WCAG 2.1 SC 2.1.1
// requires a keyboard-only way to dismiss modal dialogs; pre-fix the
// user could only click the × button, the overlay background, or the
// footer Cancel button. The MiniMax fallback modal deliberately does
// NOT close on Escape or overlay click (line ~864) — that's a "force a
// choice" UX — so we exclude it by id check below.
document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    const NON_DISMISSABLE = new Set(['MiniMax-fallback-modal']);
    document.querySelectorAll('.modal:not(.hidden)').forEach(modal => {
        if (NON_DISMISSABLE.has(modal.id)) return;
        modal.classList.add('hidden');
    });
});

// ==================== Correction Modal ==================== //
function openCorrectionModal(paperId, figureId, panelPath, record) {
    // Defensive: if the record was missing paperId / figureId (e.g. an
    // extremely old result row), the dataset would receive the literal
    // string "undefined", which the backend would happily accept and
    // persist as a dirty review row. Bail early instead.
    if (!paperId || !figureId || paperId === 'undefined' || figureId === 'undefined') {
        showNotification('该记录缺少 paper_id / figure_id，无法提交纠正', 'error');
        return;
    }
    const modal = document.getElementById('correction-modal');
    // Reset the form before re-opening so values from a previous
    // session (corrected_species, reviewer name, notes) don't bleed
    // into the new correction. Without this, opening the modal on
    // record B after submitting on record A would pre-fill B with
    // A's text and submit a wrong review row.
    const form = document.getElementById('correction-form');
    if (form) form.reset();
    const speciesInput = document.getElementById('corrected-species');
    speciesInput.dataset.paperId = paperId;
    speciesInput.dataset.figureId = figureId;
    speciesInput.dataset.panelPath = panelPath || '';
    // audit 2026-08-17 (WEB-B7): pre-populate the image_verified
    // checkbox from the row's current value so the operator can see
    // whether the row has already been verified and decide whether
    // their correction should re-verify it. Default unchecked + the
    // hidden "omit" value means a regular text correction does NOT
    // silently clobber the existing image_verified flag.
    const ivCheckbox = document.getElementById('correction-image-verified');
    if (ivCheckbox) {
        const currentVer = record && record.metadata
            ? record.metadata.image_verified === true
            : false;
        ivCheckbox.checked = !!currentVer;
    }
    modal.classList.remove('hidden');
}

function closeCorrectionModal() {
    document.getElementById('correction-modal')?.classList.add('hidden');
}

// Close correction modal via close button or cancel button (both are
// now registered by ID instead of the previous inline onclick).
document.getElementById('correction-modal-cancel-btn')?.addEventListener('click', closeCorrectionModal);
document.querySelector('#correction-modal .modal-close')?.addEventListener('click', closeCorrectionModal);

document.getElementById('correction-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();

    const speciesInput = document.getElementById('corrected-species');
    const paperId = speciesInput.dataset.paperId;
    const figureId = speciesInput.dataset.figureId;
    // Re-validate at submit time too — the user might have edited the form
    // long after the modal opened if some other JS reset the dataset.
    if (!paperId || !figureId || paperId === 'undefined' || figureId === 'undefined') {
        showNotification('记录数据不完整，无法提交', 'error');
        return;
    }
    const payload = {
        paper_id: paperId,
        figure_id: figureId,
        panel_path: speciesInput.dataset.panelPath || null,
        corrected_species: document.getElementById('corrected-species').value,
        corrected_label: document.getElementById('corrected-label').value,
        reviewer: document.getElementById('reviewer-name').value,
        notes: document.getElementById('correction-notes').value
    };
    // audit 2026-08-17 (WEB-B7): include image_verified so a reviewer
    // can flip the verified bit as part of a correction. Always send
    // it (true OR false) — the previous behaviour omitted the field
    // entirely which meant operators had no UI to un-verify a row.
    const ivEl = document.getElementById('correction-image-verified');
    if (ivEl) {
        payload.image_verified = !!ivEl.checked;
    }

    try {
        const response = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/review/correction`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            // audit 2026-07-31: the correction is consumed on the NEXT
    // pipeline run (corrections.jsonl overlay) — the current table
    // still shows the old value. Saying just "submitted" implied it
    // was applied, which misled operators into trusting stale rows.
    showNotification('纠正已保存，将在下次运行时生效');
            closeCorrectionModal();
            document.getElementById('correction-form').reset();
        } else {
            // Surface the server's detail (e.g. "paper_id: field required")
            // so the user can fix the form instead of seeing a generic error.
            let detail = '提交失败';
            try {
                const errBody = await response.json();
                if (errBody && errBody.detail) detail = `提交失败: ${errBody.detail}`;
            } catch (_) { /* body wasn't JSON */ }
            throw new Error(detail);
        }
    } catch (error) {
        // Phase F-2 M1: handle AbortError (timeout) explicitly.
        const msg = error.name === 'AbortError' ? '提交失败: 请求超时' : error.message;
        showNotification(msg, 'error');
    }
});

// ==================== Settings ==================== //
document.getElementById('save-settings-btn')?.addEventListener('click', () => {
    const apiUrl = document.getElementById('api-base-url').value;
    const refreshRaw = document.getElementById('refresh-interval').value;
    // Validate refresh interval before saving — an empty / non-numeric /
    // out-of-range value would otherwise bake NaN into CONFIG and break
    // setInterval (NaN ms = "fire as fast as possible", or never).
    const refreshSec = parseInt(refreshRaw, 10);
    if (isNaN(refreshSec) || refreshSec < 1 || refreshSec > 600) {
        showNotification(`刷新间隔必须是 1..600 的整数，当前值: "${refreshRaw}"`, 'error');
        return;
    }

    // Quota / private-mode safe — see _safeStorageSet.
    _safeStorageSet('apiBaseUrl', apiUrl);
    _safeStorageSet('refreshInterval', String(refreshSec));

    CONFIG.apiBaseUrl = apiUrl;
    CONFIG.refreshInterval = refreshSec;
    // Restart the polling loop at the new interval. Without this, a
    // user changing the value from 3 → 30 would still see poll ticks
    // every 3s (the original setInterval keeps running).
    if (refreshIntervalId) {
        startJobPolling();
    }

    showNotification('设置已保存');
});

// ==================== Initialization ==================== //
document.addEventListener('DOMContentLoaded', () => {
    // Load saved settings
    document.getElementById('api-base-url').value = CONFIG.apiBaseUrl;
    document.getElementById('refresh-interval').value = CONFIG.refreshInterval;

    // Load system info
    loadSystemInfo();

    // Check API health
    checkApiHealth();
    setInterval(checkApiHealth, 10000);

    // Load initial data
    loadJobs();
    loadResults();

    // ============== UX upgrades (#PR — default-MiniMax + onboarding) ==============
    // 1) Show the first-time onboarding banner unless dismissed
    initOnboardingBanner();
    // 2) Wire up the basic/advanced view toggle
    initViewToggle();
    // 3) Sync the basic-view LLM backend dropdown with the advanced-view one
    initLLMBackendSync();
    // 4) Wire up the API key show/hide toggle
    initApiKeyToggle();
    // 5) Wire up the LLM status card (poll once now and after each upload)
    refreshLLMStatus();
    // 6) Wire up the "Test connection" button
    document.getElementById('llm-test-btn')?.addEventListener('click', testLLMConnection);
    // 7) Wire up the cost estimate update on file change
    initCostEstimate();
    // 8) Show MiniMax usage in settings tab
    refreshMiniMaxUsage();
    // 9) Wire up the results-tab batch delete + one-click delete (Round 16)
    initResultsDeleteButtons();
});

// Auto-restart polling when the page becomes visible (tab switch / window
// refocus). The previous design stopped polling once all jobs settled and
// only restarted on a fresh upload; if a user ran a task, switched to
// another browser tab, then came back after the task completed, the "jobs"
// tab showed stale data until they manually clicked "刷新" or re-uploaded.
document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
        loadJobs();
        loadResults();
    }
});

// Warn the user before closing/navigating away while a job is actively
// running. The pipeline continues server-side regardless, but a novice
// user who accidentally closes the tab loses track of their job and may
// re-upload the same PDF (wasting API calls). The check is lightweight
// (reads the in-memory cache, no network call) and only fires when the
// page is actually being unloaded.
window.addEventListener('beforeunload', (e) => {
    const hasActive = Object.values(jobsData).some(
        j => j.status === 'queued' || j.status === 'running' || j.status === 'awaiting_user_decision'
    );
    if (hasActive || _processing) {
        e.preventDefault();
        e.returnValue = '';
    }
});

// Also restart polling when the window regains focus (e.g. Alt+Tab back).
window.addEventListener('focus', () => {
    loadJobs();
    loadResults();
});

async function loadSystemInfo() {
    try {
        const response = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/system/info`);
        if (!response.ok) return;

        const info = await response.json();
        const infoDiv = document.getElementById('system-info');
        if (infoDiv) {
            infoDiv.innerHTML = `
                <div class="info-row">
                    <span class="info-label">RLPE 版本:</span>
                    <span class="info-value">${info.version || 'N/A'}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">GROBID 服务:</span>
                    <span class="info-value">${info.grobid_url || 'N/A'}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">Python 版本:</span>
                    <span class="info-value">${info.python_version || 'N/A'}</span>
                </div>
                <div class="info-row">
                    <span class="info-label">活跃任务:</span>
                    <span class="info-value">${info.active_jobs || 0}</span>
                </div>
            `;
        }
    } catch (error) {
        // Phase F-2 M1: handle AbortError (timeout) explicitly.
        console.error('Failed to load system info:', error);
    }
}

// ====================================================================
// Onboarding banner — shown on first visit, dismissible, restorable
// from the settings tab via "重新显示新手引导".
// ====================================================================
const ONBOARDING_DISMISSED_KEY = 'rlpe.onboardingDismissed';

function initOnboardingBanner() {
    const banner = document.getElementById('onboarding-banner');
    if (!banner) return;
    const dismissed = _safeStorageGet(ONBOARDING_DISMISSED_KEY) === '1';
    if (!dismissed) {
        banner.classList.remove('hidden');
    }
    document.getElementById('onboarding-close-btn')?.addEventListener('click', () => {
        banner.classList.add('hidden');
        _safeStorageSet(ONBOARDING_DISMISSED_KEY, '1');
    });
    document.getElementById('show-onboarding-btn')?.addEventListener('click', () => {
        _safeStorageRemove(ONBOARDING_DISMISSED_KEY);
        banner.classList.remove('hidden');
        // Switch to the upload tab so the banner is visible.
        document.querySelector('[data-tab="upload"]')?.click();
        showNotification('新手引导已重新显示', 'success');
    });
}

// ====================================================================
// Basic / Advanced view toggle on the config card. Persisted in
// localStorage so the user's preferred view is remembered.
// ====================================================================
const VIEW_PREF_KEY = 'rlpe.configView';
const LLM_BACKEND_KEY = 'rlpe.llmBackend';

function initViewToggle() {
    const buttons = document.querySelectorAll('.view-toggle-btn');
    if (!buttons.length) return;
    const savedView = _safeStorageGet(VIEW_PREF_KEY) || 'basic';
    applyConfigView(savedView);
    buttons.forEach(btn => {
        btn.addEventListener('click', () => {
            const view = btn.dataset.view;
            applyConfigView(view);
            _safeStorageSet(VIEW_PREF_KEY, view);
        });
    });
}

function applyConfigView(view) {
    document.querySelectorAll('.view-toggle-btn').forEach(b => {
        b.classList.toggle('active', b.dataset.view === view);
    });
    const basic = document.getElementById('config-basic-view');
    const advanced = document.getElementById('config-advanced-view');
    if (basic) basic.classList.toggle('hidden', view !== 'basic');
    if (advanced) advanced.classList.toggle('hidden', view !== 'advanced');
}

// ====================================================================
// Sync basic-view LLM backend dropdown <-> advanced-view dropdown.
// Both dropdowns control the SAME upload behaviour, so changing one
// must update the other. We use the basic dropdown's value as the
// primary source of truth (since it's the default visible one).
// ====================================================================
function initLLMBackendSync() {
    const basic = document.getElementById('llm-backend-basic');
    const advanced = document.getElementById('llm-backend');
    if (!basic || !advanced) return;
    // Round 16 audit: persist the user's LLM backend choice across
    // page reloads so the default doesn't silently revert to a
    // vendor-specific value every visit. Stored under
    // LLM_BACKEND_KEY so a privacy-conscious user can clear it via
    // DevTools / site data.
    //
    // audit 2026-08-17 (WEB-B6): the previous restore check
    // ``[basic.value, advanced.value].includes(saved)`` compared the
    // saved value against the <select>'s CURRENT value — but at
    // DOMContentLoaded both selects still default to "MiniMax", so
    // any non-MiniMax saved choice (e.g. "openai", "anthropic",
    // "MiniMax") silently failed the check and the user's preference
    // was discarded on every reload. Validate against the union of
    // all available <option> values across both selects instead.
    const saved = _safeStorageGet(LLM_BACKEND_KEY);
    if (saved) {
        const allValues = new Set([
            ...[...basic.options].map(o => o.value),
            ...[...advanced.options].map(o => o.value),
        ]);
        if (allValues.has(saved)) {
            basic.value = saved;
            advanced.value = saved;
        }
    }
    // basic → advanced
    basic.addEventListener('change', () => {
        advanced.value = basic.value;
        _safeStorageSet(LLM_BACKEND_KEY, basic.value);
        _syncLLMBackendVisibility();
    });
    // advanced → basic
    advanced.addEventListener('change', () => {
        basic.value = advanced.value;
        _safeStorageSet(LLM_BACKEND_KEY, advanced.value);
    });
    // initial: copy basic's value into advanced, then trigger the
    // visibility sync.
    advanced.value = basic.value;
    _syncLLMBackendVisibility();
}

// Override _buildLLMOptions so it reads from whichever dropdown is
// currently authoritative. We keep the original function (defined
// earlier in this file) reading `#llm-backend`; here we make sure
// `#llm-backend` is always in sync with `#llm-backend-basic` via
// the change handlers wired in initLLMBackendSync.

// ====================================================================
// API Key show/hide toggle
// ====================================================================
function initApiKeyToggle() {
    const btn = document.getElementById('api-key-toggle');
    const input = document.getElementById('MiniMax-api-key');
    if (!btn || !input) return;
    btn.addEventListener('click', () => {
        input.type = input.type === 'password' ? 'text' : 'password';
    });
}

// ====================================================================
// LLM status card — polls /system/llm-status to render
//   ✅ Key configured  /  ⚠️ Key missing  /  ❌ Test failed
// ====================================================================
async function refreshLLMStatus() {
    const iconEl = document.getElementById('llm-status-icon');
    const bodyEl = document.getElementById('llm-status-body');
    if (!bodyEl) return;
    try {
        const resp = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/system/llm-status`);
        if (!resp.ok) {
            if (iconEl) iconEl.textContent = '❌';
            bodyEl.innerHTML = `<span class="llm-status-error">无法获取 LLM 状态（HTTP ${resp.status}）</span>`;
            return;
        }
        const data = await resp.json();
        // Resolve endpoint / model with fallback to legacy field names.
        // The backend returns ``active_endpoint`` / ``active_model`` (the
        // resolved values) plus the deprecated ``default_endpoint`` /
        // ``default_model`` aliases. Prefer the new names if present.
        const endpoint = data.active_endpoint || data.default_endpoint || '—';
        const model = data.active_model || data.default_model || 'MiniMax-M3';
        const totalCost = Number(data.total_cost_cny) || 0;
        const totalCalls = Number(data.total_calls) || 0;
        const approxPerCall = Number(data.approx_cny_per_call) || 0;
        if (data.key_configured) {
            if (iconEl) iconEl.textContent = '✅';
            const keyHTML = data.key_preview
                ? `<span class="llm-status-key-preview">${escapeHtml(data.key_preview)}</span>`
                : '';
            // Build the cumulative usage line conditionally so a missing
            // total_cost_cny (server bug or older schema) doesn't crash
            // ``.toFixed`` on undefined.
            const usageLine = totalCalls > 0
                ? `· 累计 ${totalCalls} 次调用，¥${totalCost.toFixed(4)}`
                : '';
            bodyEl.innerHTML = `
                <span class="llm-status-ok">
                    API Key 已配置 ${keyHTML}
                </span>
                <span class="llm-status-detail">
                    来源：${escapeHtml(data.key_source || 'unknown')}
                    · 当前模型：${escapeHtml(model)}
                    · Endpoint：${escapeHtml(endpoint)}
                </span>
                <span class="llm-status-detail">
                    单次调用约 ¥${approxPerCall.toFixed(4)} ${usageLine}
                </span>
            `;
        } else {
            if (iconEl) iconEl.textContent = '⚠️';
            bodyEl.innerHTML = `
                <span class="llm-status-warn">
                    未检测到 MiniMax API Key
                </span>
                <span class="llm-status-detail">
                    项目根目录的 <code>.env</code> 文件中未发现 <code>ANTHROPIC_API_KEY</code>。
                    您可以：
                    （1）在「高级」视图的"API Key"输入框中临时填入；或
                    （2）在 .env 中写入并重启服务（推荐）。
                </span>
                <a class="llm-status-action-link"
                   href="https://platform.minimaxi.com/user-center/payment/token-plan"
                   target="_blank" rel="noopener">
                    🔗 申请 MiniMax Token Plan
                </a>
            `;
    } catch (err) {
        // Phase F-2 M1: handle AbortError (timeout) explicitly.
        if (iconEl) iconEl.textContent = '❌';
        const msg = err.name === 'AbortError' ? '检查超时' : (err.message || String(err));
        bodyEl.innerHTML = `<span class="llm-status-error">检查失败：${escapeHtml(msg)}</span>`;
    }
}

// ====================================================================
// Test the configured API key by hitting /system/test-llm.
// Runs in the foreground (button shows spinner). Cost ≈ ¥0.001 per call.
// ====================================================================
async function testLLMConnection() {
    const btn = document.getElementById('llm-test-btn');
    if (!btn) return;
    const originalHtml = btn.innerHTML;
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner-small"></span> 测试中…';
    try {
        // Pull the user-overridden values from the advanced-view inputs
        // (if visible), otherwise let the server fall back to .env.
        const body = {};
        const apiKeyVal = document.getElementById('MiniMax-api-key')?.value.trim();
        const endpointVal = document.getElementById('MiniMax-endpoint')?.value.trim();
        const modelVal = document.getElementById('MiniMax-model')?.value.trim();
        if (apiKeyVal) body.api_key = apiKeyVal;
        if (endpointVal) body.endpoint = endpointVal;
        if (modelVal) body.model = modelVal;
        const resp = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/system/test-llm`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (data.ok) {
            // Build the success message piece by piece so missing
            // optional fields (cost_cny, note) don't produce dangling
            // delimiters or unbalanced brackets.
            const parts = [`${data.latency_ms}ms`, data.model || 'MiniMax-M3'];
            if (data.cost_cny != null) {
                parts.push(`¥${data.cost_cny}`);
            }
            let msg = `✅ 连接成功（${parts.join(' · ')}）`;
            if (data.note) {
                msg += ` ${data.note}`;
            }
            showNotification(msg, 'success');
        } else {
            showNotification(
                `❌ 测试失败：${data.error || data.error_type || '未知错误'}`,
                'error',
            );
        }
    } catch (err) {
        showNotification(`❌ 测试请求失败：${err.message || err}`, 'error');
    } finally {
        btn.disabled = false;
        btn.innerHTML = originalHtml;
        // Refresh the status card so the user sees the latest state.
        refreshLLMStatus();
    }
}

// ====================================================================
// Cost estimate — when files are added/removed, estimate the MiniMax
// cost. Heuristic: ~3 figures per PDF page × ~10 panels per figure ×
// ¥0.0085 per call. We use file size as a proxy for page count
// (~80 KB / page is typical for OA radiolarian PDFs).
// ====================================================================
function initCostEstimate() {
    const updateEstimate = () => {
        const stripEl = document.getElementById('cost-estimate');
        const textEl = document.getElementById('cost-estimate-text');
        if (!stripEl || !textEl) return;
        if (!uploadedFiles.length) {
            stripEl.classList.add('hidden');
            return;
        }
        // Only show the estimate when the user has LLM enabled with MiniMax
        const useLLM = document.getElementById('use-gemma4')?.checked;
        const backend = document.getElementById('llm-backend-basic')?.value
            || document.getElementById('llm-backend')?.value;
        if (!useLLM || backend !== 'MiniMax') {
            stripEl.classList.add('hidden');
            return;
        }
        // Crude estimate: pages = sum(size_KB) / 80; calls ≈ pages * 3 * 8.
        // The 3× is "figures per page" and 8× is "panels per figure".
        // These are heuristics tuned on the gold corpus (~40 panels per
        // 10-page paper).
        const totalBytes = uploadedFiles.reduce((s, f) => s + (f.size || 0), 0);
        const estPages = Math.max(1, totalBytes / 1024 / 80);
        const estCalls = Math.round(estPages * 3 * 8);
        const estCost = (estCalls * 0.0085).toFixed(2);
        textEl.textContent = `预估调用 MiniMax ≈ ${estCalls} 次，约 ¥${estCost}（基于 ${uploadedFiles.length} 个文件，${Math.round(estPages)} 页）`;
        stripEl.classList.remove('hidden');
    };
    // Hook into add/remove events. addFiles / removeFile / clear-btn
    // already call renderFileList; we attach a MutationObserver to the
    // file-list element to catch all of them with a single hook.
    // Phase F-2 M7 fix: save the observer reference so it can be
    // disconnected on page unload to prevent memory leaks.
    const fileList = document.getElementById('file-list');
    if (fileList) {
        window._costEstimateObserver = new MutationObserver(updateEstimate);
        window._costEstimateObserver.observe(fileList, { childList: true, subtree: true });
    }
    // Also re-estimate when the LLM toggle / backend changes.
    document.getElementById('use-gemma4')?.addEventListener('change', updateEstimate);
    document.getElementById('llm-backend-basic')?.addEventListener('change', updateEstimate);
    document.getElementById('llm-backend')?.addEventListener('change', updateEstimate);
    // Initial render.
    updateEstimate();
}

// Phase F-2 M7: disconnect MutationObserver on page unload to prevent
// memory leaks in long-lived sessions.
window.addEventListener('beforeunload', () => {
    if (window._costEstimateObserver) {
        window._costEstimateObserver.disconnect();
        window._costEstimateObserver = null;
    }
});

// ====================================================================
// MiniMax usage panel (settings tab) — renders cumulative call counts
// and total cost from the same /system/llm-status endpoint.
// ====================================================================
async function refreshMiniMaxUsage() {
    const panel = document.getElementById('minimax-usage');
    if (!panel) return;
    try {
        const resp = await fetchWithTimeout(`${CONFIG.apiBaseUrl}/system/llm-status`);
        if (!resp.ok) {
            panel.innerHTML = `<p style="color: var(--text-muted);">无法读取用量（HTTP ${resp.status}）</p>`;
            return;
        }
        const data = await resp.json();
        const totalCost = Number(data.total_cost_cny) || 0;
        const totalCalls = Number(data.total_calls) || 0;
        const approxPerCall = Number(data.approx_cny_per_call) || 0;
        const model = data.active_model || data.default_model || 'MiniMax-M3';
        panel.innerHTML = `
            <div class="minimax-usage-item">
                <span class="minimax-usage-label">累计调用次数</span>
                <span class="minimax-usage-value">${totalCalls}</span>
            </div>
            <div class="minimax-usage-item">
                <span class="minimax-usage-label">累计费用（CNY）</span>
                <span class="minimax-usage-value">¥${totalCost.toFixed(4)}</span>
            </div>
            <div class="minimax-usage-item">
                <span class="minimax-usage-label">单次调用估价</span>
                <span class="minimax-usage-value">¥${approxPerCall.toFixed(4)}</span>
            </div>
            <div class="minimax-usage-item">
                <span class="minimax-usage-label">当前模型</span>
                <span class="minimax-usage-value" style="font-size: 0.95rem;">${escapeHtml(model)}</span>
            </div>
        `;
    } catch (err) {
        panel.innerHTML = `<p style="color: var(--danger-color);">读取失败：${escapeHtml(err.message || String(err))}</p>`;
    }
}

// Hook the existing tab-switch handler (defined earlier in this file
// at the top with ``document.querySelectorAll('.tab-btn').forEach``)
// to also refresh LLM status / usage when the user navigates to the
// settings tab. Implemented as a SINGLE event listener attached to
// each settings tab button (not to ``document``) so we don't fire on
// unrelated clicks elsewhere on the page. The previous global click
// listener fired on ANY click that bubbled to ``document`` and
// matched ``[data-tab="settings"]`` — including bubbled clicks from
// child elements that don't actually trigger a tab change.
document.querySelectorAll('.tab-btn[data-tab="settings"]').forEach(btn => {
    btn.addEventListener('click', () => {
        refreshLLMStatus();
        refreshMiniMaxUsage();
    });
});
