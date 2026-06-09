// ==================== Configuration ==================== //
const CONFIG = {
    apiBaseUrl: localStorage.getItem('apiBaseUrl') || 'http://localhost:8000',
    refreshInterval: parseInt(localStorage.getItem('refreshInterval') || '3', 10),
};

let uploadedFiles = [];
let jobsData = {};
let resultsData = [];
let refreshIntervalId = null;
let _notificationTimer = null;
// Per-page stash of full record objects. Indexed by `data-record-index`
// on each <img> / <button> in the rendered results table. Reset at the
// top of every renderResults() call to avoid stale references across
// re-renders. Replaces the previous pattern of embedding the entire
// JSON in a data-record attribute, which (a) duplicated data 25+ times
// per page, (b) was a XSS vector if records contained `&` or `<`.
let __rlpeRecords = [];

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

function resolveAssetUrl(path) {
    if (!path) return '';
    if (/^https?:\/\//i.test(path)) return path;
    if (path.startsWith('/')) return `${CONFIG.apiBaseUrl}${path}`;
    return path;
}

async function checkApiHealth() {
    const status = document.getElementById('api-status');
    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/health`);
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
        document.getElementById(`${tabName}-tab`).classList.add('active');

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

uploadArea.addEventListener('click', () => pdfInput.click());

// Keyboard accessibility: the upload area is a div (not a button), so
// without this handler, Tab skips it and pressing Enter / Space does
// nothing. role="button" + tabindex="0" is set in HTML; this wires up
// the keypress.
uploadArea.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        pdfInput.click();
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
    const pdfFiles = files.filter(f => f.type === 'application/pdf' || f.name.endsWith('.pdf'));
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
                    <div class="file-item-name">${esc(file.name)}</div>
                    <div class="file-item-size">${esc(formatFileSize(file.size))}</div>
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
        // Merge LLM + PBDB + extractor options into a single JSON body
        let combinedOptions = null;
        const baseOpts = { use_opendataloader: useOpenDataLoader };
        if (llmOptions || paleodbOptions) {
            combinedOptions = { ...baseOpts, ...(llmOptions || {}), ...(paleodbOptions || {}) };
        } else if (useOpenDataLoader) {
            combinedOptions = baseOpts;
        }

        for (const file of uploadedFiles) {
            const formData = new FormData();
            formData.append('file', file);
            if (combinedOptions) {
                formData.append('options', JSON.stringify(combinedOptions));
            }

            const response = await fetch(`${CONFIG.apiBaseUrl}/jobs/upload`, {
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

        // Switch to jobs tab
        document.querySelector('[data-tab="jobs"]').click();

        // Start polling
        startJobPolling();
    } catch (error) {
        showNotification(error.message, 'error');
    } finally {
        _processing = false;
        btn.disabled = false;
        btn.classList.remove('btn-loading');
        btn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> 开始处理';
    }
});

// ==================== Config Toggles ==================== //
function _syncLLMBackendVisibility() {
    const backend = document.getElementById('llm-backend').value;
    const localConfig = document.getElementById('llm-local-config');
    const MiniMaxConfig = document.getElementById('MiniMax-config');
    if (backend === 'MiniMax') {
        localConfig.classList.add('hidden');
        MiniMaxConfig.classList.remove('hidden');
    } else {
        localConfig.classList.remove('hidden');
        MiniMaxConfig.classList.add('hidden');
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
    const enabled = document.getElementById('use-paleodb').checked;
    if (!enabled) return null;
    const opts = { use_paleodb: true };
    const maxRaw = document.getElementById('paleodb-max-occurrences').value.trim();
    const maxOcc = parseInt(maxRaw, 10);
    if (maxRaw === '' || isNaN(maxOcc) || maxOcc < 1) {
        throw new Error(`PBDB 最大出现记录数必须是 ≥1 的整数，当前值: "${maxRaw}"`);
    }
    if (maxOcc > 500) {
        throw new Error(`PBDB 最大出现记录数不能超过 500，当前值: ${maxOcc}`);
    }
    opts.paleodb_max_occurrences = maxOcc;
    const endpoint = document.getElementById('paleodb-endpoint').value.trim();
    if (endpoint) opts.paleodb_endpoint = endpoint;
    opts.paleodb_offline = document.getElementById('paleodb-offline').checked;
    return opts;
}

// ==================== Build LLM options from form ==================== //
function _buildLLMOptions() {
    const useGemma = document.getElementById('use-gemma4').checked;
    if (!useGemma) return null;

    const backend = document.getElementById('llm-backend').value;

    // Validate conf threshold up-front so the user gets immediate feedback
    // instead of a server round-trip.
    const confRaw = document.getElementById('gemma-conf-threshold').value.trim();
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
        const apiKey = document.getElementById('MiniMax-api-key').value.trim();
        if (apiKey) options.MiniMax_api_key = apiKey;
        const endpoint = document.getElementById('MiniMax-endpoint').value.trim();
        if (endpoint) options.MiniMax_endpoint = endpoint;
        const model = document.getElementById('MiniMax-model').value.trim();
        if (model) options.MiniMax_model = model;
        options.MiniMax_enable_thinking = document.getElementById('MiniMax-enable-thinking').checked;

        // Validate thinking budget up-front
        const thinkingRaw = document.getElementById('MiniMax-thinking-budget').value.trim();
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

        options.MiniMax_fallback_default = document.getElementById('MiniMax-fallback-default').value;
        // Web mode always uses non-interactive popup (block on event.wait)
        options.MiniMax_interactive = false;
    } else {
        // Local backend path
        const host = document.getElementById('llm-host').value.trim();
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
async function loadJobs() {
    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/jobs`);
        if (!response.ok) return;

        const jobs = await response.json();
        jobsData = jobs.reduce((acc, job) => {
            acc[job.job_id] = job;
            return acc;
        }, jobsData);

        renderJobsList();
        maybeStopPolling();
    } catch (error) {
        console.error('Failed to load jobs:', error);
        showNotification('加载任务列表失败: ' + (error.message || error), 'error');
    }
}

// Polling should be adaptive: stop hammering /jobs once all jobs have
// settled. Without this guard, a 1-hour-old task keeps the browser
// pulling the endpoint every 3 s forever (wasted bandwidth, extra
// server load, and noisy console errors if the server is down).
function maybeStopPolling() {
    if (!refreshIntervalId) return;
    const hasActive = Object.values(jobsData).some(
        j => j.status === 'queued' || j.status === 'running' || j.status === 'awaiting_user_decision'
    );
    if (!hasActive) {
        clearInterval(refreshIntervalId);
        refreshIntervalId = null;
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
        jobsList.innerHTML = '<div style="text-align: center; color: #999; padding: 2rem;">暂无任务</div>';
        return;
    }

    // Use esc() on every backend string; use data-action + data-job-id for
    // event delegation (replaces previous inline onclick="..." which
    // concatenated job_id and would have XSS'd if a job_id ever contained
    // a quote character). The click handler lives at module load (below).
    jobsList.innerHTML = jobs.map(job => `
        <div class="job-card" data-job-id="${esc(job.job_id)}">
            <input type="checkbox" class="job-card-checkbox" data-job-id="${esc(job.job_id)}"
                   ${selectedJobIds.has(job.job_id) ? 'checked' : ''}>
            <div class="job-header">
                <div class="job-id">ID: ${esc(job.job_id.substring(0, 12))}...</div>
                <span class="job-status status-${esc(job.status)}">${esc(getStatusLabel(job.status))}</span>
            </div>
            <div class="job-details">
                <div class="job-detail-item">
                    <span class="job-detail-label">创建时间:</span>
                    <span class="job-detail-value">${esc(formatDate(job.created_at))}</span>
                </div>
                <div class="job-detail-item">
                    <span class="job-detail-label">文件:</span>
                    <span class="job-detail-value">${esc(job.filename || 'N/A')}</span>
                </div>
                <div class="job-detail-item">
                    <span class="job-detail-label">进度:</span>
                    <span class="job-detail-value">${esc(job.progress || 0)}%</span>
                </div>
                ${job.stage ? `
                <div class="job-detail-item">
                    <span class="job-detail-label">阶段:</span>
                    <span class="job-detail-value">${esc(job.stage)}</span>
                </div>` : ''}
                ${job.elapsed_sec != null ? `
                <div class="job-detail-item">
                    <span class="job-detail-label">已用时:</span>
                    <span class="job-detail-value">${esc(formatElapsed(job.elapsed_sec))}</span>
                </div>` : ''}
                ${job.detail ? `
                <div class="job-detail-item">
                    <span class="job-detail-label">说明:</span>
                    <span class="job-detail-value">${esc(job.detail)}</span>
                </div>` : ''}
            </div>
            <div class="job-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${esc(job.progress || 0)}%"></div>
                </div>
            </div>
            <div class="job-actions">
                <button type="button" class="btn btn-small" data-action="details" data-job-id="${esc(job.job_id)}">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                    详情
                </button>
                ${job.status === 'done' ? `<button type="button" class="btn btn-small" data-action="results" data-job-id="${esc(job.job_id)}">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
                    结果
                </button>` : ''}
                <button type="button" class="btn btn-small btn-secondary" data-action="cancel" data-job-id="${esc(job.job_id)}">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    取消
                </button>
                ${(job.status === 'done' || job.status === 'failed' || job.status === 'cancelled') ? `
                <button type="button" class="btn btn-small btn-danger" data-action="delete" data-job-id="${esc(job.job_id)}" title="删除任务及文件">
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
        'done': '已完成',
        'failed': '失败',
        'cancelled': '已取消'
    };
    return labels[status] || status;
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
    const bytes = await estimateSelectedBytes(idsToShow);
    const sizeStr = bytes > 0 ? `，预计释放 ${formatBytes(bytes)}` : '';
    summary.innerHTML = `将删除 <strong>${n}</strong> 个任务${sizeStr}。此操作不可撤销。${truncWarn}`;

    // Build per-job list with filename and a remove button
    list.innerHTML = idsToShow.map(id => {
        const job = jobsData[id];
        const filename = job?.filename || '(无文件)';
        return `<div class="job-row" data-row-id="${esc(id)}">
            <span class="job-row-id">${esc(id.substring(0, 12))}...</span>
            <span class="job-row-name" title="${esc(filename)}">${esc(filename)}</span>
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

async function estimateSelectedBytes(jobIds) {
    // Best-effort size estimate by summing job directory sizes when available.
    // Falls back to 0 if directories cannot be stat'd (e.g. cli_ jobs).
    let total = 0;
    for (const id of jobIds) {
        try {
            // We don't have a "size" endpoint, so skip; the server returns
            // bytes_freed in the response. UI will show post-hoc if needed.
        } catch (_) { /* noop */ }
    }
    return total;
}

function formatBytes(n) {
    if (n < 1024) return `${n} B`;
    if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
    return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

async function confirmDelete() {
    const confirmBtn = document.getElementById('delete-modal-confirm');
    const jobIds = JSON.parse(confirmBtn.dataset.jobIds || '[]');
    const deleteFiles = document.getElementById('delete-modal-files').checked;
    if (jobIds.length === 0) return;

    confirmBtn.disabled = true;
    confirmBtn.textContent = '删除中...';

    try {
        let resp, data;
        if (jobIds.length === 1) {
            const url = `${CONFIG.apiBaseUrl}/jobs/${encodeURIComponent(jobIds[0])}?delete_files=${deleteFiles}`;
            resp = await fetch(url, { method: 'DELETE' });
            data = await resp.json();
        } else {
            resp = await fetch(`${CONFIG.apiBaseUrl}/jobs/batch-delete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ job_ids: jobIds, delete_files: deleteFiles }),
            });
            data = await resp.json();
        }

        if (!resp.ok) {
            alert(`删除失败: ${data.detail || resp.statusText}`);
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
            } else {
                showToast(`已删除任务${freed}`, 'success');
            }
        } else {
            const deleted = data.deleted || 0;
            const notFound = results.filter(r => r.status === 'not_found').length;
            const fileErr = results.filter(r => r.status === 'file_error').length;
            let suffix = '';
            if (notFound) suffix += `，${notFound} 个已不存在`;
            if (fileErr) suffix += `，${fileErr} 个文件清理失败`;
            showToast(`已删除 ${deleted} 个任务${suffix}${freed}`, deleted > 0 ? 'success' : 'info');
        }
    } catch (err) {
        console.error('Delete failed', err);
        alert(`删除失败: ${err}`);
        confirmBtn.disabled = false;
        confirmBtn.textContent = '确认删除';
    }
}

function showToast(message, type = 'info') {
    // Minimal toast — uses a div if available, otherwise alert fallback.
    const existing = document.getElementById('rlpe-toast');
    if (existing) existing.remove();
    const el = document.createElement('div');
    el.id = 'rlpe-toast';
    el.textContent = message;
    el.style.cssText = `
        position: fixed; bottom: 24px; right: 24px; z-index: 2000;
        padding: 12px 20px; border-radius: 8px; color: white; font-size: 14px;
        background: ${type === 'success' ? '#28a745' : type === 'error' ? '#dc3545' : '#333'};
        box-shadow: 0 4px 12px rgba(0,0,0,0.2); animation: slideUp 0.3s ease;
    `;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 3500);
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

// Single canonical escape function for both text content and attribute values.
// Previously the codebase had two near-duplicate helpers (escapeHtml + escapeAttr)
// with subtle differences; the local `escapeAttr` skipped `&`, which broke
// attribute parsing when records contained `&` (e.g. species "A & B").
function esc(v) {
    return escapeHtml(v);
}

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
        const response = await fetch(`${CONFIG.apiBaseUrl}/jobs/${jobId}/result`);

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
                        <span class="value" style="font-family: monospace;">${esc(jobId)}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">状态</span>
                        <span class="value"><span class="job-status ${esc(statusClass)}">${esc(getStatusLabel(job.status))}</span></span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">文件名</span>
                        <span class="value">${esc(job.filename || 'N/A')}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">创建时间</span>
                        <span class="value">${esc(formatDate(job.created_at))}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">进度</span>
                        <span class="value">${esc(job.progress || 0)}%</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">阶段</span>
                        <span class="value">${esc(job.stage || '—')}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">已用时</span>
                        <span class="value">${esc(job.elapsed_sec != null ? formatElapsed(job.elapsed_sec) : '—')}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">说明</span>
                        <span class="value">${esc(job.detail || '无')}</span>
                    </div>
                </div>
            </div>
        `;

        if (job.status === 'failed' && job.error) {
            // Both error and error_trace come from server-side tracebacks and
            // may contain `<`, `>`, `&` (e.g. "AttributeError: '<' not supported").
            // MUST go through esc() — they are inserted into innerHTML.
            html += `
                <div class="job-detail-section">
                    <h3>错误信息</h3>
                    <div class="job-error-message">${esc(job.error)}</div>
                    ${job.error_trace ? `<div class="job-error-trace">${esc(job.error_trace)}</div>` : ''}
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
        content.innerHTML = `
            <div class="job-detail-section">
                <p style="text-align: center; color: var(--danger-color); padding: 2rem;">
                    获取详情失败: ${error.message}
                </p>
            </div>
        `;
    }
}

async function viewJobResults(jobId) {
    // Switch to results tab and filter by job
    document.querySelector('[data-tab="results"]').click();
    const filter = document.getElementById('result-filter');
    if (filter) {
        filter.value = jobId;
        loadResults();
    }
}

async function cancelJob(jobId, cancelBtn) {
    if (!confirm('确认取消此任务？')) return;

    const originalText = cancelBtn.innerHTML;
    cancelBtn.disabled = true;
    cancelBtn.innerHTML = '<span class="spinner-small"></span> 取消中...';

    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/jobs/${jobId}/cancel`, {
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
        showNotification(error.message, 'error');
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

// ==================== Results ==================== //
async function loadResults() {
    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/results`);
        if (!response.ok) return;

        resultsData = await response.json();
        populateResultFilter();
        renderResults();
        updateStats();
    } catch (error) {
        console.error('Failed to load results:', error);
    }
}

function populateResultFilter() {
    const filter = document.getElementById('result-filter');
    if (!filter) return;

    // Get unique job IDs from results
    const jobIds = [...new Set(resultsData.map(r => r.job_id).filter(Boolean))];
    if (jobIds.length <= 1) {
        // 0 or 1 job — nothing to filter. If the user previously had a
        // selection, clear it so the table shows everything (instead of
        // silently filtering to 0 rows when the job no longer exists).
        if (filter.value && filter.value !== '') {
            filter.value = '';
        }
        return;
    }

    // Remember the user's previous selection so we can detect "stale" IDs
    // (job was deleted on the server but the <option> is gone).
    const prevValue = filter.value;

    // Keep the first "All papers" option
    filter.innerHTML = '<option value="">全部论文</option>';
    jobIds.forEach(jobId => {
        const shortId = jobId.substring(0, 12) + '...';
        filter.innerHTML += `<option value="${esc(jobId)}">${esc(shortId)}</option>`;
    });

    // If the previous selection is no longer in the option list, reset.
    // (The <select> retains its value even when the matching <option> is
    // removed, so the filter would silently match nothing.)
    if (prevValue && prevValue !== '' && !jobIds.includes(prevValue)) {
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
    if (md.v18_panel_id_source === 'image_ocr') return 'image_ocr';
    if (r.panel_path) return 'positional';
    return 'no_image';
}

function renderResults() {
    // Reset the record stash so indices in the new render refer to fresh data.
    __rlpeRecords = [];
    const searchTerm = document.getElementById('result-search')?.value.toLowerCase() || '';
    const filterJob = document.getElementById('result-filter')?.value || '';

    const all = resultsData;
    const filtered = all.filter(r => {
        const paperId = (r.paper_id || '').toLowerCase();
        const species = (r.species || '').toLowerCase();
        const panelId = String(r.panel_id || '').toLowerCase();
        const matchesSearch = !searchTerm ||
            paperId.includes(searchTerm) ||
            species.includes(searchTerm) ||
            panelId.includes(searchTerm);
        const matchesFilter = !filterJob || r.job_id === filterJob;
        const matchesStatus = resultsTableState.statusFilter === 'all'
            || getRecordStatus(r) === resultsTableState.statusFilter;
        return matchesSearch && matchesFilter && matchesStatus;
    });

    // Sort
    const { sortKey, sortDir } = resultsTableState;
    const dir = sortDir === 'asc' ? 1 : -1;
    filtered.sort((a, b) => {
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
        tbody.innerHTML = '<tr class="placeholder"><td colspan="8" style="text-align: center; color: #999;">暂无结果</td></tr>';
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
        const paperId = esc(r.paper_id);
        const figureId = esc(r.figure_id);
        const species = esc(r.species);
        const panelPath = esc(r.panel_path);
        const panelPathEscaped = esc(resolveAssetUrl(r.panel_path || ''));
        const md = r.metadata || {};
        const ocrSource = md.v18_panel_id_source;
        const oldPanelId = md.v18_old_panel_id;
        const status = getRecordStatus(r);
        const ocrCell = status === 'image_ocr'
            ? `<span class="badge badge-ok" title="Re-OCR'd from panel image${oldPanelId && oldPanelId !== r.panel_id ? ' (was: ' + esc(oldPanelId) + ')' : ''}">✓ ${esc(r.panel_id) || 'N/A'}</span>`
            : (status === 'positional'
                ? `<span class="badge badge-warn" title="panel_id came from caption list (positional); image OCR did not return a usable label">⚠ ${esc(r.panel_id) || 'N/A'}</span>`
                : `<span class="badge badge-muted" title="No panel image available for OCR verification">— ${esc(r.panel_id) || 'N/A'}</span>`);
        return `
        <tr>
            <td>${esc(r.paper_id)}</td>
            <td>${esc(r.figure_id)}</td>
            <td>${esc(r.panel_id) || 'N/A'}</td>
            <td>${ocrCell}</td>
            <td>${esc(r.species) || 'N/A'}</td>
            <td>
                <span class="confidence-badge ${getConfidenceClass(r.confidence)}">
                    ${(r.confidence * 100).toFixed(0)}%
                </span>
            </td>
            <td>
                ${r.panel_path ? `<img src="${panelPathEscaped}" class="thumbnail-img" data-record-index="${recordIndex}" data-species="${species}" alt="panel thumbnail">` : 'N/A'}
            </td>
            <td>
                <button class="btn btn-small" data-correct-index="${recordIndex}">纠正</button>
            </td>
        </tr>`;
    }).join('');

    // Wire up click handlers (avoids quote-escaping bugs from inline onclick)
    tbody.querySelectorAll('.thumbnail-img').forEach(img => {
        img.addEventListener('click', () => {
            const idx = parseInt(img.getAttribute('data-record-index'), 10);
            const record = __rlpeRecords[idx];
            if (!record) return;
            openImageModal(img.getAttribute('src'), img.getAttribute('data-species'), record);
        });
        img.addEventListener('error', () => {
            // Replace broken image with a plain "N/A" text. { once: true } ensures
            // the fallback only fires once (avoids loops if the placeholder 404s).
            const span = document.createElement('span');
            span.className = 'text-muted';
            span.textContent = 'N/A';
            img.replaceWith(span);
        }, { once: true });
    });
    tbody.querySelectorAll('[data-correct-index]').forEach(btn => {
        btn.addEventListener('click', () => {
            const idx = parseInt(btn.getAttribute('data-correct-index'), 10);
            const r = __rlpeRecords[idx];
            if (!r) return;
            openCorrectionModal(r.paper_id, r.figure_id, r.panel_path);
        });
    });

    renderResultsPagination(total, resultsTableState.page, totalPages);
    const counts = {
        all: all.length,
        image_ocr: all.filter(r => getRecordStatus(r) === 'image_ocr').length,
        positional: all.filter(r => getRecordStatus(r) === 'positional').length,
        no_image: all.filter(r => getRecordStatus(r) === 'no_image').length,
    };
    renderResultsStatusFilterCounts(counts);
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
        `<button type="button" role="tab" class="status-filter-btn ${k === cur ? 'active' : ''}" data-status="${esc(k)}" aria-selected="${k === cur ? 'true' : 'false'}">${label} <span class="status-filter-count">${n}</span></button>`
    ).join('');
    container.querySelectorAll('.status-filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            resultsTableState.statusFilter = btn.getAttribute('data-status');
            resultsTableState.page = 1;
            renderResults();
        });
    });
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
    document.getElementById('page-first')?.addEventListener('click', () => { resultsTableState.page = 1; renderResults(); });
    document.getElementById('page-prev')?.addEventListener('click', () => { resultsTableState.page = Math.max(1, page - 1); renderResults(); });
    document.getElementById('page-next')?.addEventListener('click', () => { resultsTableState.page = Math.min(totalPages, page + 1); renderResults(); });
    document.getElementById('page-last')?.addEventListener('click', () => { resultsTableState.page = totalPages; renderResults(); });
    const ps = document.getElementById('page-size-select');
    if (ps) {
        ps.value = String(resultsTableState.pageSize);
        ps.addEventListener('change', (e) => {
            resultsTableState.pageSize = parseInt(e.target.value, 10) || 25;
            resultsTableState.page = 1;
            renderResults();
        });
    }
}

function getConfidenceClass(confidence) {
    if (confidence >= 0.8) return 'confidence-high';
    if (confidence >= 0.5) return 'confidence-medium';
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
function getFilteredResults() {
    const searchTerm = document.getElementById('result-search')?.value.toLowerCase() || '';
    const filterJob = document.getElementById('result-filter')?.value || '';
    return resultsData.filter(r => {
        const paperId = (r.paper_id || '').toLowerCase();
        const species = (r.species || '').toLowerCase();
        const panelId = String(r.panel_id || '').toLowerCase();
        const matchesSearch = !searchTerm ||
            paperId.includes(searchTerm) ||
            species.includes(searchTerm) ||
            panelId.includes(searchTerm);
        const matchesFilter = !filterJob || r.job_id === filterJob;
        const matchesStatus = resultsTableState.statusFilter === 'all'
            || getRecordStatus(r) === resultsTableState.statusFilter;
        return matchesSearch && matchesFilter && matchesStatus;
    });
}

// CSV cell formatter: handles null/undefined, escapes embedded quotes by
// doubling them (RFC 4180), wraps in double quotes.
function csvCell(v) {
    if (v == null) return '""';
    const s = String(v);
    return `"${s.replace(/"/g, '""')}"`;
}

document.getElementById('export-btn')?.addEventListener('click', () => {
    // Export the currently-filtered subset (not the full resultsData) so
    // the user gets what they see. Prepend a UTF-8 BOM so Excel auto-detects
    // the encoding and Chinese species names render correctly.
    const rows = getFilteredResults();
    const header = ['论文ID', '图版ID', 'Panel标签', '物种', '置信度', 'OCR来源'];
    const dataRows = rows.map(r => [
        r.paper_id ?? '',
        r.figure_id ?? '',
        r.panel_id ?? '',
        r.species ?? '',
        r.confidence != null ? r.confidence : '',
        (r.metadata && r.metadata.v18_panel_id_source) || '',
    ]);
    const csv = [header, ...dataRows]
        .map(row => row.map(csvCell).join(','))
        .join('\n');
    const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const filterSuffix = resultsTableState.statusFilter !== 'all' ? `_${resultsTableState.statusFilter}` : '';
    a.download = `rlpe_results${filterSuffix}_${stamp}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showNotification(`已导出 ${rows.length} 条结果`);
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
        const ocrSource = md.v18_panel_id_source;
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
        info.innerHTML = `
            <div class="modal-grid">
                <div class="modal-row"><span class="modal-label">论文 ID:</span> <code>${escapeHtml(record.paper_id || 'N/A')}</code></div>
                <div class="modal-row"><span class="modal-label">图版 ID:</span> <code>${escapeHtml(record.figure_id || 'N/A')}</code></div>
                <div class="modal-row"><span class="modal-label">Panel 标签:</span> <code>${escapeHtml(record.panel_id || 'N/A')}</code></div>
                <div class="modal-row"><span class="modal-label">Panel 来源:</span> ${ocrBadge}</div>
                <div class="modal-row"><span class="modal-label">物种:</span> <strong>${escapeHtml(record.species || 'N/A')}</strong></div>
                <div class="modal-row"><span class="modal-label">置信度:</span> ${(record.confidence * 100).toFixed(0)}%</div>
                ${reassignNote}
                ${record.bbox && record.bbox.some(v => v > 0) ? `<div class="modal-row"><span class="modal-label">BBox:</span> <code>[${record.bbox.join(', ')}]</code></div>` : ''}
                ${captionSnippet ? `<div class="modal-row modal-row-wide"><span class="modal-label">Caption:</span><div class="modal-caption">${escapeHtml(captionSnippet)}${captionSnippet.length >= 280 ? '…' : ''}</div></div>` : ''}
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

// ==================== Correction Modal ==================== //
function openCorrectionModal(paperId, figureId, panelPath) {
    // Defensive: if the record was missing paperId / figureId (e.g. an
    // extremely old result row), the dataset would receive the literal
    // string "undefined", which the backend would happily accept and
    // persist as a dirty review row. Bail early instead.
    if (!paperId || !figureId || paperId === 'undefined' || figureId === 'undefined') {
        showNotification('该记录缺少 paper_id / figure_id，无法提交纠正', 'error');
        return;
    }
    const modal = document.getElementById('correction-modal');
    document.getElementById('corrected-species').dataset.paperId = paperId;
    document.getElementById('corrected-species').dataset.figureId = figureId;
    document.getElementById('corrected-species').dataset.panelPath = panelPath || '';
    modal.classList.remove('hidden');
}

function closeCorrectionModal() {
    document.getElementById('correction-modal').classList.add('hidden');
}

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

    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/review/correction`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            showNotification('纠正已提交');
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
        showNotification(error.message, 'error');
    }
});

// ==================== Settings ==================== //
document.getElementById('save-settings-btn')?.addEventListener('click', () => {
    const apiUrl = document.getElementById('api-base-url').value;
    const refreshInterval = document.getElementById('refresh-interval').value;
    
    localStorage.setItem('apiBaseUrl', apiUrl);
    localStorage.setItem('refreshInterval', refreshInterval);
    
    CONFIG.apiBaseUrl = apiUrl;
    CONFIG.refreshInterval = parseInt(refreshInterval, 10);
    
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
});

async function loadSystemInfo() {
    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/system/info`);
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
        console.error('Failed to load system info:', error);
    }
}
