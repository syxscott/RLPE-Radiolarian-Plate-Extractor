// ==================== Configuration ==================== //
const CONFIG = {
    apiBaseUrl: localStorage.getItem('apiBaseUrl') || 'http://localhost:8000',
    refreshInterval: parseInt(localStorage.getItem('refreshInterval') || '3', 10),
};

let uploadedFiles = [];
let jobsData = {};
let resultsData = [];
let refreshIntervalId = null;

// ==================== Utilities ==================== //
function showNotification(message, type = 'success') {
    const notification = document.getElementById('notification');
    notification.textContent = message;
    notification.className = `notification ${type}`;
    setTimeout(() => {
        notification.classList.add('hidden');
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

function resolveAssetUrl(path) {
    if (!path) return '';
    if (/^https?:\/\//i.test(path)) return path;
    if (path.startsWith('/')) return `${CONFIG.apiBaseUrl}${path}`;
    return path;
}

async function checkApiHealth() {
    try {
        const response = await fetch(`${CONFIG.apiBaseUrl}/health`);
        const status = document.getElementById('api-status');
        if (response.ok) {
            status.textContent = '已连接';
            status.className = 'status-indicator status-connected';
            return true;
        } else {
            status.textContent = '服务异常';
            status.className = 'status-indicator status-error';
            return false;
        }
    } catch (error) {
        const status = document.getElementById('api-status');
        status.textContent = '无法连接';
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
                    <div class="file-item-name">${file.name}</div>
                    <div class="file-item-size">${formatFileSize(file.size)}</div>
                </div>
            </div>
            <button class="file-item-remove" onclick="removeFile(${index})">删除</button>
        </div>
    `).join('');
}

function removeFile(index) {
    uploadedFiles.splice(index, 1);
    renderFileList();
    document.getElementById('process-btn').disabled = uploadedFiles.length === 0;
}

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
        for (const file of uploadedFiles) {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${CONFIG.apiBaseUrl}/jobs/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`上传失败: ${response.statusText}`);
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
document.getElementById('use-gemma4').addEventListener('change', (e) => {
    const gemmaConfig = document.getElementById('gemma-config');
    if (e.target.checked) {
        gemmaConfig.classList.remove('hidden');
    } else {
        gemmaConfig.classList.add('hidden');
    }
});

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
    } catch (error) {
        console.error('Failed to load jobs:', error);
    }
}

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
    
    jobsList.innerHTML = jobs.map(job => `
        <div class="job-card">
            <div class="job-header">
                <div class="job-id">ID: ${job.job_id.substring(0, 12)}...</div>
                <span class="job-status status-${job.status}">${getStatusLabel(job.status)}</span>
            </div>
            <div class="job-details">
                <div class="job-detail-item">
                    <span class="job-detail-label">创建时间:</span>
                    <span class="job-detail-value">${formatDate(job.created_at)}</span>
                </div>
                <div class="job-detail-item">
                    <span class="job-detail-label">文件:</span>
                    <span class="job-detail-value">${job.filename || 'N/A'}</span>
                </div>
                <div class="job-detail-item">
                    <span class="job-detail-label">进度:</span>
                    <span class="job-detail-value">${job.progress || 0}%</span>
                </div>
                ${job.detail ? `
                <div class="job-detail-item">
                    <span class="job-detail-label">说明:</span>
                    <span class="job-detail-value">${job.detail}</span>
                </div>` : ''}
            </div>
            <div class="job-progress">
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${job.progress || 0}%"></div>
                </div>
            </div>
            <div class="job-actions">
                <button class="btn btn-small" onclick="viewJobDetails('${job.job_id}')">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                    详情
                </button>
                ${job.status === 'done' ? `<button class="btn btn-small" onclick="viewJobResults('${job.job_id}')">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/></svg>
                    结果
                </button>` : ''}
                <button class="btn btn-small btn-secondary" onclick="cancelJob('${job.job_id}', this)">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
                    取消
                </button>
            </div>
        </div>
    `).join('');
}

function getStatusLabel(status) {
    const labels = {
        'queued': '队列中',
        'running': '处理中',
        'done': '已完成',
        'failed': '失败'
    };
    return labels[status] || status;
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
                        <span class="value" style="font-family: monospace;">${jobId}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">状态</span>
                        <span class="value"><span class="job-status ${statusClass}">${getStatusLabel(job.status)}</span></span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">文件名</span>
                        <span class="value">${job.filename || 'N/A'}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">创建时间</span>
                        <span class="value">${formatDate(job.created_at)}</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">进度</span>
                        <span class="value">${job.progress || 0}%</span>
                    </div>
                    <div class="job-detail-item">
                        <span class="label">说明</span>
                        <span class="value">${job.detail || '无'}</span>
                    </div>
                </div>
            </div>
        `;

        if (job.status === 'failed' && job.error) {
            html += `
                <div class="job-detail-section">
                    <h3>错误信息</h3>
                    <div class="job-error-message">${job.error}</div>
                    ${job.error_trace ? `<div class="job-error-trace">${job.error_trace}</div>` : ''}
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
            showNotification('任务已取消');
            loadJobs();
        } else {
            throw new Error('取消失败');
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
    if (jobIds.length <= 1) return; // No need to filter if only one or zero jobs

    // Keep the first "All papers" option
    filter.innerHTML = '<option value="">全部论文</option>';
    jobIds.forEach(jobId => {
        const shortId = jobId.substring(0, 12) + '...';
        filter.innerHTML += `<option value="${jobId}">${shortId}</option>`;
    });
}

function renderResults() {
    const searchTerm = document.getElementById('result-search')?.value.toLowerCase() || '';
    const filterJob = document.getElementById('result-filter')?.value || '';
    
    const results = resultsData
        .filter(r => {
            const paperId = r.paper_id || '';
            const species = r.species || '';
            const matchesSearch = !searchTerm ||
                paperId.toLowerCase().includes(searchTerm) ||
                species.toLowerCase().includes(searchTerm);
            const matchesFilter = !filterJob || r.job_id === filterJob;
            return matchesSearch && matchesFilter;
        })
        .slice(0, 100);
    
    const tbody = document.getElementById('results-tbody');
    if (results.length === 0) {
        tbody.innerHTML = '<tr class="placeholder"><td colspan="7" style="text-align: center; color: #999;">暂无结果</td></tr>';
        return;
    }
    
    tbody.innerHTML = results.map(r => {
        const escapeAttr = (v) => String(v || '').replace(/'/g, "\\'").replace(/"/g, '\\"');
        const paperId = escapeAttr(r.paper_id);
        const figureId = escapeAttr(r.figure_id);
        const species = escapeAttr(r.species);
        const panelPath = escapeAttr(r.panel_path);
        const panelPathEscaped = resolveAssetUrl(r.panel_path || '').replace(/'/g, "\\'");
        return `
        <tr>
            <td>${escapeAttr(r.paper_id)}</td>
            <td>${escapeAttr(r.figure_id)}</td>
            <td>${escapeAttr(r.panel_id) || 'N/A'}</td>
            <td>${escapeAttr(r.species) || 'N/A'}</td>
            <td>
                <span class="confidence-badge ${getConfidenceClass(r.confidence)}">
                    ${(r.confidence * 100).toFixed(0)}%
                </span>
            </td>
            <td>
                ${r.panel_path ? `<img src="${resolveAssetUrl(r.panel_path)}" class="thumbnail-img" onclick="viewImage('${panelPathEscaped}', '${species || 'Unknown'}')">` : 'N/A'}
            </td>
            <td>
                <button class="btn btn-small" onclick="openCorrectionModal('${paperId}', '${figureId}', '${panelPath}')">纠正</button>
            </td>
        </tr>
    `}).join('');
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
        unique_species: new Set(resultsData.map(r => r.species).filter(s => s)).size
    };
    
    const statsHtml = `
        <div class="stat-card">
            <div class="stat-label">总匹配数</div>
            <div class="stat-value">${stats.total}</div>
        </div>
        <div class="stat-card secondary">
            <div class="stat-label">高置信度</div>
            <div class="stat-value">${stats.high_confidence}</div>
        </div>
        <div class="stat-card warning">
            <div class="stat-label">已识别物种</div>
            <div class="stat-value">${stats.unique_species}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">物种匹配数</div>
            <div class="stat-value">${stats.species_matched}</div>
        </div>
    `;
    
    const statsContainer = document.getElementById('results-stats');
    if (statsContainer) {
        statsContainer.innerHTML = statsHtml;
    }
}

document.getElementById('result-search')?.addEventListener('input', renderResults);
document.getElementById('result-filter')?.addEventListener('change', renderResults);

document.getElementById('export-btn')?.addEventListener('click', () => {
    const csv = [
        ['论文ID', '图版ID', 'Panel标签', '物种', '置信度'],
        ...resultsData.map(r => [
            r.paper_id,
            r.figure_id,
            r.panel_id || '',
            r.species || '',
            r.confidence
        ])
    ].map(row => row.map(cell => `"${cell}"`).join(','))
    .join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rlpe_results_${new Date().getTime()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showNotification('已导出结果');
});

// ==================== Image Modal ==================== //
function viewImage(src, title) {
    const modal = document.getElementById('image-modal');
    const img = document.getElementById('modal-image');
    const info = document.getElementById('modal-info');
    
    img.src = src;
    info.innerHTML = `<strong>物种:</strong> ${title}`;
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
    const modal = document.getElementById('correction-modal');
    document.getElementById('corrected-species').dataset.paperId = paperId;
    document.getElementById('corrected-species').dataset.figureId = figureId;
    document.getElementById('corrected-species').dataset.panelPath = panelPath;
    modal.classList.remove('hidden');
}

function closeCorrectionModal() {
    document.getElementById('correction-modal').classList.add('hidden');
}

document.querySelector('#correction-modal .modal-close')?.addEventListener('click', closeCorrectionModal);

document.getElementById('correction-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const speciesInput = document.getElementById('corrected-species');
    const payload = {
        paper_id: speciesInput.dataset.paperId,
        figure_id: speciesInput.dataset.figureId,
        panel_path: speciesInput.dataset.panelPath,
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
            throw new Error('提交失败');
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
