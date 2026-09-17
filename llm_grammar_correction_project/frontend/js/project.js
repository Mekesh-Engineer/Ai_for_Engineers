/**
 * Local LLM Studio - Repository Documentation & Docstring Grammar Auditor
 */

const ProjectModule = {
  treeData: null,
  filesList: [],
  selectedFiles: ['README.md', 'main.py', 'requirements.txt'],
  lastReport: null,

  init() {
    this.bindEvents();
  },

  onShow() {
    this.loadProjectTree();
    this.loadProjectFiles();
  },

  bindEvents() {
    const analyzeBtn = document.getElementById('run-project-audit-btn');
    const debugForm = document.getElementById('project-debug-form');
    const exportBtn = document.getElementById('export-audit-report-btn');
    const sendToChatBtn = document.getElementById('send-audit-to-chat-btn');

    if (analyzeBtn) {
      analyzeBtn.addEventListener('click', () => this.runProjectAnalysis());
    }

    if (debugForm) {
      debugForm.addEventListener('submit', (e) => {
        e.preventDefault();
        this.runDebugger();
      });
    }

    if (exportBtn) {
      exportBtn.addEventListener('click', () => this.exportReport());
    }

    if (sendToChatBtn) {
      sendToChatBtn.addEventListener('click', () => this.sendReportToChat());
    }
  },

  async loadProjectTree() {
    const container = document.getElementById('project-tree-container');
    if (!container) return;

    try {
      const res = await fetch('/api/project/tree');
      if (res.ok) {
        this.treeData = await res.json();
        container.innerHTML = this.renderTreeNode(this.treeData);
      }
    } catch (err) {
      container.innerHTML = `<div class="text-xs text-rose-400 p-2">Failed to load tree: ${err.message}</div>`;
    }
  },

  renderTreeNode(node, depth = 0) {
    if (!node) return '';
    const isDir = node.is_dir;
    const paddingLeft = depth * 14;

    if (isDir) {
      const childrenHtml = (node.children || []).map(c => this.renderTreeNode(c, depth + 1)).join('');
      return `
        <div class="tree-node">
          <div style="padding-left: ${paddingLeft}px" class="flex items-center py-1 px-1.5 rounded hover:bg-slate-800/60 cursor-pointer text-xs text-slate-300 font-medium">
            <span class="mr-1.5 text-amber-400">📁</span>
            <span class="truncate">${node.name}</span>
          </div>
          <div class="tree-children">${childrenHtml}</div>
        </div>
      `;
    } else {
      const isSelected = this.selectedFiles.includes(node.relative_path);
      return `
        <div style="padding-left: ${paddingLeft}px" 
             onclick="ProjectModule.toggleFileSelection('${node.relative_path}')"
             class="flex items-center justify-between py-1 px-1.5 rounded hover:bg-slate-800/60 cursor-pointer text-xs ${isSelected ? 'text-indigo-300 bg-indigo-950/30' : 'text-slate-400'}">
          <div class="flex items-center truncate mr-2">
            <span class="mr-1.5 text-indigo-400">📄</span>
            <span class="truncate font-mono text-[11px]">${node.name}</span>
          </div>
          <input type="checkbox" ${isSelected ? 'checked' : ''} class="rounded border-slate-700 bg-slate-900 text-indigo-500 pointer-events-none">
        </div>
      `;
    }
  },

  toggleFileSelection(relPath) {
    if (this.selectedFiles.includes(relPath)) {
      this.selectedFiles = this.selectedFiles.filter(f => f !== relPath);
    } else {
      this.selectedFiles.push(relPath);
    }
    this.loadProjectTree();
    this.updateSelectedFilesBadge();
  },

  updateSelectedFilesBadge() {
    const badge = document.getElementById('project-selected-files-count');
    if (badge) {
      badge.innerText = `${this.selectedFiles.length} files selected for audit`;
    }
  },

  async loadProjectFiles() {
    const container = document.getElementById('project-files-grid');
    if (!container) return;

    try {
      const res = await fetch('/api/project/files');
      if (res.ok) {
        this.filesList = await res.json();
        container.innerHTML = this.filesList.slice(0, 16).map(f => `
          <div onclick="ProjectModule.toggleFileSelection('${f.relative_path}')" 
               class="p-2.5 rounded-xl border transition cursor-pointer flex items-center justify-between text-xs ${this.selectedFiles.includes(f.relative_path) ? 'bg-indigo-950/40 border-indigo-500/50 text-indigo-200' : 'bg-slate-950/60 border-slate-800/80 hover:bg-slate-800/60 text-slate-300'}">
            <div class="truncate mr-2">
              <div class="font-medium truncate font-mono text-[11px]">${f.filename}</div>
              <div class="text-[10px] text-slate-500 truncate">${f.relative_path}</div>
            </div>
            <span class="text-[10px] px-1.5 py-0.5 rounded bg-slate-800 text-slate-400 font-mono">${(f.size_bytes / 1024).toFixed(1)}k</span>
          </div>
        `).join('');
      }
    } catch (err) {
      console.warn('Could not load project files:', err);
    }
  },

  async runProjectAnalysis() {
    const btn = document.getElementById('run-project-audit-btn');
    const spinner = document.getElementById('project-audit-spinner');
    const resultsContainer = document.getElementById('project-audit-results');

    if (btn) btn.disabled = true;
    if (spinner) spinner.classList.remove('hidden');

    showToast(`Generating Documentation & Grammar Audit using ${AppState.activeMode === 'ollama' ? 'Qwen 2.5 7B' : 'Local Model'}...`, 'info');

    try {
      const res = await fetch('/api/project/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selected_files: this.selectedFiles,
          mode: AppState.activeMode
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Audit failed');
      }

      const data = await res.json();
      this.lastReport = data;
      this.displayAuditReport(data);

      logActivity(`Ran workspace documentation & grammar audit (${data.duration_seconds}s)`);
      showToast('Documentation & Grammar Audit completed!', 'success');
    } catch (err) {
      showToast(`Audit error: ${err.message}`, 'error');
    } finally {
      if (btn) btn.disabled = false;
      if (spinner) spinner.classList.add('hidden');
    }
  },

  displayAuditReport(data) {
    const container = document.getElementById('project-audit-results');
    const textEl = document.getElementById('project-audit-report-text');
    const heuristicsEl = document.getElementById('project-heuristics-list');

    if (!container || !textEl) return;

    container.classList.remove('hidden');

    // Render markdown report
    textEl.innerHTML = window.marked ? window.marked.parse(data.ai_report) : data.ai_report;

    // Render heuristics checklist
    if (heuristicsEl && data.heuristic_checks) {
      heuristicsEl.innerHTML = data.heuristic_checks.map(h => `
        <div class="flex items-center justify-between p-2 rounded-lg bg-slate-900 border border-slate-800 text-xs">
          <div class="flex items-center space-x-2">
            <span class="${h.passed ? 'text-emerald-400' : 'text-amber-400'}">${h.passed ? '✓' : '⚠️'}</span>
            <span class="text-slate-300 font-medium">${h.check}</span>
          </div>
          <span class="text-[10px] text-slate-500">${h.message}</span>
        </div>
      `).join('');
    }
  },

  async runDebugger() {
    const errMsg = document.getElementById('debug-error-message')?.value.trim();
    const stackTrace = document.getElementById('debug-stack-trace')?.value.trim();
    const relevantFile = document.getElementById('debug-relevant-file')?.value.trim();
    const outputEl = document.getElementById('debug-diagnosis-output');

    if (!errMsg || !stackTrace) {
      showToast('Please enter an error message and stack trace', 'warning');
      return;
    }

    showToast('Diagnosing error with AI assistant...', 'info');

    try {
      const res = await fetch('/api/project/debug', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          error_message: errMsg,
          stack_trace: stackTrace,
          relevant_file: relevantFile || null,
          mode: AppState.activeMode
        })
      });

      if (!res.ok) throw new Error('Diagnosis failed');
      const data = await res.json();

      if (outputEl) {
        outputEl.classList.remove('hidden');
        outputEl.innerHTML = `
          <div class="text-xs font-semibold text-indigo-300 mb-2">Diagnosis & Resolution (${data.duration_seconds}s)</div>
          <div class="markdown-body">${window.marked ? window.marked.parse(data.diagnosis) : data.diagnosis}</div>
        `;
      }
      logActivity(`Diagnosed error: ${errMsg.slice(0, 30)}...`);
      showToast('Diagnosis generated successfully!', 'success');
    } catch (err) {
      showToast(`Diagnosis error: ${err.message}`, 'error');
    }
  },

  exportReport() {
    if (!this.lastReport) return;
    const blob = new Blob([this.lastReport.ai_report], { type: 'text/markdown;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `grammar_and_doc_audit_report.md`;
    link.click();
    showToast('Exported audit report', 'success');
  },

  sendReportToChat() {
    if (!this.lastReport) return;
    switchTab('chat');
    if (window.ChatModule) {
      window.ChatModule.usePromptSuggestion(`Here is the repository documentation and grammar audit report for review:\n\n${this.lastReport.ai_report}`);
    }
  }
};

window.ProjectModule = ProjectModule;
