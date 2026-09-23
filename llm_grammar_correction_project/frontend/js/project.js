/**
 * Local LLM Studio - Project Documentation & Code Grammar Auditor Module
 * Experiment 8: Automated Grammar Error Correction & Text Rewriting
 */

const ProjectModule = {
  treeData: null,
  filesList: [],
  isAnalyzing: false,
  lastAuditReport: null,

  init() {
    this.bindEvents();
  },

  onShow() {
    this.loadProjectTree();
    this.loadProjectFiles();
  },

  bindEvents() {
    const analyzeBtn = document.getElementById('run-project-analysis-btn') || document.getElementById('run-project-audit-btn');
    const debugBtn = document.getElementById('run-debugger-btn');
    const selectAllBtn = document.getElementById('project-select-all-btn');
    const clearAllBtn = document.getElementById('project-clear-all-btn');
    const sendToChatBtn = document.getElementById('send-project-to-chat-btn');
    const exportReportBtn = document.getElementById('export-audit-report-btn');
    const sendAuditChatBtn = document.getElementById('send-audit-to-chat-btn');

    if (analyzeBtn) {
      analyzeBtn.addEventListener('click', () => this.runAnalysis());
    }

    if (debugBtn) {
      debugBtn.addEventListener('click', () => this.runDebugger());
    }

    if (selectAllBtn) {
      selectAllBtn.addEventListener('click', () => {
        document.querySelectorAll('.analysis-file-cb').forEach(cb => cb.checked = true);
      });
    }

    if (clearAllBtn) {
      clearAllBtn.addEventListener('click', () => {
        document.querySelectorAll('.analysis-file-cb').forEach(cb => cb.checked = false);
      });
    }

    if (sendToChatBtn) {
      sendToChatBtn.addEventListener('click', () => this.sendProjectContextToChat());
    }

    if (exportReportBtn) {
      exportReportBtn.addEventListener('click', () => this.exportAuditReport());
    }

    if (sendAuditChatBtn) {
      sendAuditChatBtn.addEventListener('click', () => this.sendAuditToChat());
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
      container.innerHTML = `<div class="text-rose-400 text-xs p-2">Error loading directory tree: ${err.message}</div>`;
    }
  },

  renderTreeNode(node) {
    if (node.is_dir) {
      const childrenHtml = (node.children || []).map(child => this.renderTreeNode(child)).join('');
      return `
        <div class="my-0.5">
          <div class="flex items-center text-xs text-indigo-400 font-semibold py-1 px-1.5 rounded hover:bg-[var(--color-elevated)] cursor-pointer">
            <span class="mr-1.5">📁</span>
            <span>${node.name}</span>
          </div>
          <div class="pl-4 border-l border-[var(--color-border)] my-0.5 space-y-0.5">
            ${childrenHtml}
          </div>
        </div>
      `;
    } else {
      const isCore = ['README.md', 'main.py', 'requirements.txt', 'Experiment8.md', 'model_config.json'].includes(node.name);
      return `
        <div class="flex items-center justify-between text-xs text-[var(--color-text-secondary)] py-0.5 px-1.5 rounded hover:bg-[var(--color-elevated)] font-mono">
          <span class="flex items-center">
            <span class="mr-1.5 text-[var(--color-text-muted)]">📄</span>
            <span class="${isCore ? 'text-indigo-400 font-medium' : ''}">${node.name}</span>
          </span>
          <span class="text-[10px] text-[var(--color-text-muted)]">${(node.size_bytes / 1024).toFixed(1)} KB</span>
        </div>
      `;
    }
  },

  async loadProjectFiles() {
    try {
      const res = await fetch('/api/project/files');
      if (res.ok) {
        this.filesList = await res.json();
        this.populateFileCheckboxes();
        this.populateDebugFileDropdown();
        this.updateProjectStatsBadge();
      }
    } catch (err) {
      console.warn('Error fetching project files:', err);
    }
  },

  updateProjectStatsBadge() {
    const badge = document.getElementById('sidebar-project-files-count');
    if (badge) {
      badge.innerText = `${this.filesList.length} files`;
    }
  },

  populateFileCheckboxes() {
    const container = document.getElementById('project-analysis-file-list');
    if (!container) return;

    container.innerHTML = '';
    this.filesList.forEach(f => {
      const item = document.createElement('label');
      item.className = 'flex items-center space-x-2 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] p-1 rounded hover:bg-[var(--color-elevated)] cursor-pointer';
      item.innerHTML = `
        <input type="checkbox" value="${f.relative_path}" class="analysis-file-cb rounded border-[var(--color-border)] text-indigo-600 focus:ring-indigo-500" ${f.is_important ? 'checked' : ''}>
        <span class="truncate font-mono">${f.relative_path}</span>
        ${f.is_important ? '<span class="text-[9px] px-1 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">Core</span>' : ''}
      `;
      container.appendChild(item);
    });
  },

  populateDebugFileDropdown() {
    const select = document.getElementById('debug-file-select');
    if (!select) return;

    select.innerHTML = '<option value="">-- None (General Issue) --</option>';
    this.filesList.forEach(f => {
      const opt = document.createElement('option');
      opt.value = f.relative_path;
      opt.textContent = f.relative_path;
      select.appendChild(opt);
    });
  },

  async runAnalysis() {
    if (this.isAnalyzing) return;
    this.isAnalyzing = true;

    const btn = document.getElementById('run-project-analysis-btn') || document.getElementById('run-project-audit-btn');
    const spinner = document.getElementById('project-analysis-spinner') || document.getElementById('project-audit-spinner');

    if (btn) btn.disabled = true;
    if (spinner) spinner.classList.remove('hidden');

    const selected = [];
    document.querySelectorAll('.analysis-file-cb:checked').forEach(cb => selected.push(cb.value));

    try {
      const res = await fetch('/api/project/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selected_files: selected.length > 0 ? selected : null,
          mode: AppState.activeMode
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Audit failed');
      }

      const report = await res.json();
      this.lastAuditReport = report;
      this.displayAnalysisReport(report);
      logActivity(`Completed documentation and grammar audit for ${selected.length || this.filesList.length} files`);
      showToast('Workspace documentation audit finished!', 'success');
    } catch (err) {
      showToast(`Audit error: ${err.message}`, 'error');
    } finally {
      this.isAnalyzing = false;
      if (btn) btn.disabled = false;
      if (spinner) spinner.classList.add('hidden');
    }
  },

  displayAnalysisReport(report) {
    const resultsContainer = document.getElementById('project-analysis-results') || document.getElementById('project-audit-results');
    const reportMarkdown = document.getElementById('project-report-markdown') || document.getElementById('project-audit-report-text');
    const healthText = document.getElementById('project-health-score-text');
    const healthBar = document.getElementById('project-health-bar');
    const checksContainer = document.getElementById('project-health-checks') || document.getElementById('project-heuristics-list');

    if (resultsContainer) resultsContainer.classList.remove('hidden');

    // Score & progress bar
    const score = report.health_score !== undefined ? report.health_score : 100;
    if (healthText) healthText.innerText = `Documentation Health: ${score}%`;
    if (healthBar) healthBar.style.width = `${score}%`;

    // Checks Grid
    if (checksContainer && report.checks) {
      checksContainer.innerHTML = report.checks.map(c => `
        <div class="p-2.5 rounded-lg border text-xs ${c.passed ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-amber-500/10 border-amber-500/30'}">
          <div class="flex items-center justify-between font-semibold ${c.passed ? 'text-emerald-400' : 'text-amber-400'}">
            <span>${c.name}</span>
            <span>${c.passed ? '✓ PASS' : '⚠ NOTICE'}</span>
          </div>
          <div class="text-[11px] text-[var(--color-text-secondary)] mt-0.5">${c.detail}</div>
        </div>
      `).join('');
    }

    if (reportMarkdown) {
      const content = report.report_markdown || report.summary || 'Audit complete.';
      reportMarkdown.innerHTML = (typeof marked !== 'undefined') ? marked.parse(content) : content;
    }

    if (resultsContainer) {
      resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  },

  async runDebugger() {
    const errorMsgInput = document.getElementById('debug-error-msg') || document.getElementById('debug-error-message');
    const stackTraceInput = document.getElementById('debug-stack-trace');
    const fileSelect = document.getElementById('debug-file-select') || document.getElementById('debug-relevant-file');
    const expectedInput = document.getElementById('debug-expected');
    const actualInput = document.getElementById('debug-actual');
    const btn = document.getElementById('run-debugger-btn');
    const spinner = document.getElementById('debugger-spinner');
    const resultCard = document.getElementById('debugger-result-card') || document.getElementById('debug-diagnosis-output');
    const outputEl = document.getElementById('debugger-output') || document.getElementById('debug-diagnosis-output');

    const errorMsg = errorMsgInput ? errorMsgInput.value.trim() : '';
    const stackTrace = stackTraceInput ? stackTraceInput.value.trim() : '';

    if (!errorMsg && !stackTrace) {
      showToast('Please enter an error message or stack trace to diagnose.', 'warning');
      return;
    }

    if (btn) btn.disabled = true;
    if (spinner) spinner.classList.remove('hidden');
    if (resultCard) resultCard.classList.add('hidden');

    try {
      const payload = {
        error_message: errorMsg,
        stack_trace: stackTrace,
        relevant_file: fileSelect ? fileSelect.value : null,
        expected_behavior: expectedInput ? expectedInput.value.trim() : null,
        actual_behavior: actualInput ? actualInput.value.trim() : null,
        mode: AppState.activeMode
      };

      const res = await fetch('/api/project/debug', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Diagnosis failed');
      }

      const data = await res.json();
      if (resultCard) resultCard.classList.remove('hidden');
      if (outputEl) {
        outputEl.innerHTML = (typeof marked !== 'undefined') ? marked.parse(data.diagnosis) : data.diagnosis;
      }
      logActivity(`Diagnosed error: ${errorMsg || 'Exception traceback'}`);
      showToast('Error diagnosis completed!', 'success');
    } catch (err) {
      showToast(`Debugger error: ${err.message}`, 'error');
    } finally {
      if (btn) btn.disabled = false;
      if (spinner) spinner.classList.add('hidden');
    }
  },

  exportAuditReport() {
    if (!this.lastAuditReport) return;
    const content = this.lastAuditReport.report_markdown || '# Documentation & Grammar Audit Report\n\nNo report data.';
    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `grammar_audit_report_${new Date().toISOString().slice(0, 10)}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast('Exported audit report!', 'success');
  },

  sendAuditToChat() {
    if (!this.lastAuditReport) return;
    switchTab('chat');
    if (window.ChatModule) {
      const prompt = `Review the following repository documentation and docstring audit report and create a detailed action plan to fix all grammar, punctuation, and style issues:\n\n${(this.lastAuditReport.report_markdown || '').slice(0, 1500)}`;
      window.ChatModule.setPromptAndSend(prompt);
    }
  },

  sendProjectContextToChat() {
    const selected = [];
    document.querySelectorAll('.analysis-file-cb:checked').forEach(cb => selected.push(cb.value));

    if (selected.length === 0) {
      showToast('No files selected.', 'warning');
      return;
    }

    switchTab('chat');
    if (window.ChatModule) {
      window.ChatModule.updateContextSelection(selected);
      showToast(`Loaded ${selected.length} workspace files into AI chat context`, 'success');
    }
  }
};

window.ProjectModule = ProjectModule;
