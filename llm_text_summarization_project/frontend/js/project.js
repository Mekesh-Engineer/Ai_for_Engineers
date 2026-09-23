/**
 * Local LLM Studio - Project Intelligence & Debugger Module
 */

const ProjectModule = {
  treeData: null,
  filesList: [],
  isAnalyzing: false,

  init() {
    this.bindEvents();
  },

  onShow() {
    this.loadProjectTree();
    this.loadProjectFiles();
  },

  bindEvents() {
    const analyzeBtn = document.getElementById('run-project-analysis-btn');
    const debugBtn = document.getElementById('run-debugger-btn');
    const selectAllBtn = document.getElementById('project-select-all-btn');
    const clearAllBtn = document.getElementById('project-clear-all-btn');
    const sendToChatBtn = document.getElementById('send-project-to-chat-btn');

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
      const isCore = ['README.md', 'main.py', 'requirements.txt', 'config.json', 'model_config.json'].includes(node.name);
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

    const btn = document.getElementById('run-project-analysis-btn');
    const spinner = document.getElementById('project-analysis-spinner');
    const resultCard = document.getElementById('project-analysis-results');
    const reportContent = document.getElementById('project-report-markdown');
    const checksContainer = document.getElementById('project-health-checks');
    const healthScoreBar = document.getElementById('project-health-bar');
    const healthScoreText = document.getElementById('project-health-score-text');

    if (btn) btn.disabled = true;
    if (spinner) spinner.classList.remove('hidden');
    if (resultCard) resultCard.classList.add('hidden');

    const selectedFiles = [];
    document.querySelectorAll('.analysis-file-cb:checked').forEach(cb => {
      selectedFiles.push(cb.value);
    });

    try {
      const res = await fetch('/api/project/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selected_files: selectedFiles,
          mode: AppState.activeMode
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Analysis failed');
      }

      const data = await res.json();
      if (resultCard) resultCard.classList.remove('hidden');

      // Calculate health score from real heuristics
      if (data.heuristic_checks) {
        const passedCount = data.heuristic_checks.filter(c => c.passed).length;
        const totalCount = data.heuristic_checks.length;
        const scorePct = totalCount > 0 ? Math.round((passedCount / totalCount) * 100) : 100;

        if (healthScoreBar) healthScoreBar.style.width = `${scorePct}%`;
        if (healthScoreText) healthScoreText.innerText = `Project Health: ${scorePct}%`;

        if (checksContainer) {
          checksContainer.innerHTML = data.heuristic_checks.map(c => `
            <div class="p-3 rounded-xl border ${c.passed ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' : 'bg-amber-500/10 border-amber-500/30 text-amber-400'} flex items-center justify-between text-xs">
              <span class="font-medium flex items-center">
                <span class="mr-1.5 font-bold">${c.passed ? '✓' : '⚠️'}</span>
                ${c.check}
              </span>
              <span class="opacity-80 text-[11px]">${c.message}</span>
            </div>
          `).join('');
        }
      }

      // Render Markdown report
      if (reportContent) {
        reportContent.innerHTML = (typeof marked !== 'undefined') ? marked.parse(data.ai_report) : data.ai_report;
      }

      resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
      logActivity(`Ran workspace architecture audit (${selectedFiles.length} files)`);
      showToast('Project architecture analysis completed!', 'success');
    } catch (err) {
      showToast(`Analysis error: ${err.message}`, 'error');
    } finally {
      this.isAnalyzing = false;
      if (btn) btn.disabled = false;
      if (spinner) spinner.classList.add('hidden');
    }
  },

  async runDebugger() {
    const errorMsg = document.getElementById('debug-error-msg')?.value.trim();
    const stackTrace = document.getElementById('debug-stack-trace')?.value.trim();
    const relFile = document.getElementById('debug-file-select')?.value;
    const expected = document.getElementById('debug-expected')?.value.trim();
    const actual = document.getElementById('debug-actual')?.value.trim();

    if (!errorMsg && !stackTrace) {
      showToast('Please provide an error message or stack trace to diagnose.', 'warning');
      return;
    }

    const debugBtn = document.getElementById('run-debugger-btn');
    const spinner = document.getElementById('debugger-spinner');
    const resultCard = document.getElementById('debugger-result-card');
    const outputEl = document.getElementById('debugger-output');

    if (debugBtn) debugBtn.disabled = true;
    if (spinner) spinner.classList.remove('hidden');
    if (resultCard) resultCard.classList.add('hidden');

    try {
      const payload = {
        error_message: errorMsg || 'Exception occurred',
        stack_trace: stackTrace || 'N/A',
        relevant_file: relFile || null,
        expected_behavior: expected || null,
        actual_behavior: actual || null,
        mode: AppState.activeMode
      };

      const res = await fetch('/api/project/debug', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Debug failed');
      }

      const data = await res.json();
      if (resultCard) resultCard.classList.remove('hidden');
      if (outputEl) {
        outputEl.innerHTML = (typeof marked !== 'undefined') ? marked.parse(data.diagnosis) : data.diagnosis;
      }
      resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
      logActivity(`Diagnosed error: ${(errorMsg || 'Exception').slice(0, 30)}...`);
      showToast('Diagnostic completed!', 'success');
    } catch (err) {
      showToast(`Debugger error: ${err.message}`, 'error');
    } finally {
      if (debugBtn) debugBtn.disabled = false;
      if (spinner) spinner.classList.add('hidden');
    }
  },

  sendProjectContextToChat() {
    const selectedFiles = [];
    document.querySelectorAll('.analysis-file-cb:checked').forEach(cb => {
      selectedFiles.push(cb.value);
    });

    if (selectedFiles.length === 0) {
      showToast('Please check at least one project file first.', 'warning');
      return;
    }

    switchTab('chat');
    if (window.ChatModule) {
      window.ChatModule.updateContextSelection(selectedFiles);
      window.ChatModule.setPromptAndSend(`I have attached ${selectedFiles.length} core project files to the context. Can you review them and recommend architectural improvements?`);
      showToast(`Attached ${selectedFiles.length} project files to chat context`, 'success');
    }
  }
};

window.ProjectModule = ProjectModule;
document.addEventListener('DOMContentLoaded', () => ProjectModule.init());
