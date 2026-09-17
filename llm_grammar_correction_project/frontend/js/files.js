/**
 * Local LLM Studio - Document Proofreading & Text Rewriting Studio
 */

const FilesModule = {
  currentDocument: null,
  uploadedFilesList: [],
  activeCorrectionLevel: 'standard',
  isProcessing: false,
  lastCorrectionResult: null,

  init() {
    this.bindEvents();
  },

  bindEvents() {
    const dropZone = document.getElementById('file-drop-zone');
    const fileInput = document.getElementById('file-input');
    const runBtn = document.getElementById('run-proofread-btn');
    const sendToChatBtn = document.getElementById('send-proofread-to-chat-btn');
    const downloadBtn = document.getElementById('download-proofread-btn');
    const diffToggleBtn = document.getElementById('toggle-diff-view-btn');

    if (dropZone && fileInput) {
      dropZone.addEventListener('click', () => fileInput.click());

      dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('border-indigo-500', 'bg-indigo-950/20');
      });

      dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('border-indigo-500', 'bg-indigo-950/20');
      });

      dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('border-indigo-500', 'bg-indigo-950/20');
        if (e.dataTransfer.files.length > 0) {
          this.handleFileUpload(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
          this.handleFileUpload(e.target.files[0]);
        }
      });
    }

    // Correction level buttons
    document.querySelectorAll('.correction-level-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.correction-level-btn').forEach(b => {
          b.classList.remove('bg-indigo-600', 'text-white', 'border-indigo-500');
          b.classList.add('bg-slate-900', 'text-slate-300', 'border-slate-800');
        });
        btn.classList.add('bg-indigo-600', 'text-white', 'border-indigo-500');
        btn.classList.remove('bg-slate-900', 'text-slate-300', 'border-slate-800');
        this.activeCorrectionLevel = btn.getAttribute('data-level');
      });
    });

    if (runBtn) {
      runBtn.addEventListener('click', () => this.runProofreading());
    }

    if (sendToChatBtn) {
      sendToChatBtn.addEventListener('click', () => this.sendToChat());
    }

    if (downloadBtn) {
      downloadBtn.addEventListener('click', () => this.downloadResult());
    }

    if (diffToggleBtn) {
      diffToggleBtn.addEventListener('click', () => this.toggleDiffView());
    }
  },

  classifyFile(filename, ext) {
    ext = (ext || '').toLowerCase();
    if (['.py', '.js', '.ts', '.html', '.css', '.c', '.cpp', '.java', '.sql'].includes(ext)) return { label: 'Source Code', badge: '💻 Code', color: 'text-indigo-400 bg-indigo-950/60 border-indigo-800' };
    if (['.pdf'].includes(ext)) return { label: 'PDF Document', badge: '📕 PDF', color: 'text-rose-400 bg-rose-950/60 border-rose-800' };
    if (['.docx', '.doc'].includes(ext)) return { label: 'Word Document', badge: '📘 DOCX', color: 'text-blue-400 bg-blue-950/60 border-blue-800' };
    if (['.csv'].includes(ext)) return { label: 'Tabular Dataset', badge: '📊 CSV', color: 'text-emerald-400 bg-emerald-950/60 border-emerald-800' };
    if (['.json', '.yaml', '.yml', '.toml'].includes(ext)) return { label: 'Config / JSON', badge: '⚙️ Config', color: 'text-amber-400 bg-amber-950/60 border-amber-800' };
    return { label: 'Text Document', badge: '📝 Text', color: 'text-slate-400 bg-slate-900 border-slate-800' };
  },

  async handleFileUpload(file) {
    const formData = new FormData();
    formData.append('file', file);

    const uploadIndicator = document.getElementById('file-upload-indicator');
    if (uploadIndicator) uploadIndicator.classList.remove('hidden');

    try {
      const res = await fetch('/api/files/upload', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Upload failed');
      }

      const data = await res.json();
      this.currentDocument = data;
      this.uploadedFilesList.unshift(data);

      this.displayDocumentMetadata(data);
      this.renderUploadedFilesList();
      logActivity(`Uploaded & analyzed document: ${file.name}`);
      showToast(`Uploaded and parsed '${file.name}' (${data.document.word_count} words)`, 'success');
    } catch (err) {
      showToast(`File error: ${err.message}`, 'error');
    } finally {
      if (uploadIndicator) uploadIndicator.classList.add('hidden');
    }
  },

  renderUploadedFilesList() {
    const container = document.getElementById('uploaded-files-history-list');
    if (!container) return;

    if (this.uploadedFilesList.length === 0) {
      container.innerHTML = '<div class="text-[11px] text-slate-500 p-2 italic">No uploaded files yet.</div>';
      return;
    }

    container.innerHTML = this.uploadedFilesList.map((d, idx) => {
      const cls = this.classifyFile(d.document.filename, '.' + d.document.file_type.toLowerCase());
      return `
        <div onclick="FilesModule.selectUploadedFile(${idx})" class="p-2.5 rounded-xl border transition cursor-pointer flex items-center justify-between text-xs ${this.currentDocument === d ? 'bg-indigo-950/40 border-indigo-500/50 text-indigo-200' : 'bg-slate-950/60 border-slate-800 hover:bg-slate-800/60 text-slate-300'}">
          <div class="truncate flex-1 pr-2">
            <div class="font-medium truncate">${d.document.filename}</div>
            <div class="text-[10px] text-slate-500 mt-0.5">${d.document.word_count} words • ~${d.document.estimated_tokens} tokens</div>
          </div>
          <span class="text-[10px] px-2 py-0.5 rounded border ${cls.color}">${cls.badge}</span>
        </div>
      `;
    }).join('');
  },

  selectUploadedFile(idx) {
    if (this.uploadedFilesList[idx]) {
      this.currentDocument = this.uploadedFilesList[idx];
      this.displayDocumentMetadata(this.currentDocument);
      this.renderUploadedFilesList();
    }
  },

  displayDocumentMetadata(data) {
    const card = document.getElementById('file-metadata-card');
    const inputArea = document.getElementById('file-raw-text-input');
    if (!card) return;

    card.classList.remove('hidden');
    document.getElementById('meta-filename').innerText = data.document.filename;
    document.getElementById('meta-type').innerText = data.document.file_type;
    document.getElementById('meta-words').innerText = data.document.word_count;
    document.getElementById('meta-tokens').innerText = data.document.estimated_tokens;
    document.getElementById('meta-chunks').innerText = `${data.chunk_count} chunk${data.chunk_count > 1 ? 's' : ''}`;

    if (inputArea) {
      inputArea.value = data.full_content;
    }
  },

  async runProofreading() {
    const inputArea = document.getElementById('file-raw-text-input');
    const textToProcess = inputArea ? inputArea.value.trim() : (this.currentDocument ? this.currentDocument.full_content : '');

    if (!textToProcess) {
      showToast('Please enter text or upload a document to proofread', 'warning');
      return;
    }

    const runBtn = document.getElementById('run-proofread-btn');
    const loader = document.getElementById('proofread-spinner');
    const resultsContainer = document.getElementById('proofread-results-container');
    const customInstructions = document.getElementById('file-custom-instructions')?.value.trim();

    if (runBtn) runBtn.disabled = true;
    if (loader) loader.classList.remove('hidden');

    showToast(`Proofreading with '${this.activeCorrectionLevel}' level using ${AppState.activeMode === 'ollama' ? 'Qwen 2.5 7B' : 'Local Model'}...`, 'info');

    try {
      const res = await fetch('/api/files/proofread', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text: textToProcess,
          filename: this.currentDocument ? this.currentDocument.document.filename : 'document.txt',
          correction_level: this.activeCorrectionLevel,
          mode: AppState.activeMode,
          custom_instructions: customInstructions
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Proofreading failed');
      }

      const data = await res.json();
      this.lastCorrectionResult = data;
      this.displayResults(data);

      logActivity(`Proofread '${data.filename}' (Levenshtein: ${data.levenshtein_distance}, Token F1: ${data.token_f1})`);
      showToast(`Proofreading completed in ${data.duration_seconds}s!`, 'success');
    } catch (err) {
      showToast(`Error: ${err.message}`, 'error');
    } finally {
      if (runBtn) runBtn.disabled = false;
      if (loader) loader.classList.add('hidden');
    }
  },

  displayResults(data) {
    const container = document.getElementById('proofread-results-container');
    if (!container) return;

    container.classList.remove('hidden');

    // Update metric badges
    document.getElementById('res-duration').innerText = `${data.duration_seconds}s`;
    document.getElementById('res-edit-dist').innerText = data.levenshtein_distance;
    document.getElementById('res-token-f1').innerText = (data.token_f1 * 100).toFixed(1) + '%';
    document.getElementById('res-words-count').innerText = `${data.original_words} → ${data.corrected_words}`;

    // Render corrected output
    const outputEl = document.getElementById('proofread-output-text');
    if (outputEl) {
      outputEl.innerHTML = window.marked ? window.marked.parse(data.corrected_text) : data.corrected_text;
    }

    // Render inline diff
    const diffEl = document.getElementById('proofread-diff-text');
    if (diffEl) {
      diffEl.innerHTML = window.marked ? window.marked.parse(data.diff_markup) : data.diff_markup;
    }
  },

  toggleDiffView() {
    const outputCard = document.getElementById('proofread-output-card');
    const diffCard = document.getElementById('proofread-diff-card');
    const toggleBtn = document.getElementById('toggle-diff-view-btn');

    if (!outputCard || !diffCard) return;

    const isDiffVisible = !diffCard.classList.contains('hidden');
    if (isDiffVisible) {
      diffCard.classList.add('hidden');
      outputCard.classList.remove('hidden');
      if (toggleBtn) toggleBtn.innerText = 'Show Inline Diff View';
    } else {
      diffCard.classList.remove('hidden');
      outputCard.classList.add('hidden');
      if (toggleBtn) toggleBtn.innerText = 'Show Clean Output View';
    }
  },

  sendToChat() {
    if (!this.lastCorrectionResult) return;
    switchTab('chat');
    if (window.ChatModule) {
      window.ChatModule.usePromptSuggestion(`Here is the proofread text for review:\n\n${this.lastCorrectionResult.corrected_text}`);
    }
  },

  downloadResult() {
    if (!this.lastCorrectionResult) return;
    const blob = new Blob([this.lastCorrectionResult.corrected_text], { type: 'text/markdown;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `corrected_${this.lastCorrectionResult.filename || 'document.md'}`;
    link.click();
    showToast('Downloaded corrected document', 'success');
  }
};

window.FilesModule = FilesModule;
