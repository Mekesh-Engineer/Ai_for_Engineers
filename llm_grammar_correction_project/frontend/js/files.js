/**
 * Local LLM Studio - Document Proofreader & Text Rewriting Module
 * Experiment 8: Automated Grammar Error Correction & Text Rewriting
 */

const FilesModule = {
  currentDocument: null,
  uploadedFilesList: [],
  activeLevel: 'standard',
  isProcessing: false,
  showDiffView: false,
  lastResult: null,

  init() {
    this.bindEvents();
  },

  onShow() {
    // If text is in raw editor but no document metadata, maintain state
  },

  bindEvents() {
    const dropZone = document.getElementById('file-drop-zone');
    const fileInput = document.getElementById('file-input');
    const runBtn = document.getElementById('run-proofread-btn');
    const toggleDiffBtn = document.getElementById('toggle-diff-view-btn');
    const sendToChatBtn = document.getElementById('send-proofread-to-chat-btn');
    const downloadBtn = document.getElementById('download-proofread-btn');

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

    // Level selection buttons
    document.querySelectorAll('.file-task-btn, .correction-level-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.file-task-btn, .correction-level-btn').forEach(b => {
          b.classList.remove('bg-indigo-600', 'text-white', 'border-indigo-500');
          b.classList.add('bg-[var(--color-surface)]', 'text-[var(--color-text-secondary)]', 'border-[var(--color-border)]');
        });
        btn.classList.add('bg-indigo-600', 'text-white', 'border-indigo-500');
        btn.classList.remove('bg-[var(--color-surface)]', 'text-[var(--color-text-secondary)]', 'border-[var(--color-border)]');
        this.activeLevel = btn.getAttribute('data-level') || 'standard';
      });
    });

    if (runBtn) {
      runBtn.addEventListener('click', () => this.runProofread());
    }

    if (toggleDiffBtn) {
      toggleDiffBtn.addEventListener('click', () => this.toggleDiffView());
    }

    if (sendToChatBtn) {
      sendToChatBtn.addEventListener('click', () => this.sendProofreadToChat());
    }

    if (downloadBtn) {
      downloadBtn.addEventListener('click', () => this.downloadProofread());
    }
  },

  classifyFile(filename, ext) {
    ext = (ext || '').toLowerCase();
    if (['.py', '.js', '.ts', '.html', '.css', '.c', '.cpp', '.java', '.sql'].includes(ext)) return { label: 'Source Code', badge: '💻 Code', color: 'text-indigo-400 bg-indigo-500/10 border-indigo-500/30' };
    if (['.pdf'].includes(ext)) return { label: 'PDF Document', badge: '📕 PDF', color: 'text-rose-400 bg-rose-500/10 border-rose-500/30' };
    if (['.docx', '.doc'].includes(ext)) return { label: 'Word Document', badge: '📘 DOCX', color: 'text-blue-400 bg-blue-500/10 border-blue-500/30' };
    if (['.csv'].includes(ext)) return { label: 'Tabular Dataset', badge: '📊 CSV', color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30' };
    if (['.json', '.yaml', '.yml', '.toml'].includes(ext)) return { label: 'Config / JSON', badge: '⚙️ Config', color: 'text-amber-400 bg-amber-500/10 border-amber-500/30' };
    return { label: 'Text Document', badge: '📝 Text', color: 'text-[var(--color-text-secondary)] bg-[var(--color-elevated)] border-[var(--color-border)]' };
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
      container.innerHTML = '<div class="text-[11px] text-[var(--color-text-muted)] p-2 italic">No uploaded files yet.</div>';
      return;
    }

    container.innerHTML = this.uploadedFilesList.map((d, idx) => {
      const cls = this.classifyFile(d.document.filename, '.' + d.document.file_type.toLowerCase());
      return `
        <div onclick="FilesModule.selectUploadedFile(${idx})" class="p-2.5 rounded-xl border transition cursor-pointer flex items-center justify-between text-xs ${this.currentDocument === d ? 'bg-indigo-500/10 border-indigo-500/40 text-indigo-300' : 'bg-[var(--color-surface)] border-[var(--color-border)] hover:bg-[var(--color-elevated)] text-[var(--color-text-primary)]'}">
          <div class="truncate flex-1 pr-2">
            <div class="font-medium truncate">${d.document.filename}</div>
            <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">${d.document.word_count} words • ~${d.document.estimated_tokens} tokens</div>
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
    const preview = document.getElementById('file-content-preview');
    const rawInput = document.getElementById('file-raw-text-input');
    const badgeContainer = document.getElementById('file-classification-badge');

    if (card) card.classList.remove('hidden');

    const doc = data.document;
    const cls = this.classifyFile(doc.filename, '.' + doc.file_type.toLowerCase());

    if (badgeContainer) {
      badgeContainer.innerHTML = `<span class="text-xs px-2.5 py-1 rounded-full border ${cls.color}">${cls.badge} (${cls.label})</span>`;
    }

    const setElText = (id, val) => {
      const el = document.getElementById(id);
      if (el) el.innerText = val;
    };

    setElText('meta-filename', doc.filename);
    setElText('meta-type', doc.file_type);
    setElText('meta-size', `${(doc.size_bytes / 1024).toFixed(1)} KB`);
    setElText('meta-words', `${doc.word_count} words`);
    setElText('meta-tokens', `~${doc.estimated_tokens} tokens`);
    setElText('meta-chunks', `${data.chunk_count} chunk${data.chunk_count > 1 ? 's (Hierarchical Multi-Stage)' : ''}`);

    if (preview) {
      preview.innerText = data.full_content;
    }
    if (rawInput && (!rawInput.value || rawInput.value.trim() === '')) {
      rawInput.value = data.full_content;
    }
  },

  async runProofread() {
    if (this.isProcessing) return;

    const rawInput = document.getElementById('file-raw-text-input');
    let textToProcess = rawInput ? rawInput.value.trim() : '';

    if (!textToProcess && this.currentDocument) {
      textToProcess = this.currentDocument.full_content;
    }

    if (!textToProcess) {
      showToast('Please upload a file or enter text to proofread.', 'warning');
      return;
    }

    this.isProcessing = true;
    const runBtn = document.getElementById('run-proofread-btn');
    const spinner = document.getElementById('proofread-spinner');
    const resultCard = document.getElementById('proofread-results-container');
    const customInput = document.getElementById('file-custom-instructions');

    if (runBtn) runBtn.disabled = true;
    if (spinner) spinner.classList.remove('hidden');
    if (resultCard) resultCard.classList.add('hidden');

    try {
      const filename = this.currentDocument ? this.currentDocument.document.filename : 'document';
      const payload = {
        text: textToProcess,
        filename: filename,
        correction_level: this.activeLevel,
        mode: AppState.activeMode,
        custom_instructions: customInput ? customInput.value.trim() : null
      };

      const res = await fetch('/api/files/proofread', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Proofreading execution failed');
      }

      const result = await res.json();
      this.lastResult = result;
      this.displayProofreadResult(result);
      logActivity(`Proofread & corrected ${filename} (${result.original_words} words, level: ${this.activeLevel})`);
      showToast('Text successfully corrected & polished!', 'success');
    } catch (err) {
      showToast(`Proofread error: ${err.message}`, 'error');
    } finally {
      this.isProcessing = false;
      if (runBtn) runBtn.disabled = false;
      if (spinner) spinner.classList.add('hidden');
    }
  },

  displayProofreadResult(result) {
    const resultCard = document.getElementById('proofread-results-container');
    const outputText = document.getElementById('proofread-output-text');
    const diffText = document.getElementById('proofread-diff-text');
    const statsContainer = document.getElementById('proofread-summary-stats');

    if (resultCard) resultCard.classList.remove('hidden');

    const f1Pct = Math.round((result.token_f1 || 0) * 100);

    if (statsContainer) {
      statsContainer.innerHTML = `
        <span class="bg-indigo-500/10 text-indigo-300 border border-indigo-500/30 px-2 py-0.5 rounded text-xs font-mono">Engine: ${result.mode_used}</span>
        <span class="bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded text-xs font-mono">Words: ${result.original_words} → ${result.corrected_words}</span>
        <span class="bg-amber-500/10 text-amber-300 border border-amber-500/30 px-2 py-0.5 rounded text-xs font-mono">Levenshtein: ${result.levenshtein_distance}</span>
        <span class="bg-violet-500/10 text-violet-300 border border-violet-500/30 px-2 py-0.5 rounded text-xs font-mono">Token F1: ${f1Pct}%</span>
        <span class="bg-[var(--color-elevated)] text-[var(--color-text-secondary)] border border-[var(--color-border)] px-2 py-0.5 rounded text-xs font-mono">${result.duration_seconds}s</span>
        ${result.is_hierarchical ? `<span class="bg-violet-500/10 text-violet-300 border border-violet-500/30 px-2 py-0.5 rounded text-xs font-semibold">Hierarchical (${result.total_chunks} Chunks)</span>` : ''}
      `;
    }

    if (outputText) {
      outputText.innerHTML = (typeof marked !== 'undefined') ? marked.parse(result.corrected_text) : result.corrected_text;
    }

    if (diffText) {
      diffText.innerHTML = (typeof marked !== 'undefined') ? marked.parse(result.diff_markup || result.corrected_text) : (result.diff_markup || result.corrected_text);
    }

    this.showDiffView = false;
    this.updateDiffViewVisibility();

    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  },

  toggleDiffView() {
    this.showDiffView = !this.showDiffView;
    this.updateDiffViewVisibility();
  },

  updateDiffViewVisibility() {
    const cleanCard = document.getElementById('proofread-output-card');
    const diffCard = document.getElementById('proofread-diff-card');
    const toggleBtn = document.getElementById('toggle-diff-view-btn');

    if (cleanCard) cleanCard.classList.toggle('hidden', this.showDiffView);
    if (diffCard) diffCard.classList.toggle('hidden', !this.showDiffView);

    if (toggleBtn) {
      toggleBtn.innerText = this.showDiffView ? 'Show Clean Output' : 'Show Inline Diff';
      toggleBtn.className = `px-2.5 py-1 rounded-md text-xs transition font-medium border ${this.showDiffView ? 'bg-indigo-600 text-white border-indigo-500' : 'bg-[var(--color-elevated)] hover:bg-[var(--color-surface)] border-[var(--color-border)] text-[var(--color-text-secondary)]'}`;
    }
  },

  async copyResult() {
    if (!this.lastResult) return;
    const textToCopy = this.showDiffView ? (this.lastResult.diff_markup || this.lastResult.corrected_text) : this.lastResult.corrected_text;
    const ok = await copyToClipboard(textToCopy);
    if (ok) showToast('Copied corrected text to clipboard!', 'success');
  },

  downloadProofread() {
    if (!this.lastResult) return;
    const filename = `corrected_${(this.lastResult.filename || 'text').replace(/\.[^/.]+$/, "")}.md`;
    const content = `# Corrected Document: ${this.lastResult.filename || 'text'}\n` +
                    `*Level: ${this.lastResult.correction_level} | Mode: ${this.lastResult.mode_used} | Levenshtein Distance: ${this.lastResult.levenshtein_distance}*\n\n` +
                    `## Corrected Text\n\n${this.lastResult.corrected_text}\n\n` +
                    `## Inline Diff\n\n${this.lastResult.diff_markup}\n`;

    const blob = new Blob([content], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`Downloaded '${filename}'`, 'success');
  },

  sendProofreadToChat() {
    if (!this.lastResult) return;
    switchTab('chat');
    if (window.ChatModule) {
      const prompt = `Review this grammar-corrected text and explain the rationale for changes:\n\n**Corrected Text:**\n${this.lastResult.corrected_text.slice(0, 1500)}`;
      window.ChatModule.setPromptAndSend(prompt);
    }
  },

  askSuggestedQuestion(question) {
    if (!this.lastResult) return;
    switchTab('chat');
    if (window.ChatModule) {
      const prompt = `${question}\n\n**Context (Corrected Text):**\n${this.lastResult.corrected_text.slice(0, 1500)}`;
      window.ChatModule.setPromptAndSend(prompt);
    }
  }
};

window.FilesModule = FilesModule;
