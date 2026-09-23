/**
 * Local LLM Studio - File Intelligence Center & Multi-Document Summarizer
 */

const FilesModule = {
  currentDocument: null,
  uploadedFilesList: [],
  activeTask: 'summarize',
  isProcessing: false,

  init() {
    this.bindEvents();
  },

  bindEvents() {
    const dropZone = document.getElementById('file-drop-zone');
    const fileInput = document.getElementById('file-input');
    const runBtn = document.getElementById('run-file-task-btn');
    const sendToChatBtn = document.getElementById('send-summary-to-chat-btn');
    const downloadSummaryBtn = document.getElementById('download-summary-btn');

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

    // Task selection buttons
    document.querySelectorAll('.file-task-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.file-task-btn').forEach(b => {
          b.classList.remove('bg-indigo-600', 'text-white', 'border-indigo-500');
          b.classList.add('bg-[var(--color-surface)]', 'text-[var(--color-text-secondary)]', 'border-[var(--color-border)]');
        });
        btn.classList.add('bg-indigo-600', 'text-white', 'border-indigo-500');
        btn.classList.remove('bg-[var(--color-surface)]', 'text-[var(--color-text-secondary)]', 'border-[var(--color-border)]');
        this.activeTask = btn.getAttribute('data-task');
      });
    });

    if (runBtn) {
      runBtn.addEventListener('click', () => this.runSummarization());
    }

    if (sendToChatBtn) {
      sendToChatBtn.addEventListener('click', () => this.sendSummaryToChat());
    }

    if (downloadSummaryBtn) {
      downloadSummaryBtn.addEventListener('click', () => this.downloadSummary());
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
    const runBtn = document.getElementById('run-file-task-btn');
    const badgeContainer = document.getElementById('file-classification-badge');

    if (card) card.classList.remove('hidden');
    if (runBtn) runBtn.disabled = false;

    const doc = data.document;
    const cls = this.classifyFile(doc.filename, '.' + doc.file_type.toLowerCase());

    if (badgeContainer) {
      badgeContainer.innerHTML = `<span class="text-xs px-2.5 py-1 rounded-full border ${cls.color}">${cls.badge} (${cls.label})</span>`;
    }

    document.getElementById('meta-filename').innerText = doc.filename;
    document.getElementById('meta-type').innerText = doc.file_type;
    document.getElementById('meta-size').innerText = `${(doc.size_bytes / 1024).toFixed(1)} KB`;
    document.getElementById('meta-words').innerText = `${doc.word_count} words`;
    document.getElementById('meta-tokens').innerText = `~${doc.estimated_tokens} tokens`;
    document.getElementById('meta-chunks').innerText = `${data.chunk_count} chunk${data.chunk_count > 1 ? 's (Hierarchical Multi-Stage)' : ''}`;

    if (preview) {
      preview.innerText = data.full_content;
    }
  },

  async runSummarization() {
    if (!this.currentDocument || this.isProcessing) return;

    this.isProcessing = true;
    const runBtn = document.getElementById('run-file-task-btn');
    const spinner = document.getElementById('file-spinner');
    const resultCard = document.getElementById('file-result-card');
    const customInput = document.getElementById('file-custom-instructions');

    if (runBtn) runBtn.disabled = true;
    if (spinner) spinner.classList.remove('hidden');
    if (resultCard) resultCard.classList.add('hidden');

    try {
      const payload = {
        text: this.currentDocument.full_content,
        filename: this.currentDocument.document.filename,
        task: this.activeTask,
        mode: AppState.activeMode,
        custom_instructions: customInput ? customInput.value.trim() : null
      };

      const res = await fetch('/api/files/summarize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Summarization failed');
      }

      const result = await res.json();
      this.displaySummaryResult(result);
      logActivity(`Generated ${this.activeTask} for ${this.currentDocument.document.filename}`);
      showToast('Document processed successfully!', 'success');
    } catch (err) {
      showToast(`Processing error: ${err.message}`, 'error');
    } finally {
      this.isProcessing = false;
      if (runBtn) runBtn.disabled = false;
      if (spinner) spinner.classList.add('hidden');
    }
  },

  displaySummaryResult(result) {
    const resultCard = document.getElementById('file-result-card');
    const resultContent = document.getElementById('file-summary-content');
    const resultStats = document.getElementById('file-summary-stats');

    if (resultCard) resultCard.classList.remove('hidden');

    if (resultStats) {
      resultStats.innerHTML = `
        <span class="bg-indigo-500/10 text-indigo-300 border border-indigo-500/30 px-2 py-0.5 rounded text-xs font-mono">Engine: ${result.mode_used}</span>
        <span class="bg-emerald-500/10 text-emerald-300 border border-emerald-500/30 px-2 py-0.5 rounded text-xs">${result.summary_words} words</span>
        <span class="bg-[var(--color-elevated)] text-[var(--color-text-secondary)] border border-[var(--color-border)] px-2 py-0.5 rounded text-xs">Compression: ${(result.compression_ratio * 100).toFixed(1)}%</span>
        <span class="bg-[var(--color-elevated)] text-[var(--color-text-secondary)] border border-[var(--color-border)] px-2 py-0.5 rounded text-xs font-mono">${result.duration_seconds}s</span>
        ${result.is_hierarchical ? `<span class="bg-violet-500/10 text-violet-300 border border-violet-500/30 px-2 py-0.5 rounded text-xs font-semibold">Hierarchical (${result.total_chunks} Chunks)</span>` : ''}
      `;
    }

    if (resultContent) {
      resultContent.innerHTML = (typeof marked !== 'undefined') ? marked.parse(result.final_summary) : result.final_summary;
    }

    resultCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
  },

  async copyResult() {
    const resultContent = document.getElementById('file-summary-content');
    if (resultContent) {
      const ok = await copyToClipboard(resultContent.innerText);
      if (ok) showToast('Summary copied to clipboard', 'success');
      else showToast('Failed to copy', 'error');
    }
  },

  downloadSummary() {
    const resultContent = document.getElementById('file-summary-content');
    if (!resultContent || !this.currentDocument) return;

    const text = resultContent.innerText;
    const blob = new Blob([`# Executive Summary: ${this.currentDocument.document.filename}\n\n${text}`], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `summary_${this.currentDocument.document.filename}.md`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('Downloaded summary as Markdown', 'success');
  },

  sendSummaryToChat() {
    if (!this.currentDocument) return;
    const resultContent = document.getElementById('file-summary-content')?.innerText || '';
    
    // Switch to Chat tab
    switchTab('chat');

    if (window.ChatModule) {
      // Attach file info to chat
      window.ChatModule.attachedChatFiles = [{
        filename: this.currentDocument.document.filename,
        size_bytes: this.currentDocument.document.size_bytes,
        file_type: this.currentDocument.document.file_type,
        word_count: this.currentDocument.document.word_count,
        estimated_tokens: this.currentDocument.document.estimated_tokens,
        content: this.currentDocument.full_content
      }];
      window.ChatModule.renderAttachedChips();
      window.ChatModule.setPromptAndSend(`I have reviewed the summary for '${this.currentDocument.document.filename}'. Can you elaborate on the key conclusions and recommend actionable next steps?`);
      showToast(`Transferred '${this.currentDocument.document.filename}' to Interactive Chat`, 'success');
    }
  },

  askSuggestedQuestion(questionText) {
    if (!this.currentDocument) {
      showToast('Please upload or select a document first.', 'warning');
      return;
    }
    switchTab('chat');
    if (window.ChatModule) {
      window.ChatModule.attachedChatFiles = [{
        filename: this.currentDocument.document.filename,
        size_bytes: this.currentDocument.document.size_bytes,
        file_type: this.currentDocument.document.file_type,
        word_count: this.currentDocument.document.word_count,
        estimated_tokens: this.currentDocument.document.estimated_tokens,
        content: this.currentDocument.full_content
      }];
      window.ChatModule.renderAttachedChips();
      window.ChatModule.setPromptAndSend(questionText);
    }
  }
};

window.FilesModule = FilesModule;
document.addEventListener('DOMContentLoaded', () => FilesModule.init());
