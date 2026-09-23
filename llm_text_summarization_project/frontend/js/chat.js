/**
 * Local LLM Studio - Interactive AI Chat & Workspace Controller
 */

const ChatModule = {
  messages: [],
  currentConversationId: null,
  activeAbortController: null,
  isGenerating: false,
  selectedContextFiles: [],
  attachedChatFiles: [],
  searchQuery: '',
  hasUnreadStreamTokens: false,

  init() {
    this.bindEvents();
    this.loadProfiles();
    this.loadProjectFilesForContext();
    this.initScrollListener();
  },

  onShow() {
    this.updateHeader();
    this.loadConversations();
    this.updateContextInspector();
    setTimeout(() => this.scrollToBottom(true), 100);
  },

  updateHeader() {
    const badge = document.getElementById('chat-model-badge');
    if (badge) {
      const modeText = AppState.activeMode === 'ollama' ? `Ollama (${AppState.activeOllamaModel})` : 'Project Local Model';
      badge.innerText = `Model: ${modeText}`;
    }
  },

  bindEvents() {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('chat-input');
    const stopBtn = document.getElementById('chat-stop-btn');
    const clearBtn = document.getElementById('chat-clear-btn');
    const newChatBtn = document.getElementById('new-chat-btn');
    const profileSelect = document.getElementById('chat-profile-select');
    const contextBtn = document.getElementById('chat-context-toggle-btn');
    const attachBtn = document.getElementById('chat-attach-btn');
    const attachInput = document.getElementById('chat-file-input');
    const searchInput = document.getElementById('search-conversations-input');

    if (form) {
      form.addEventListener('submit', (e) => {
        e.preventDefault();
        this.sendMessage();
      });
    }

    if (input) {
      input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 160) + 'px';
      });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.sendMessage();
        }
      });
    }

    if (stopBtn) {
      stopBtn.addEventListener('click', () => this.stopGeneration());
    }

    if (clearBtn) {
      clearBtn.addEventListener('click', () => this.clearChat());
    }

    if (newChatBtn) {
      newChatBtn.addEventListener('click', () => this.newChat());
    }

    if (profileSelect) {
      profileSelect.addEventListener('change', (e) => {
        AppState.activeProfile = e.target.value;
        this.updateContextInspector();
        showToast(`Profile: ${e.target.options[e.target.selectedIndex].text}`, 'info');
      });
    }

    if (contextBtn) {
      contextBtn.addEventListener('click', () => {
        const drawer = document.getElementById('context-files-drawer');
        if (drawer) drawer.classList.toggle('hidden');
      });
    }

    if (attachBtn && attachInput) {
      attachBtn.addEventListener('click', () => attachInput.click());
      attachInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
          this.handleChatFileAttachment(e.target.files[0]);
        }
      });
    }

    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        this.searchQuery = e.target.value.toLowerCase().trim();
        this.loadConversations();
      });
    }
  },

  initScrollListener() {
    const container = document.getElementById('chat-messages-container');
    if (container) {
      container.addEventListener('scroll', () => {
        if (this.isChatNearBottom()) {
          this.hideScrollBottomButton();
          this.hasUnreadStreamTokens = false;
        } else {
          this.showScrollBottomButton(this.hasUnreadStreamTokens);
        }
      });
    }
  },

  isChatNearBottom() {
    const el = document.getElementById('chat-messages-container');
    if (!el) return true;
    return (el.scrollHeight - el.scrollTop - el.clientHeight) < 120;
  },

  scrollToBottom(force = false) {
    const container = document.getElementById('chat-messages-container');
    if (!container) return;

    if (force || this.isChatNearBottom()) {
      container.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      });
      this.hideScrollBottomButton();
      this.hasUnreadStreamTokens = false;
    } else {
      this.hasUnreadStreamTokens = true;
      this.showScrollBottomButton(true);
    }
  },

  showScrollBottomButton(hasNewMessages = false) {
    const btn = document.getElementById('chat-scroll-bottom-btn');
    if (!btn) return;
    btn.classList.remove('hidden');
    const text = btn.querySelector('.scroll-btn-text');
    if (text) {
      text.innerText = hasNewMessages ? '↓ New messages' : '↓ Scroll to bottom';
    }
  },

  hideScrollBottomButton() {
    const btn = document.getElementById('chat-scroll-bottom-btn');
    if (btn) btn.classList.add('hidden');
  },

  async handleChatFileAttachment(file) {
    const formData = new FormData();
    formData.append('file', file);

    showToast(`Uploading '${file.name}'...`, 'info', 2000);

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
      this.attachedChatFiles.push({
        filename: file.name,
        size_bytes: data.document.size_bytes,
        file_type: data.document.file_type,
        word_count: data.document.word_count,
        estimated_tokens: data.document.estimated_tokens,
        content: data.full_content
      });

      this.renderAttachedChips();
      this.updateContextInspector();
      showToast(`Attached '${file.name}'`, 'success');
    } catch (err) {
      showToast(`Attachment error: ${err.message}`, 'error');
    }
  },

  removeAttachedFile(idx) {
    this.attachedChatFiles.splice(idx, 1);
    this.renderAttachedChips();
    this.updateContextInspector();
  },

  renderAttachedChips() {
    const container = document.getElementById('chat-attached-chips');
    if (!container) return;

    if (this.attachedChatFiles.length === 0) {
      container.classList.add('hidden');
      container.innerHTML = '';
      return;
    }

    container.classList.remove('hidden');
    container.innerHTML = this.attachedChatFiles.map((f, idx) => `
      <div class="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-md bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-xs font-mono">
        <span>${f.filename}</span>
        <span class="text-[10px] text-indigo-400">(${(f.size_bytes / 1024).toFixed(1)} KB)</span>
        <button type="button" onclick="ChatModule.removeAttachedFile(${idx})" class="ml-1 text-indigo-400 hover:text-rose-400 font-bold">×</button>
      </div>
    `).join('');
  },

  async loadProfiles() {
    try {
      const res = await fetch('/api/settings');
      if (res.ok) {
        const data = await res.json();
        const select = document.getElementById('chat-profile-select');
        if (select) {
          select.innerHTML = '';
          for (const [key, profile] of Object.entries(data.profiles)) {
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = profile.name;
            if (key === AppState.activeProfile) opt.selected = true;
            select.appendChild(opt);
          }
        }
      }
    } catch (err) {
      console.warn('Failed to load profiles:', err);
    }
  },

  async loadProjectFilesForContext() {
    try {
      const res = await fetch('/api/project/files');
      if (res.ok) {
        const files = await res.json();
        const list = document.getElementById('context-files-list');
        if (list) {
          list.innerHTML = '';
          files.forEach(f => {
            const item = document.createElement('label');
            item.className = 'flex items-center space-x-2 text-xs text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)] p-1.5 rounded hover:bg-[var(--color-elevated)] cursor-pointer';
            item.innerHTML = `
              <input type="checkbox" value="${f.relative_path}" class="context-file-checkbox rounded border-[var(--color-border)] text-indigo-600 focus:ring-indigo-500">
              <span class="truncate font-mono">${f.relative_path}</span>
              ${f.is_important ? '<span class="text-[9px] bg-indigo-500/10 text-indigo-300 px-1 rounded border border-indigo-500/20">Core</span>' : ''}
            `;
            list.appendChild(item);
          });

          document.querySelectorAll('.context-file-checkbox').forEach(cb => {
            cb.addEventListener('change', () => this.updateContextSelection());
          });
        }
      }
    } catch (err) {
      console.warn('Failed to load project context files:', err);
    }
  },

  updateContextSelection(filesOverride = null) {
    if (Array.isArray(filesOverride)) {
      this.selectedContextFiles = filesOverride;
      document.querySelectorAll('.context-file-checkbox').forEach(cb => {
        cb.checked = filesOverride.includes(cb.value);
      });
    } else {
      const selected = [];
      document.querySelectorAll('.context-file-checkbox:checked').forEach(cb => {
        selected.push(cb.value);
      });
      this.selectedContextFiles = selected;
    }

    const badge = document.getElementById('context-count-badge');
    if (badge) {
      const totalAttached = this.selectedContextFiles.length + this.attachedChatFiles.length;
      if (totalAttached > 0) {
        badge.innerText = `${totalAttached} Attached`;
        badge.classList.remove('hidden');
      } else {
        badge.classList.add('hidden');
      }
    }

    this.updateContextInspector();
  },

  updateContextInspector() {
    const modelEl = document.getElementById('inspector-model-name');
    const profileEl = document.getElementById('inspector-profile-name');
    const filesCountEl = document.getElementById('inspector-files-count');
    const tokensEl = document.getElementById('inspector-token-est');
    const progressBar = document.getElementById('inspector-token-bar');

    const totalFiles = this.selectedContextFiles.length + this.attachedChatFiles.length;
    let estimatedTokens = totalFiles * 450 + this.messages.length * 80;
    this.attachedChatFiles.forEach(f => estimatedTokens += (f.estimated_tokens || 0));

    if (modelEl) modelEl.innerText = AppState.activeMode === 'ollama' ? `Qwen 2.5 7B (Ollama)` : 'Local Checkpoint';
    if (profileEl) profileEl.innerText = AppState.activeProfile.replace('_', ' ').toUpperCase();
    if (filesCountEl) filesCountEl.innerText = `${totalFiles} files`;
    if (tokensEl) tokensEl.innerText = `~${estimatedTokens.toLocaleString()} / 32,000 tokens`;
    if (progressBar) {
      const pct = Math.min(100, Math.round((estimatedTokens / 32000) * 100));
      progressBar.style.width = `${pct}%`;
    }
  },

  async loadConversations() {
    try {
      const res = await fetch('/api/chat/conversations');
      if (res.ok) {
        let list = await res.json();
        if (this.searchQuery) {
          list = list.filter(c => c.title.toLowerCase().includes(this.searchQuery));
        }

        const container = document.getElementById('conversation-history-list');
        if (container) {
          if (list.length === 0) {
            container.innerHTML = '<div class="text-[11px] text-[var(--color-text-muted)] p-2 italic">No conversations found.</div>';
            return;
          }

          container.innerHTML = list.map(c => `
            <div onclick="ChatModule.switchConversation('${c.id}')" class="group flex items-center justify-between p-2 rounded-lg text-xs cursor-pointer transition ${c.id === this.currentConversationId ? 'bg-indigo-500/10 text-indigo-400 border border-indigo-500/30 font-medium' : 'text-[var(--color-text-secondary)] hover:bg-[var(--color-elevated)] hover:text-[var(--color-text-primary)]'}">
              <div class="truncate flex-1 pr-2">
                <div class="truncate">${c.title || 'Untitled Session'}</div>
                <div class="text-[10px] text-[var(--color-text-muted)] flex items-center space-x-1.5 mt-0.5 font-mono">
                  <span>${c.message_count} msgs</span>
                  <span>•</span>
                  <span>${c.mode}</span>
                </div>
              </div>
              <button onclick="ChatModule.deleteConversation('${c.id}', event)" class="opacity-0 group-hover:opacity-100 text-[var(--color-text-muted)] hover:text-rose-400 text-sm px-1 transition">×</button>
            </div>
          `).join('');
        }
      }
    } catch (err) {
      console.warn('Failed to load conversations:', err);
    }
  },

  async switchConversation(cid) {
    try {
      const res = await fetch(`/api/chat/conversations/${cid}`);
      if (res.ok) {
        const data = await res.json();
        this.currentConversationId = cid;
        this.messages = data.messages || [];
        this.renderMessages();
        this.loadConversations();
        this.updateContextInspector();
        setTimeout(() => this.scrollToBottom(true), 50);
      }
    } catch (err) {
      showToast(`Error loading chat: ${err.message}`, 'error');
    }
  },

  async deleteConversation(cid, e) {
    if (e) e.stopPropagation();
    try {
      await fetch(`/api/chat/conversations/${cid}`, { method: 'DELETE' });
      if (this.currentConversationId === cid) {
        this.newChat();
      } else {
        this.loadConversations();
      }
    } catch (err) {
      showToast(`Delete failed: ${err.message}`, 'error');
    }
  },

  newChat() {
    this.currentConversationId = null;
    this.messages = [];
    this.attachedChatFiles = [];
    this.renderAttachedChips();
    this.renderMessages();
    this.loadConversations();
    this.updateContextInspector();
    this.hideScrollBottomButton();
  },

  clearChat() {
    this.messages = [];
    this.renderMessages();
    this.updateContextInspector();
    this.hideScrollBottomButton();
  },

  renderMessages() {
    const container = document.getElementById('chat-messages-container');
    if (!container) return;

    if (this.messages.length === 0) {
      container.innerHTML = `
        <div class="h-full flex flex-col items-center justify-center text-center p-6 text-[var(--color-text-muted)]">
          <div class="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/20 flex items-center justify-center text-indigo-400 mb-2.5">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
          </div>
          <h3 class="text-sm font-semibold text-[var(--color-text-primary)] mb-1">Local AI Development Workspace</h3>
          <p class="text-xs max-w-sm text-[var(--color-text-secondary)] mb-5">Interact with local <strong>Qwen 2.5 7B</strong> or project models. Ingest codebases, synthesize documents, and debug exceptions.</p>

          <!-- Starter Action Grid -->
          <div class="grid grid-cols-2 md:grid-cols-3 gap-2 max-w-lg w-full text-left">
            <div onclick="ChatModule.setPromptAndSend('Explain the architecture and main modules of this project.')" class="glass-panel glass-panel-hover p-2.5 rounded-lg cursor-pointer">
              <div class="text-xs font-semibold text-[var(--color-text-primary)]">Architecture</div>
              <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">Explore modules & data flow</div>
            </div>

            <div onclick="ChatModule.setPromptAndSend('Review the codebase for potential bugs, security issues, or performance bottlenecks.')" class="glass-panel glass-panel-hover p-2.5 rounded-lg cursor-pointer">
              <div class="text-xs font-semibold text-[var(--color-text-primary)]">Find Code Bugs</div>
              <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">Static diagnostic review</div>
            </div>

            <div onclick="ChatModule.setPromptAndSend('Generate comprehensive pytest test cases for the main backend services.')" class="glass-panel glass-panel-hover p-2.5 rounded-lg cursor-pointer">
              <div class="text-xs font-semibold text-[var(--color-text-primary)]">Generate Tests</div>
              <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">Unit & edge-case suites</div>
            </div>

            <div onclick="ChatModule.setPromptAndSend('Draft a professional README.md with system architecture diagrams and API docs.')" class="glass-panel glass-panel-hover p-2.5 rounded-lg cursor-pointer">
              <div class="text-xs font-semibold text-[var(--color-text-primary)]">Documentation</div>
              <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">Technical API specs</div>
            </div>

            <div onclick="ChatModule.setPromptAndSend('How do I optimize local LLM inference performance and reduce token latency?')" class="glass-panel glass-panel-hover p-2.5 rounded-lg cursor-pointer">
              <div class="text-xs font-semibold text-[var(--color-text-primary)]">Optimize Runtime</div>
              <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">Batching & GPU offload</div>
            </div>

            <div onclick="switchTab('files')" class="glass-panel glass-panel-hover p-2.5 rounded-lg cursor-pointer">
              <div class="text-xs font-semibold text-[var(--color-text-primary)]">Summarize File</div>
              <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">Upload PDF, code or doc</div>
            </div>
          </div>
        </div>
      `;
      return;
    }

    container.innerHTML = '';
    this.messages.forEach((msg, idx) => {
      const isUser = msg.role === 'user';
      const msgDiv = document.createElement('div');
      msgDiv.className = `flex ${isUser ? 'justify-end' : 'justify-start'} mb-3`;
      msgDiv.id = `chat-msg-${idx}`;

      let displayContent = msg.content;
      let attachedBadgeHtml = '';

      if (isUser && msg.content.startsWith('[ATTACHED FILES:')) {
        const endAttachIdx = msg.content.indexOf('[END OF ATTACHMENTS]');
        if (endAttachIdx !== -1) {
          const attachHeader = msg.content.substring(0, endAttachIdx);
          displayContent = msg.content.substring(endAttachIdx + 20).trim();
          
          const match = attachHeader.match(/Attached File: ([^\n\r]+)/);
          const filename = match ? match[1].trim() : 'Document';
          attachedBadgeHtml = `
            <div class="inline-flex items-center space-x-1 px-2 py-0.5 rounded bg-indigo-500/10 border border-indigo-500/20 text-[10px] font-mono mb-1.5 text-indigo-300">
              <span>Attached: ${filename}</span>
            </div>
          `;
        }
      }

      const parsedHtml = (typeof marked !== 'undefined') ? marked.parse(displayContent) : displayContent;

      msgDiv.innerHTML = `
        <div class="chat-message-bubble rounded-xl p-3.5 ${isUser ? 'bg-indigo-600 text-white shadow-sm' : 'glass-panel text-[var(--color-text-primary)]'}">
          <div class="flex items-center justify-between text-[11px] font-medium mb-1.5 opacity-80 border-b border-white/10 pb-1">
            <span class="flex items-center">
              ${isUser ? 'You' : 'Local LLM Studio'}
            </span>
            ${!isUser ? `
              <div class="flex items-center space-x-2 text-xs font-normal">
                <button onclick="ChatModule.copyText(${idx})" class="hover:text-indigo-400 transition">Copy</button>
                <button onclick="ChatModule.regenerateResponse()" class="hover:text-indigo-400 transition">Regenerate</button>
              </div>
            ` : ''}
          </div>
          ${attachedBadgeHtml}
          <div class="markdown-body text-xs sm:text-sm leading-relaxed">${parsedHtml}</div>

          ${!isUser && idx === this.messages.length - 1 && !this.isGenerating ? `
            <div class="mt-2.5 pt-2 border-t border-[var(--color-border)] flex flex-wrap gap-1 text-[11px]">
              <span class="text-[var(--color-text-muted)] py-0.5">Suggestions:</span>
              <button onclick="ChatModule.setPromptAndSend('Can you explain that in more technical detail?')" class="px-2 py-0.5 rounded-md bg-[var(--color-elevated)] hover:bg-[var(--color-surface)] text-[var(--color-text-secondary)] border border-[var(--color-border)] transition">Explain further</button>
              <button onclick="ChatModule.setPromptAndSend('Provide a complete, executable code implementation.')" class="px-2 py-0.5 rounded-md bg-[var(--color-elevated)] hover:bg-[var(--color-surface)] text-[var(--color-text-secondary)] border border-[var(--color-border)] transition">Show implementation</button>
              <button onclick="ChatModule.setPromptAndSend('Are there any edge cases or potential pitfalls with this?')" class="px-2 py-0.5 rounded-md bg-[var(--color-elevated)] hover:bg-[var(--color-surface)] text-[var(--color-text-secondary)] border border-[var(--color-border)] transition">Check edge cases</button>
            </div>
          ` : ''}
        </div>
      `;
      container.appendChild(msgDiv);
    });

    this.scrollToBottom();
  },

  setPromptAndSend(prompt) {
    const input = document.getElementById('chat-input');
    if (input) {
      input.value = prompt;
      this.sendMessage();
    }
  },

  async copyText(idx) {
    if (this.messages[idx]) {
      const ok = await copyToClipboard(this.messages[idx].content);
      if (ok) showToast('Copied to clipboard', 'success');
      else showToast('Failed to copy to clipboard', 'error');
    }
  },

  regenerateResponse() {
    if (this.messages.length < 2 || this.isGenerating) return;
    if (this.messages[this.messages.length - 1].role === 'assistant') {
      this.messages.pop();
    }
    const lastUserMsg = this.messages[this.messages.length - 1];
    if (lastUserMsg && lastUserMsg.role === 'user') {
      this.messages.pop();
      let promptText = lastUserMsg.content;
      if (promptText.startsWith('[ATTACHED FILES:')) {
        const endAttachIdx = promptText.indexOf('[END OF ATTACHMENTS]');
        if (endAttachIdx !== -1) {
          promptText = promptText.substring(endAttachIdx + 20).trim();
        }
      }
      this.setPromptAndSend(promptText);
    }
  },

  exportConversation(format = 'markdown') {
    if (this.messages.length === 0) {
      showToast('No conversation to export.', 'warning');
      return;
    }

    let exportContent = '';
    let filename = `session_${Date.now()}`;

    if (format === 'markdown') {
      filename += '.md';
      exportContent = `# Local LLM Studio Session Export\nDate: ${new Date().toLocaleString()}\nModel: ${AppState.activeMode}\n\n---\n\n`;
      this.messages.forEach(m => {
        exportContent += `### ${m.role === 'user' ? 'User' : 'Assistant'}\n\n${m.content}\n\n---\n\n`;
      });
    } else if (format === 'json') {
      filename += '.json';
      exportContent = JSON.stringify({
        exported_at: new Date().toISOString(),
        mode: AppState.activeMode,
        messages: this.messages
      }, null, 2);
    } else {
      filename += '.txt';
      this.messages.forEach(m => {
        exportContent += `[${m.role.toUpperCase()}]:\n${m.content}\n\n========================================\n\n`;
      });
    }

    const blob = new Blob([exportContent], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    showToast(`Exported session (${format.toUpperCase()})`, 'success');
  },

  stopGeneration() {
    if (this.activeAbortController) {
      this.activeAbortController.abort();
      this.activeAbortController = null;
    }
    this.setGeneratingState(false);
  },

  setGeneratingState(generating) {
    this.isGenerating = generating;
    const sendBtn = document.getElementById('chat-send-btn');
    const stopBtn = document.getElementById('chat-stop-btn');
    const indicator = document.getElementById('chat-loading-indicator');

    if (sendBtn) sendBtn.classList.toggle('hidden', generating);
    if (stopBtn) stopBtn.classList.toggle('hidden', !generating);
    if (indicator) indicator.classList.toggle('hidden', !generating);
  },

  async sendMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text || this.isGenerating) return;

    input.value = '';
    input.style.height = 'auto';

    let finalUserText = text;
    if (this.attachedChatFiles.length > 0) {
      const fileBlocks = this.attachedChatFiles.map(f => `--- Attached File: ${f.filename} ---\n${f.content}`);
      finalUserText = `[ATTACHED FILES:\n${fileBlocks.join('\n\n')}\n[END OF ATTACHMENTS]\n\n${text}`;
    }

    this.messages.push({ role: 'user', content: finalUserText });
    this.renderMessages();

    const assistantMsgIndex = this.messages.length;
    this.messages.push({ role: 'assistant', content: '' });
    this.renderMessages();
    this.scrollToBottom(true);

    this.setGeneratingState(true);
    this.activeAbortController = new AbortController();

    try {
      const payload = {
        conversation_id: this.currentConversationId,
        messages: this.messages.slice(0, -1),
        profile: AppState.activeProfile,
        mode: AppState.activeMode,
        context_files: this.selectedContextFiles
      };

      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: this.activeAbortController.signal
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let partialLine = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunkText = decoder.decode(value, { stream: true });
        const lines = (partialLine + chunkText).split('\n');
        partialLine = lines.pop();

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const rawData = line.substring(6).trim();
            if (!rawData) continue;
            try {
              const event = JSON.parse(rawData);
              if (event.type === 'start') {
                this.currentConversationId = event.conversation_id;
              } else if (event.type === 'token') {
                this.messages[assistantMsgIndex].content += event.text;
                
                const activeBubble = document.querySelector(`#chat-msg-${assistantMsgIndex} .markdown-body`);
                if (activeBubble) {
                  activeBubble.innerHTML = (typeof marked !== 'undefined') ? marked.parse(this.messages[assistantMsgIndex].content) : this.messages[assistantMsgIndex].content;
                } else {
                  this.renderMessages();
                }

                this.scrollToBottom(false);
              } else if (event.type === 'end') {
                this.renderMessages();
                this.loadConversations();
                this.updateContextInspector();
                logActivity(`Completed inference with ${AppState.activeMode}`);
              } else if (event.type === 'error') {
                showToast(event.message, 'error');
              }
            } catch (jsonErr) {
              console.warn('Error parsing SSE data:', jsonErr);
            }
          }
        }
      }

      // Flush any trailing buffer
      if (partialLine && partialLine.startsWith('data: ')) {
        const rawData = partialLine.substring(6).trim();
        if (rawData) {
          try {
            const event = JSON.parse(rawData);
            if (event.type === 'token') {
              this.messages[assistantMsgIndex].content += event.text;
            }
          } catch (_) {}
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        showToast(`Chat error: ${err.message}`, 'error');
        this.messages[assistantMsgIndex].content += `\n\n*[Generation error: ${err.message}]*`;
        this.renderMessages();
      }
    } finally {
      this.setGeneratingState(false);
      this.activeAbortController = null;
      this.renderMessages();
    }
  }
};

window.ChatModule = ChatModule;
document.addEventListener('DOMContentLoaded', () => ChatModule.init());