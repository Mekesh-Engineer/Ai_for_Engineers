/**
 * Local LLM Studio - Interactive AI Chat & Workspace Controller
 * Experiment 8: Automated Grammar Error Correction & Text Rewriting
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
      const modeText = AppState.activeMode === 'ollama' ? `Ollama (${AppState.activeOllamaModel})` : 'Project Local Model (Seq2Seq)';
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
        if (select && data.profiles) {
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

    if (modelEl) modelEl.innerText = AppState.activeMode === 'ollama' ? `Qwen 2.5 7B (Ollama)` : 'Local Seq2Seq Model';
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
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
          </div>
          <h3 class="text-sm font-semibold text-[var(--color-text-primary)] mb-1">Grammar Error Correction & Text Rewriting Studio</h3>
          <p class="text-xs max-w-sm text-[var(--color-text-secondary)] mb-5">Interact with local <strong>Qwen 2.5 7B</strong> or project Seq2Seq models. Fix grammatical errors, polish academic writing, diagnose linguistic rules, and rewrite text.</p>

          <!-- Starter Action Grid -->
          <div class="grid grid-cols-2 md:grid-cols-3 gap-2 max-w-lg w-full text-left">
            <div onclick="ChatModule.setPromptAndSend('Correct all grammatical errors in this sentence and explain the rule: He go to the laboratory yesterday for doing the experiment.')" class="glass-panel glass-panel-hover p-2.5 rounded-lg cursor-pointer">
              <div class="text-xs font-semibold text-[var(--color-text-primary)]">Grammar Correction</div>
              <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">Tense, agreement & syntax</div>
            </div>

            <div onclick="ChatModule.setPromptAndSend('Polish this paragraph for academic publication in a top-tier peer-reviewed journal: The experimental results demonstrates that our proposed neural network is more superior than baseline methods.')" class="glass-panel glass-panel-hover p-2.5 rounded-lg cursor-pointer">
              <div class="text-xs font-semibold text-[var(--color-text-primary)]">Academic Polish</div>
              <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">Formal vocabulary & tone</div>
            </div>

            <div onclick="ChatModule.setPromptAndSend('Diagnose all grammatical errors in this text and provide rule explanations with category labels: Each of the component have different function.')" class="glass-panel glass-panel-hover p-2.5 rounded-lg cursor-pointer">
              <div class="text-xs font-semibold text-[var(--color-text-primary)]">Diagnose Rules</div>
              <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">Detailed error categories</div>
            </div>

            <div onclick="ChatModule.setPromptAndSend('Rewrite this text to be clear, concise, and engaging without changing the underlying meaning: Because of it lacks of enough training data, the model perform poorly.')" class="glass-panel glass-panel-hover p-2.5 rounded-lg cursor-pointer">
              <div class="text-xs font-semibold text-[var(--color-text-primary)]">Clarity & Rewriting</div>
              <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">Streamlined structure</div>
            </div>

            <div onclick="ChatModule.setPromptAndSend('Proofread and enhance the docstrings and comments in our Python codebase.')" class="glass-panel glass-panel-hover p-2.5 rounded-lg cursor-pointer">
              <div class="text-xs font-semibold text-[var(--color-text-primary)]">Code Docstrings</div>
              <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">Technical formatting</div>
            </div>

            <div onclick="ChatModule.setPromptAndSend('Generate a set of 5 difficult grammar correction benchmark questions with ungrammatical inputs and gold-standard targets.')" class="glass-panel glass-panel-hover p-2.5 rounded-lg cursor-pointer">
              <div class="text-xs font-semibold text-[var(--color-text-primary)]">Benchmark Tests</div>
              <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">Evaluation test cases</div>
            </div>
          </div>
        </div>
      `;
      return;
    }

    container.innerHTML = this.messages.map((m, idx) => {
      const isUser = m.role === 'user';
      const parsedContent = (typeof marked !== 'undefined') ? marked.parse(m.content) : m.content;

      return `
        <div class="flex ${isUser ? 'justify-end' : 'justify-start'}">
          <div class="chat-message-bubble rounded-xl p-3 text-xs sm:text-sm ${isUser ? 'bg-indigo-600 text-white rounded-br-none shadow-md' : 'glass-panel text-[var(--color-text-primary)] rounded-bl-none'}">
            ${m.attachedFiles && m.attachedFiles.length > 0 ? `
              <div class="flex flex-wrap gap-1 mb-2 pb-2 border-b ${isUser ? 'border-indigo-400/30' : 'border-[var(--color-border)]'}">
                ${m.attachedFiles.map(f => `<span class="px-1.5 py-0.5 text-[10px] font-mono rounded ${isUser ? 'bg-indigo-700/50 text-indigo-100' : 'bg-indigo-500/10 text-indigo-300'}">📎 ${f}</span>`).join('')}
              </div>
            ` : ''}

            <div class="${isUser ? '' : 'markdown-body'} leading-relaxed">${isUser ? m.content.replace(/\n/g, '<br>') : parsedContent}</div>

            <div class="flex items-center justify-between mt-2 pt-1 border-t ${isUser ? 'border-indigo-400/20 text-indigo-200' : 'border-[var(--color-border)] text-[var(--color-text-muted)]'} text-[10px]">
              <span>${m.timestamp || ''}</span>
              <div class="flex items-center space-x-1">
                <button onclick="ChatModule.copyMessage(${idx})" class="hover:underline opacity-80 hover:opacity-100">Copy</button>
                ${!isUser && idx === this.messages.length - 1 ? `<span class="opacity-50">•</span><button onclick="ChatModule.regenerateLastMessage()" class="hover:underline opacity-80 hover:opacity-100">Regenerate</button>` : ''}
              </div>
            </div>
          </div>
        </div>
      `;
    }).join('');
  },

  async copyMessage(idx) {
    const msg = this.messages[idx];
    if (msg) {
      await copyToClipboard(msg.content);
      showToast('Copied message to clipboard', 'info', 2000);
    }
  },

  setPromptAndSend(promptText) {
    const input = document.getElementById('chat-input');
    if (input) {
      input.value = promptText;
      input.focus();
      this.sendMessage();
    }
  },

  async regenerateLastMessage() {
    if (this.isGenerating || this.messages.length === 0) return;
    const lastMsg = this.messages[this.messages.length - 1];
    if (lastMsg.role === 'assistant') {
      this.messages.pop();
      const lastUserMsg = this.messages[this.messages.length - 1];
      if (lastUserMsg && lastUserMsg.role === 'user') {
        const prompt = lastUserMsg.content;
        this.executeStream(prompt, true);
      }
    }
  },

  stopGeneration() {
    if (this.activeAbortController) {
      this.activeAbortController.abort();
      this.activeAbortController = null;
    }
    this.isGenerating = false;
    this.setLoading(false);
    showToast('Generation cancelled.', 'warning', 2000);
  },

  setLoading(isLoading) {
    this.isGenerating = isLoading;
    const loadingBar = document.getElementById('chat-loading-indicator');
    const sendBtn = document.getElementById('chat-send-btn');
    const stopBtn = document.getElementById('chat-stop-btn');

    if (loadingBar) loadingBar.classList.toggle('hidden', !isLoading);
    if (sendBtn) sendBtn.classList.toggle('hidden', isLoading);
    if (stopBtn) stopBtn.classList.toggle('hidden', !isLoading);
  },

  async sendMessage() {
    const input = document.getElementById('chat-input');
    if (!input || this.isGenerating) return;

    const text = input.value.trim();
    if (!text && this.attachedChatFiles.length === 0) return;

    const attachedNames = this.attachedChatFiles.map(f => f.filename);
    const userMessage = {
      role: 'user',
      content: text,
      attachedFiles: attachedNames,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    this.messages.push(userMessage);
    input.value = '';
    input.style.height = 'auto';

    this.renderMessages();
    this.scrollToBottom(true);
    this.updateContextInspector();

    await this.executeStream(text);
  },

  async executeStream(userPrompt, isRegen = false) {
    this.setLoading(true);
    this.activeAbortController = new AbortController();

    const assistantMsg = {
      role: 'assistant',
      content: '',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };
    this.messages.push(assistantMsg);
    const assistantMsgIndex = this.messages.length - 1;

    // Build context payload
    const contextFiles = [...this.selectedContextFiles];
    const attachedData = this.attachedChatFiles.map(f => ({
      filename: f.filename,
      content: f.content
    }));

    try {
      const payload = {
        prompt: userPrompt,
        conversation_id: this.currentConversationId,
        mode: AppState.activeMode,
        profile: AppState.activeProfile,
        context_files: contextFiles,
        attached_files: attachedData,
        history: this.messages.slice(0, -2).map(m => ({ role: m.role, content: m.content }))
      };

      const res = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: this.activeAbortController.signal
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP status ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // Keep last partial line

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || trimmed.startsWith(':')) continue;

          if (trimmed.startsWith('data: ')) {
            const dataStr = trimmed.slice(6);
            if (dataStr === '[DONE]') break;

            try {
              const data = JSON.parse(dataStr);
              const textPiece = (data.text !== undefined) ? data.text : (data.token !== undefined ? data.token : '');
              if (textPiece) {
                this.messages[assistantMsgIndex].content += textPiece;
                this.renderMessages();
                this.scrollToBottom();
              }
              if (data.conversation_id) {
                this.currentConversationId = data.conversation_id;
              }
              if (data.type === 'error') {
                this.messages[assistantMsgIndex].content += `\n\n*(Error: ${data.message || 'Generation failed'})*`;
                this.renderMessages();
              }
            } catch (jsonErr) {
              // Plain text token fallback
              if (dataStr && !dataStr.startsWith('{')) {
                this.messages[assistantMsgIndex].content += dataStr;
                this.renderMessages();
                this.scrollToBottom();
              }
            }
          }
        }
      }

      logActivity(`Received AI response (${this.messages[assistantMsgIndex].content.split(/\s+/).length} words)`);
      this.loadConversations();
    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Stream aborted by user.');
      } else {
        this.messages[assistantMsgIndex].content += `\n\n*(Error during generation: ${err.message})*`;
        this.renderMessages();
        showToast(`Chat error: ${err.message}`, 'error');
      }
    } finally {
      this.setLoading(false);
      this.activeAbortController = null;
      this.attachedChatFiles = [];
      this.renderAttachedChips();
      this.updateContextInspector();
      this.scrollToBottom(true);
    }
  },

  exportConversation(format = 'markdown') {
    if (this.messages.length === 0) {
      showToast('No messages to export.', 'warning');
      return;
    }

    let content = '';
    let filename = `conversation_${new Date().toISOString().slice(0, 10)}`;
    let mime = 'text/plain';

    if (format === 'markdown') {
      content = `# Local LLM Studio — AI Grammar & Rewriting Conversation\n` +
                `Date: ${new Date().toLocaleString()}\n` +
                `Engine: ${AppState.activeMode}\n\n---\n\n`;
      this.messages.forEach(m => {
        content += `### ${m.role === 'user' ? 'User' : 'Assistant'} (${m.timestamp || ''})\n\n${m.content}\n\n`;
      });
      filename += '.md';
      mime = 'text/markdown';
    } else if (format === 'json') {
      content = JSON.stringify(this.messages, null, 2);
      filename += '.json';
      mime = 'application/json';
    } else {
      this.messages.forEach(m => {
        content += `[${m.role.toUpperCase()}] (${m.timestamp || ''}):\n${m.content}\n\n`;
      });
      filename += '.txt';
    }

    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    showToast(`Exported conversation as ${filename}`, 'success');
  }
};

window.ChatModule = ChatModule;
