/**
 * Local LLM Studio - Interactive Grammar & Rewriting AI Chat
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
        showToast(`Switched system profile to ${e.target.options[e.target.selectedIndex].text}`, 'info');
      });
    }

    if (contextBtn) {
      contextBtn.addEventListener('click', () => {
        const inspector = document.getElementById('ai-context-inspector');
        if (inspector) inspector.classList.toggle('hidden');
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

    // Scroll to bottom button
    const scrollBottomBtn = document.getElementById('chat-scroll-bottom-btn');
    if (scrollBottomBtn) {
      scrollBottomBtn.addEventListener('click', () => {
        this.scrollToBottom(true);
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

  async loadProfiles() {
    const select = document.getElementById('chat-profile-select');
    if (!select) return;

    try {
      const res = await fetch('/api/settings');
      if (res.ok) {
        const data = await res.json();
        const profiles = data.profiles || {};
        select.innerHTML = Object.entries(profiles).map(([k, v]) => `
          <option value="${k}" ${k === AppState.activeProfile ? 'selected' : ''}>${v.name}</option>
        `).join('');
      }
    } catch (err) {
      console.warn('Could not load profiles:', err);
    }
  },

  async loadProjectFilesForContext() {
    const container = document.getElementById('chat-context-files-list');
    if (!container) return;

    try {
      const res = await fetch('/api/project/files');
      if (res.ok) {
        const files = await res.json();
        container.innerHTML = files.slice(0, 30).map(f => `
          <label class="flex items-center space-x-2 text-xs text-slate-300 hover:bg-slate-800/60 p-1.5 rounded-lg cursor-pointer truncate">
            <input type="checkbox" value="${f.relative_path}" class="context-file-checkbox rounded border-slate-700 bg-slate-900 text-indigo-500 focus:ring-0">
            <span class="truncate font-mono text-[11px]">${f.relative_path}</span>
          </label>
        `).join('');

        container.querySelectorAll('.context-file-checkbox').forEach(cb => {
          cb.addEventListener('change', () => {
            this.selectedContextFiles = Array.from(container.querySelectorAll('.context-file-checkbox:checked')).map(c => c.value);
            this.updateContextInspector();
          });
        });
      }
    } catch (err) {
      console.warn('Could not load context files:', err);
    }
  },

  async handleChatFileAttachment(file) {
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/files/upload', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) throw new Error('Attachment failed');
      const data = await res.json();

      this.attachedChatFiles.push({
        filename: file.name,
        size_bytes: file.size,
        content: data.full_content,
        file_type: data.document.file_type
      });

      this.renderAttachedFilesBadges();
      showToast(`Attached '${file.name}' to chat message`, 'success');
    } catch (err) {
      showToast(`Attachment error: ${err.message}`, 'error');
    }
  },

  renderAttachedFilesBadges() {
    const container = document.getElementById('chat-attached-files-badges');
    if (!container) return;

    if (this.attachedChatFiles.length === 0) {
      container.innerHTML = '';
      container.classList.add('hidden');
      return;
    }

    container.classList.remove('hidden');
    container.innerHTML = this.attachedChatFiles.map((f, idx) => `
      <span class="inline-flex items-center text-[11px] font-medium bg-indigo-950/80 border border-indigo-500/50 text-indigo-300 px-2 py-0.5 rounded-md">
        📎 ${f.filename}
        <button type="button" onclick="ChatModule.removeAttachedFile(${idx})" class="ml-1.5 text-indigo-400 hover:text-rose-400 font-bold">×</button>
      </span>
    `).join('');
  },

  removeAttachedFile(idx) {
    this.attachedChatFiles.splice(idx, 1);
    this.renderAttachedFilesBadges();
  },

  updateContextInspector() {
    const profileBadge = document.getElementById('inspector-profile-badge');
    const filesCount = document.getElementById('inspector-files-count');
    const modeBadge = document.getElementById('inspector-mode-badge');

    if (profileBadge) profileBadge.innerText = AppState.activeProfile;
    if (filesCount) filesCount.innerText = `${this.selectedContextFiles.length} files`;
    if (modeBadge) modeBadge.innerText = AppState.activeMode === 'ollama' ? AppState.activeOllamaModel : 'Local Model';
  },

  async loadConversations() {
    const list = document.getElementById('conversations-history-list');
    if (!list) return;

    try {
      const res = await fetch('/api/chat/conversations');
      if (res.ok) {
        const convs = await res.json();
        AppState.conversations = convs;

        let filtered = convs;
        if (this.searchQuery) {
          filtered = convs.filter(c => c.title.toLowerCase().includes(this.searchQuery));
        }

        if (filtered.length === 0) {
          list.innerHTML = '<div class="text-[11px] text-slate-500 p-2 italic">No conversations yet.</div>';
          return;
        }

        list.innerHTML = filtered.map(c => `
          <div onclick="ChatModule.selectConversation('${c.id}')" 
               class="p-2 rounded-xl border transition cursor-pointer flex items-center justify-between text-xs group ${c.id === this.currentConversationId ? 'bg-indigo-950/40 border-indigo-500/50 text-indigo-200' : 'bg-slate-950/60 border-slate-800/80 hover:bg-slate-800/60 text-slate-300'}">
            <div class="truncate flex-1 pr-2">
              <div class="font-medium truncate">${c.title}</div>
              <div class="text-[10px] text-slate-500 mt-0.5 font-mono">${c.mode} • ${c.message_count} msgs</div>
            </div>
            <button onclick="event.stopPropagation(); ChatModule.deleteConversation('${c.id}')" class="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-rose-400 p-1 rounded">
              <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path></svg>
            </button>
          </div>
        `).join('');
      }
    } catch (err) {
      console.warn('Could not load conversations:', err);
    }
  },

  async selectConversation(id) {
    try {
      const res = await fetch(`/api/chat/conversations/${id}`);
      if (res.ok) {
        const data = await res.json();
        this.currentConversationId = id;
        this.messages = data.messages || [];
        this.renderMessages();
        this.loadConversations();
        this.scrollToBottom(true);
      }
    } catch (err) {
      showToast('Could not load conversation', 'error');
    }
  },

  async deleteConversation(id) {
    try {
      await fetch(`/api/chat/conversations/${id}`, { method: 'DELETE' });
      if (this.currentConversationId === id) {
        this.newChat();
      } else {
        this.loadConversations();
      }
      showToast('Conversation deleted', 'info');
    } catch (err) {
      showToast('Failed to delete conversation', 'error');
    }
  },

  newChat() {
    this.currentConversationId = null;
    this.messages = [];
    this.attachedChatFiles = [];
    this.renderAttachedFilesBadges();
    this.renderMessages();
    this.loadConversations();
    showToast('Started new conversation', 'info');
  },

  clearChat() {
    this.messages = [];
    this.renderMessages();
  },

  stopGeneration() {
    if (this.activeAbortController) {
      this.activeAbortController.abort();
      this.activeAbortController = null;
    }
    this.setGenerating(false);
    showToast('Generation stopped', 'info');
  },

  setGenerating(isGen) {
    this.isGenerating = isGen;
    const sendBtn = document.getElementById('chat-send-btn');
    const stopBtn = document.getElementById('chat-stop-btn');
    const loadingBadge = document.getElementById('chat-streaming-badge');

    if (sendBtn) sendBtn.classList.toggle('hidden', isGen);
    if (stopBtn) stopBtn.classList.toggle('hidden', !isGen);
    if (loadingBadge) loadingBadge.classList.toggle('hidden', !isGen);
  },

  renderMessages() {
    const container = document.getElementById('chat-messages-container');
    if (!container) return;

    if (this.messages.length === 0) {
      container.innerHTML = `
        <div id="chat-empty-state" class="flex flex-col items-center justify-center min-h-[420px] text-center p-8 max-w-xl mx-auto my-auto">
          <div class="w-14 h-14 rounded-2xl bg-indigo-600/15 border border-indigo-500/30 flex items-center justify-center text-indigo-400 mb-4 shadow-lg shadow-indigo-500/10">
            <svg class="w-7 h-7" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"></path></svg>
          </div>
          <h2 class="text-base font-semibold text-slate-100 mb-1.5">Grammar Correction & Rewriting AI Assistant</h2>
          <p class="text-xs text-slate-400 max-w-sm mb-6 leading-relaxed">
            Proofread draft sentences, execute comprehensive rewriting, polish research papers, or diagnose grammatical error categories in real time.
          </p>
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full text-left">
            <button onclick="ChatModule.usePromptSuggestion('Fix obvious typos and grammatical errors in this text cleanly without changing style: ')" 
                    class="p-3 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-indigo-500/40 hover:bg-slate-800/80 transition text-xs text-slate-300">
              <span class="font-medium text-indigo-300 block mb-0.5">🔍 Minimal Proofreading</span>
              Fix spelling, typos, and obvious verb errors.
            </button>
            <button onclick="ChatModule.usePromptSuggestion('Rewrite this sentence for academic clarity and publication-ready formal register: ')" 
                    class="p-3 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-indigo-500/40 hover:bg-slate-800/80 transition text-xs text-slate-300">
              <span class="font-medium text-emerald-300 block mb-0.5">🎓 Academic & Research Polish</span>
              Elevate technical syntax and formal tone.
            </button>
            <button onclick="ChatModule.usePromptSuggestion('Identify, classify, and explain all grammatical and punctuation errors in this text: ')" 
                    class="p-3 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-indigo-500/40 hover:bg-slate-800/80 transition text-xs text-slate-300">
              <span class="font-medium text-amber-300 block mb-0.5">🩺 Error Category Diagnosis</span>
              Breakdown syntax, tenses, and agreement rules.
            </button>
            <button onclick="ChatModule.usePromptSuggestion('Rewrite the following text with maximum clarity, fluency, and conciseness: ')" 
                    class="p-3 rounded-xl border border-slate-800 bg-slate-900/60 hover:border-indigo-500/40 hover:bg-slate-800/80 transition text-xs text-slate-300">
              <span class="font-medium text-purple-300 block mb-0.5">✨ Comprehensive Rewriting</span>
              Restructure sentences for optimal readability.
            </button>
          </div>
        </div>
      `;
      return;
    }

    container.innerHTML = this.messages.map((m, idx) => {
      const isUser = m.role === 'user';
      const renderedHtml = window.marked ? window.marked.parse(m.content) : m.content;
      return `
        <div class="flex flex-col ${isUser ? 'items-end' : 'items-start'} mb-4">
          <div class="flex items-center space-x-2 mb-1 px-1">
            <span class="text-[11px] font-semibold ${isUser ? 'text-indigo-400' : 'text-emerald-400'}">
              ${isUser ? 'You' : (AppState.activeMode === 'ollama' ? 'Qwen 2.5 7B' : 'Local Model')}
            </span>
            <span class="text-[10px] text-slate-500 font-mono">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
          </div>
          <div class="chat-message-bubble p-4 rounded-2xl ${isUser ? 'bg-indigo-600/20 border border-indigo-500/40 text-slate-100 rounded-tr-sm' : 'glass-panel text-slate-200 rounded-tl-sm shadow-xl'}">
            <div class="markdown-body">${renderedHtml}</div>
          </div>
          <div class="flex items-center space-x-2 mt-1 px-1 opacity-0 hover:opacity-100 transition">
            <button onclick="ChatModule.copyMessage(${idx})" class="text-[10px] text-slate-500 hover:text-slate-300 flex items-center space-x-1">
              <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"></path></svg>
              <span>Copy</span>
            </button>
          </div>
        </div>
      `;
    }).join('');

    // Apply syntax highlighting
    if (window.hljs) {
      container.querySelectorAll('pre code').forEach(block => {
        window.hljs.highlightElement(block);
      });
    }
  },

  usePromptSuggestion(text) {
    const input = document.getElementById('chat-input');
    if (input) {
      input.value = text;
      input.focus();
    }
  },

  copyMessage(idx) {
    if (this.messages[idx]) {
      navigator.clipboard.writeText(this.messages[idx].content);
      showToast('Message copied to clipboard', 'info');
    }
  },

  async sendMessage() {
    const input = document.getElementById('chat-input');
    if (!input || !input.value.trim() || this.isGenerating) return;

    let userText = input.value.trim();
    input.value = '';

    // Append attachments if any
    if (this.attachedChatFiles.length > 0) {
      const attachBlocks = this.attachedChatFiles.map(f => `--- Attached File: ${f.filename} ---\n${f.content}`).join('\n\n');
      userText += `\n\n[Attached Files]:\n${attachBlocks}`;
      this.attachedChatFiles = [];
      this.renderAttachedFilesBadges();
    }

    // Append user message
    this.messages.push({ role: 'user', content: userText });
    this.renderMessages();
    this.scrollToBottom(true);
    this.setGenerating(true);

    // Prepare assistant streaming placeholder
    const assistantIndex = this.messages.length;
    this.messages.push({ role: 'assistant', content: '' });

    const payload = {
      conversation_id: this.currentConversationId,
      messages: this.messages.slice(0, -1),
      profile: AppState.activeProfile,
      mode: AppState.activeMode,
      model: AppState.activeMode === 'ollama' ? AppState.activeOllamaModel : null,
      context_files: this.selectedContextFiles
    };

    this.activeAbortController = new AbortController();

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: this.activeAbortController.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP Error ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let accumulatedText = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const rawChunk = decoder.decode(value, { stream: true });
        const lines = rawChunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const jsonStr = line.slice(6).trim();
            if (!jsonStr) continue;

            try {
              const data = JSON.parse(jsonStr);
              if (data.type === 'start') {
                this.currentConversationId = data.conversation_id;
              } else if (data.type === 'token') {
                accumulatedText += data.text || '';
                this.messages[assistantIndex].content = accumulatedText;
                this.renderMessages();
                this.scrollToBottom();
              } else if (data.type === 'end') {
                this.currentConversationId = data.conversation_id;
              } else if (data.type === 'error') {
                accumulatedText += `\n[Error: ${data.message}]`;
                this.messages[assistantIndex].content = accumulatedText;
                this.renderMessages();
              }
            } catch (parseErr) {
              // Partial line buffer
            }
          }
        }
      }

      logActivity(`Generated AI response (${accumulatedText.split(' ').length} words)`);
      this.loadConversations();
    } catch (err) {
      if (err.name === 'AbortError') {
        // Handled cleanly
      } else {
        this.messages[assistantIndex].content = `[Connection error: ${err.message}]`;
        this.renderMessages();
        showToast(`Chat error: ${err.message}`, 'error');
      }
    } finally {
      this.setGenerating(false);
      this.activeAbortController = null;
      this.scrollToBottom(true);
    }
  }
};

window.ChatModule = ChatModule;
