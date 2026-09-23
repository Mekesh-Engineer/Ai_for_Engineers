/**
 * VOLTIX — Chat Module
 * Handles: conversation history, streaming SSE, message rendering,
 * citations, context drawer, scroll intelligence, message actions.
 */

const ChatModule = {
  messages: [],
  currentConversationId: null,
  activeAbortController: null,
  isGenerating: false,
  attachedChatFiles: [],
  lastUserPrompt: null,
  searchQuery: '',

  /* ─── INIT ─── */
  init() {
    this.bindEvents();
    this.loadConversations();
    this.initScrollListener();
  },

  /* ─── EVENT BINDING ─── */
  bindEvents() {
    // Textarea auto-resize + Enter to send
    const input = document.getElementById('chat-input');
    if (input) {
      input.addEventListener('input', () => {
        input.style.height = 'auto';
        input.style.height = Math.min(input.scrollHeight, 200) + 'px';
        // Enable/disable send button
        const sendBtn = document.getElementById('chat-send-btn');
        if (sendBtn) sendBtn.disabled = input.value.trim() === '';
      });
      input.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.sendMessage();
        }
      });
    }

    // Send & stop buttons
    document.getElementById('chat-send-btn')?.addEventListener('click', () => this.sendMessage());
    document.getElementById('chat-stop-btn')?.addEventListener('click', () => this.stopGeneration());

    // New chat button (sidebar)
    document.getElementById('btn-new-chat')?.addEventListener('click', () => {
      this.newChat();
      if (window.innerWidth < 1024) closeMobileSidebar();
    });

    // Conversation search
    const searchInput = document.getElementById('input-search-conv');
    if (searchInput) {
      let debounceTimer;
      searchInput.addEventListener('input', e => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          this.searchQuery = e.target.value.toLowerCase().trim();
          this.loadConversations();
        }, 250);
      });
    }

    // Mode select (in composer footer)
    document.getElementById('chat-mode-select')?.addEventListener('change', e => {
      AppState.activeMode = e.target.value;
      this.updateContextDrawer();
      showToast(`Mode: ${e.target.value}`, 'info', 1600);
    });

    // File upload via input
    const fileInput = document.getElementById('file-upload-input');
    if (fileInput) {
      fileInput.addEventListener('change', e => {
        const file = e.target.files?.[0];
        if (file) {
          this.handleChatFileAttachment(file);
          fileInput.value = ''; // reset so same file can be re-selected
        }
      });
    }

    // Welcome screen prompt chips
    document.querySelectorAll('.welcome-chip').forEach(chip => {
      chip.addEventListener('click', () => {
        const prompt = chip.dataset.prompt;
        if (prompt) this.setPromptAndSend(prompt);
      });
    });

    // Scroll to bottom button
    document.getElementById('scroll-to-bottom-btn')?.addEventListener('click', () => {
      this.scrollToBottom(true);
    });
  },

  /* ─── SCROLL ─── */
  initScrollListener() {
    const area = document.getElementById('chat-messages-area');
    if (!area) return;
    area.addEventListener('scroll', () => {
      if (this.isChatNearBottom()) {
        this._hideScrollBtn();
      } else {
        this._showScrollBtn();
      }
    });
  },

  isChatNearBottom() {
    const el = document.getElementById('chat-messages-area');
    if (!el) return true;
    return (el.scrollHeight - el.scrollTop - el.clientHeight) < 120;
  },

  scrollToBottom(force = false) {
    const el = document.getElementById('chat-messages-area');
    if (!el) return;
    if (force || this.isChatNearBottom()) {
      el.scrollTo({ top: el.scrollHeight, behavior: 'smooth' });
      this._hideScrollBtn();
    } else {
      this._showScrollBtn();
    }
  },

  _showScrollBtn() {
    const btn = document.getElementById('scroll-to-bottom-btn');
    btn?.classList.add('visible');
  },

  _hideScrollBtn() {
    const btn = document.getElementById('scroll-to-bottom-btn');
    btn?.classList.remove('visible');
  },

  /* ─── NEW CHAT ─── */
  newChat() {
    this.currentConversationId = null;
    AppState.activeConversationId = null;
    this.messages = [];
    this.lastUserPrompt = null;
    this.attachedChatFiles = [];

    // Show welcome screen
    const welcome = document.getElementById('welcome-screen');
    const msgs = document.getElementById('messages-container');
    if (welcome) welcome.style.display = '';
    if (msgs) msgs.innerHTML = '';

    // Reset header title
    this._setHeaderTitle('VOLTIX', null);

    this.loadConversations();
    this._clearAttachmentChips();
    this.updateContextDrawer();
    showToast('New chat', 'info', 1500);
  },

  /* ─── LOAD CONVERSATION LIST (sidebar) ─── */
  async loadConversations() {
    const container = document.getElementById('conversation-list');
    if (!container) return;

    try {
      const convs = await VoltixAPI.getConversations(AppState.activeProjectId);
      AppState.conversations = convs || [];

      let filtered = AppState.conversations;
      if (this.searchQuery) {
        filtered = filtered.filter(c => c.title.toLowerCase().includes(this.searchQuery));
      }

      if (filtered.length === 0) {
        container.innerHTML = `<div style="font-size:12px;color:var(--text-3);padding:6px 4px 2px;">${this.searchQuery ? 'No results.' : 'No chats yet.'}</div>`;
        return;
      }

      // Group by date
      const groups = this._groupByDate(filtered);
      let html = '';
      for (const [label, items] of Object.entries(groups)) {
        if (items.length === 0) continue;
        html += `<div class="sidebar-date-group">${label}</div>`;
        html += items.map(c => {
          const isActive = c.id === this.currentConversationId;
          const safetitle = this._escapeAttr(c.title);
          return `
            <div class="sidebar-item ${isActive ? 'active' : ''}" data-id="${c.id}" title="${safetitle}">
              <span class="sidebar-item-text" onclick="ChatModule.loadConversation('${c.id}')">${this._escapeHtml(c.title)}</span>
              <div class="sidebar-conv-actions">
                <button class="sidebar-conv-action-btn" onclick="event.stopPropagation();ChatModule.renameConversation('${c.id}','${safetitle}')" title="Rename">
                  <i class="fa-solid fa-pen" style="font-size:9px;"></i>
                </button>
                <button class="sidebar-conv-action-btn danger" onclick="event.stopPropagation();ChatModule.confirmDelete('${c.id}')" title="Delete">
                  <i class="fa-solid fa-trash-can" style="font-size:9px;"></i>
                </button>
              </div>
            </div>
          `;
        }).join('');
      }
      container.innerHTML = html;
    } catch (err) {
      console.warn('Failed to load conversations:', err);
    }
  },

  _groupByDate(convs) {
    const now = new Date();
    const today     = new Date(now.getFullYear(), now.getMonth(), now.getDate());
    const yesterday = new Date(today - 86400000);
    const lastWeek  = new Date(today - 7 * 86400000);
    const lastMonth = new Date(today - 30 * 86400000);

    const groups = { Today: [], Yesterday: [], 'Previous 7 days': [], 'Previous 30 days': [], Older: [] };
    convs.forEach(c => {
      const d = new Date(c.created_at || c.updated_at || 0);
      if (d >= today)     groups.Today.push(c);
      else if (d >= yesterday) groups.Yesterday.push(c);
      else if (d >= lastWeek)  groups['Previous 7 days'].push(c);
      else if (d >= lastMonth) groups['Previous 30 days'].push(c);
      else                     groups.Older.push(c);
    });
    return groups;
  },

  /* ─── LOAD CONVERSATION ─── */
  async loadConversation(convId) {
    this.currentConversationId = convId;
    AppState.activeConversationId = convId;

    const container = document.getElementById('messages-container');
    const welcome   = document.getElementById('welcome-screen');
    if (!container) return;

    // Hide welcome screen
    if (welcome) welcome.style.display = 'none';
    container.innerHTML = '';
    this.messages = [];

    try {
      const detail = await VoltixAPI.getConversationDetail(convId);
      if (detail?.messages) {
        this.messages = detail.messages;
        detail.messages.forEach(msg => {
          this.renderMessageBubble(msg.sender, msg.content, msg.metadata);
        });
        this._setHeaderTitle(detail.title || 'Conversation', AppState.activeProjectName);
      }
    } catch (err) {
      showToast('Failed to load chat history', 'error');
    }

    this.loadConversations();
    this.updateContextDrawer();
    this.scrollToBottom(true);
  },

  /* ─── RENAME ─── */
  async renameConversation(convId, oldTitle) {
    const newTitle = prompt('Rename conversation:', oldTitle);
    if (newTitle?.trim()) {
      await VoltixAPI.renameConversation(convId, newTitle.trim());
      if (convId === this.currentConversationId) {
        this._setHeaderTitle(newTitle.trim(), AppState.activeProjectName);
      }
      this.loadConversations();
      showToast('Renamed', 'success', 1500);
    }
  },

  /* ─── DELETE ─── */
  confirmDelete(convId) {
    openModal('modal-delete-confirm');
    document.getElementById('btn-confirm-delete').onclick = async () => {
      closeModal('modal-delete-confirm');
      await this.deleteConversation(convId);
    };
  },

  async deleteConversation(convId) {
    try {
      await VoltixAPI.deleteConversation(convId);
      if (this.currentConversationId === convId) this.newChat();
      else this.loadConversations();
      showToast('Conversation deleted', 'info');
    } catch (err) {
      showToast('Delete failed', 'error');
    }
  },

  /* ─── SEND MESSAGE ─── */
  async sendMessage() {
    const input = document.getElementById('chat-input');
    if (!input) return;

    const text = input.value.trim();
    if (!text || this.isGenerating) return;

    // Clear input, reset height
    input.value = '';
    input.style.height = 'auto';
    const sendBtn = document.getElementById('chat-send-btn');
    if (sendBtn) sendBtn.disabled = true;

    this.lastUserPrompt = text;

    // Hide welcome, show messages area
    const welcome = document.getElementById('welcome-screen');
    if (welcome) welcome.style.display = 'none';

    // Render user bubble
    this.renderMessageBubble('user', text);
    this.messages.push({ sender: 'user', content: text });

    // Create assistant streaming bubble
    const assistantWrapper = this._createAssistantBubble();
    const contentDiv  = assistantWrapper.querySelector('.message-content');
    const thoughtBox  = assistantWrapper.querySelector('.thought-box');
    const thoughtBody = assistantWrapper.querySelector('.thought-body');
    let   thoughtTitle = assistantWrapper.querySelector('.thought-summary-label');

    this._setGenerating(true);
    this.activeAbortController = new AbortController();

    let fullResponse = '';
    let citations    = [];
    let thoughtCount = 0;

    try {
      const response = await fetch('/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: this.activeAbortController.signal,
        body: JSON.stringify({
          prompt: text,
          conversation_id: this.currentConversationId,
          model: AppState.activeModel,
          academic_mode: AppState.activeMode,
          project_id: AppState.activeProjectId,
          enable_rag: AppState.ragEnabled !== false,
          temperature: AppState.temperature,
          top_p: AppState.topP
        })
      });

      const reader  = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          const jsonStr = line.slice(6).trim();
          if (!jsonStr) continue;

          try {
            const data = JSON.parse(jsonStr);

            if (data.type === 'meta') {
              if (data.conversation_id) {
                this.currentConversationId = data.conversation_id;
                AppState.activeConversationId = data.conversation_id;
                this.loadConversations();
              }
              citations = data.citations || [];

            } else if (data.type === 'thought') {
              thoughtCount++;
              if (thoughtBox) thoughtBox.classList.remove('hidden');
              if (thoughtBody) {
                thoughtBody.innerHTML += `
                  <div style="display:flex;align-items:flex-start;gap:6px;padding:3px 0;border-bottom:1px solid var(--border);margin-bottom:2px;">
                    <i class="fa-regular fa-lightbulb" style="color:#f59e0b;margin-top:2px;font-size:10px;flex-shrink:0;"></i>
                    <span style="color:#fcd34d;">${this._escapeHtml(data.content)}</span>
                  </div>`;
              }
              if (thoughtTitle) thoughtTitle.textContent = `Reasoning (${thoughtCount} steps)`;
              this.scrollToBottom();

            } else if (data.type === 'tool_call') {
              thoughtCount++;
              if (thoughtBox) thoughtBox.classList.remove('hidden');
              if (thoughtBody) {
                thoughtBody.innerHTML += `
                  <div class="tool-call-row">
                    <i class="fa-solid fa-wrench" style="color:var(--accent);font-size:10px;flex-shrink:0;"></i>
                    <span class="tool-call-name">${this._escapeHtml(data.name)}</span>
                    <span class="tool-call-status running">Running…</span>
                  </div>`;
              }
              this.scrollToBottom();

            } else if (data.type === 'tool_result') {
              if (thoughtBody) {
                // Update last tool row's status
                const rows = thoughtBody.querySelectorAll('.tool-call-status.running');
                if (rows.length) {
                  const last = rows[rows.length - 1];
                  last.classList.remove('running');
                  last.classList.add('done');
                  last.textContent = '✓ Done';
                }
              }
              this.scrollToBottom();

            } else if (data.type === 'token') {
              if (fullResponse === '') {
                // First token — clear the "thinking" indicator
                contentDiv.innerHTML = '';
                if (thoughtTitle) thoughtTitle.textContent = `Reasoning (${thoughtCount} steps)`;
              }
              fullResponse += data.content;
              contentDiv.innerHTML = typeof marked !== 'undefined'
                ? marked.parse(fullResponse)
                : this._escapeHtml(fullResponse);
              this._postRender(contentDiv);
              this.scrollToBottom();

            } else if (data.type === 'done') {
              citations = data.citations || citations;
              if (citations.length > 0) {
                this._attachCitationBtn(contentDiv, citations);
              }
              this._attachMessageActions(assistantWrapper, fullResponse);
              this.messages.push({ sender: 'assistant', content: fullResponse, metadata: { citations } });
              logActivity(`Response to: "${text.slice(0, 40)}"`, 'success');
              this.updateContextDrawer();
            }

          } catch (parseErr) {
            console.warn('[VOLTIX] SSE parse error:', parseErr);
          }
        }
      }

    } catch (err) {
      if (err.name === 'AbortError') {
        contentDiv.innerHTML += '<div style="color:var(--text-3);font-size:13px;font-style:italic;margin-top:8px;display:flex;align-items:center;gap:6px;"><i class="fa-solid fa-ban"></i> Generation stopped.</div>';
      } else {
        contentDiv.innerHTML += `<div style="color:var(--danger);font-size:13px;margin-top:8px;"><i class="fa-solid fa-triangle-exclamation"></i> Error: ${err.message}</div>`;
      }
    } finally {
      this._setGenerating(false);
      this.activeAbortController = null;
    }
  },

  /* ─── STOP GENERATION ─── */
  stopGeneration() {
    if (this.activeAbortController) {
      this.activeAbortController.abort();
      this.activeAbortController = null;
      this._setGenerating(false);
      showToast('Generation stopped', 'info', 1600);
    }
  },

  /* ─── SET GENERATING STATE ─── */
  _setGenerating(generating) {
    this.isGenerating = generating;
    const sendBtn = document.getElementById('chat-send-btn');
    const stopBtn = document.getElementById('chat-stop-btn');
    if (sendBtn) sendBtn.classList.toggle('hidden', generating);
    if (stopBtn) stopBtn.classList.toggle('hidden', !generating);
  },

  /* ─── RENDER USER BUBBLE ─── */
  renderMessageBubble(sender, content, metadata = null) {
    const container = document.getElementById('messages-container');
    if (!container) return;

    const wrapper = document.createElement('div');
    wrapper.className = `message-wrapper ${sender === 'user' ? 'message-user' : 'message-assistant'}`;

    if (sender === 'user') {
      wrapper.innerHTML = `
        <div class="message-user-bubble">${this._escapeHtml(content)}</div>
        <div class="message-user-actions">
          <button class="msg-action-btn" onclick="ChatModule._copyText(${JSON.stringify(content)}, this)" title="Copy">
            <i class="fa-regular fa-copy" style="font-size:11px;"></i>
          </button>
        </div>`;
    } else {
      // Assistant bubble
      const hasThought = metadata?.thought_log?.length > 0;
      let thoughtHtml = '';
      if (hasThought) {
        const steps = metadata.thought_log.map(item => {
          if (item.type === 'thought') return `<div style="color:#fcd34d;padding:2px 0;font-size:12px;">• ${this._escapeHtml(item.content)}</div>`;
          if (item.type === 'tool_call') return `<div class="tool-call-row"><i class="fa-solid fa-wrench" style="color:var(--accent);font-size:10px;"></i><span class="tool-call-name">${this._escapeHtml(item.name)}</span><span class="tool-call-status done">Done</span></div>`;
          return '';
        }).join('');

        thoughtHtml = `
          <details class="thought-box">
            <summary class="thought-summary">
              <i class="fa-solid fa-brain" style="color:var(--accent);font-size:11px;"></i>
              <span class="thought-summary-label">Reasoning (${metadata.thought_log.length} steps)</span>
              <i class="fa-solid fa-chevron-down chevron"></i>
            </summary>
            <div class="thought-body">${steps}</div>
          </details>`;
      }

      const parsedContent = typeof marked !== 'undefined' ? marked.parse(content) : this._escapeHtml(content);
      const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

      wrapper.innerHTML = `
        <div class="message-assistant-header">
          <div class="message-assistant-logo"><i class="fa-solid fa-bolt" style="font-size:11px;"></i></div>
          <span class="message-assistant-name">VOLTIX</span>
          <span class="message-assistant-time">${time}</span>
        </div>
        <div class="message-assistant-content">
          ${thoughtHtml}
          <div class="markdown-body message-content">${parsedContent}</div>
        </div>
        <div class="message-assistant-actions"></div>`;

      // Post-render: math + code highlighting
      const contentEl = wrapper.querySelector('.message-content');
      if (contentEl) {
        this._postRender(contentEl);
        if (metadata?.citations?.length > 0) {
          this._attachCitationBtn(contentEl, metadata.citations);
        }
      }
      this._attachMessageActions(wrapper, content);
    }

    container.appendChild(wrapper);
    this.scrollToBottom();
    return wrapper;
  },

  /* ─── CREATE STREAMING ASSISTANT BUBBLE ─── */
  _createAssistantBubble() {
    const container = document.getElementById('messages-container');
    const wrapper = document.createElement('div');
    wrapper.className = 'message-wrapper message-assistant';
    const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    wrapper.innerHTML = `
      <div class="message-assistant-header">
        <div class="message-assistant-logo"><i class="fa-solid fa-bolt" style="font-size:11px;"></i></div>
        <span class="message-assistant-name">VOLTIX</span>
        <span class="message-assistant-time">${time}</span>
      </div>
      <div class="message-assistant-content">
        <details class="thought-box hidden">
          <summary class="thought-summary">
            <i class="fa-solid fa-brain" style="color:var(--accent);font-size:11px;"></i>
            <span class="thought-summary-label">Reasoning…</span>
            <i class="fa-solid fa-chevron-down chevron"></i>
          </summary>
          <div class="thought-body"></div>
        </details>
        <div class="markdown-body message-content thinking-indicator">
          <span class="thinking-dots">
            <span></span><span></span><span></span>
          </span>
          <span>Thinking…</span>
        </div>
      </div>
      <div class="message-assistant-actions"></div>`;

    container.appendChild(wrapper);
    this.scrollToBottom();
    return wrapper;
  },

  /* ─── POST-RENDER (math + code) ─── */
  _postRender(el) {
    if (!el) return;
    // KaTeX auto-render
    if (typeof renderMathInElement !== 'undefined') {
      try {
        renderMathInElement(el, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$',  right: '$',  display: false },
            { left: '\\[', right: '\\]', display: true },
            { left: '\\(', right: '\\)', display: false }
          ],
          throwOnError: false
        });
      } catch (e) {}
    }
    // VoltixMath compat
    if (typeof VoltixMath?.render === 'function') VoltixMath.render(el);

    // Code highlighting + copy buttons
    el.querySelectorAll('pre code').forEach(block => {
      if (typeof hljs !== 'undefined') hljs.highlightElement(block);
      // Inject copy button into parent pre
      const pre = block.parentElement;
      if (pre && !pre.querySelector('.code-block-header')) {
        const lang = block.className.replace(/language-/,'').split(' ')[0] || 'code';
        const header = document.createElement('div');
        header.className = 'code-block-header';
        header.innerHTML = `
          <span class="code-block-lang">${lang}</span>
          <button class="code-copy-btn">
            <i class="fa-regular fa-copy"></i> Copy
          </button>`;
        header.querySelector('.code-copy-btn').addEventListener('click', () => {
          copyToClipboard(block.textContent);
          header.querySelector('.code-copy-btn').innerHTML = '<i class="fa-solid fa-check" style="color:var(--success);"></i> Copied!';
          setTimeout(() => {
            header.querySelector('.code-copy-btn').innerHTML = '<i class="fa-regular fa-copy"></i> Copy';
          }, 2000);
        });
        pre.insertBefore(header, pre.firstChild);
      }
    });

    // VoltixCode compat
    if (typeof VoltixCode?.highlight === 'function') VoltixCode.highlight(el);
  },

  /* ─── CITATION BUTTON ─── */
  _attachCitationBtn(element, citations) {
    if (!element || element.querySelector('.citation-btn')) return;
    const btn = document.createElement('button');
    btn.className = 'citation-btn';
    btn.innerHTML = `<i class="fa-solid fa-book-open" style="font-size:11px;"></i> ${citations.length} source${citations.length > 1 ? 's' : ''}`;
    btn.addEventListener('click', () => {
      const modal = document.getElementById('modal-sources');
      const body  = document.getElementById('sources-modal-body');
      if (modal && body) {
        body.innerHTML = citations.map(c => `
          <div style="margin-bottom:12px;padding:12px;border-radius:10px;background:var(--bg);border:1px solid var(--border);">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
              <span style="font-weight:600;color:var(--accent);font-size:13px;">[${c.id}] ${this._escapeHtml(c.document)}</span>
              <span style="font-size:11px;color:var(--text-3);font-family:monospace;">Page ${c.page} · Score ${c.score}</span>
            </div>
            <p style="font-size:12.5px;color:var(--text-2);font-style:italic;font-family:monospace;background:rgba(0,0,0,0.2);padding:8px;border-radius:6px;line-height:1.5;">${this._escapeHtml(c.snippet)}</p>
          </div>
        `).join('');
        openModal('modal-sources');
      }
    });
    element.appendChild(btn);
  },

  /* ─── MESSAGE ACTION BUTTONS ─── */
  _attachMessageActions(wrapperEl, text) {
    const bar = wrapperEl.querySelector('.message-assistant-actions');
    if (!bar || bar.children.length > 0) return;

    bar.innerHTML = `
      <button class="msg-action-btn" title="Copy response">
        <i class="fa-regular fa-copy" style="font-size:11px;"></i> Copy
      </button>
      <button class="msg-action-btn" title="Regenerate response" style="margin-left:2px;">
        <i class="fa-solid fa-arrows-rotate" style="font-size:11px;"></i> Regenerate
      </button>`;

    bar.querySelector('.msg-action-btn:first-child').addEventListener('click', function() {
      copyToClipboard(text);
      showToast('Copied to clipboard', 'success', 1800);
      this.innerHTML = '<i class="fa-solid fa-check" style="font-size:11px;color:var(--success);"></i> Copied';
      this.classList.add('copied');
      setTimeout(() => {
        this.innerHTML = '<i class="fa-regular fa-copy" style="font-size:11px;"></i> Copy';
        this.classList.remove('copied');
      }, 2000);
    });

    bar.querySelector('.msg-action-btn:last-child').addEventListener('click', () => {
      if (this.lastUserPrompt) {
        const input = document.getElementById('chat-input');
        if (input) {
          input.value = this.lastUserPrompt;
          input.dispatchEvent(new Event('input'));
          this.sendMessage();
        }
      }
    });
  },

  /* ─── COPY TEXT (from user bubble) ─── */
  _copyText(text, btn) {
    copyToClipboard(text);
    showToast('Copied', 'success', 1600);
  },

  /* ─── FILE ATTACHMENTS ─── */
  async handleChatFileAttachment(file) {
    showToast(`Indexing '${file.name}'…`, 'info', 3000);
    try {
      const res = await VoltixAPI.uploadDocument(file, AppState.activeProjectId);
      if (res.document) {
        this.attachedChatFiles.push({
          filename: file.name,
          chunks: res.document.total_chunks,
          size_bytes: res.document.file_size_bytes
        });
        this._renderAttachmentChips();
        this.updateContextDrawer();
        showToast(`Indexed '${file.name}' (${res.document.total_chunks} chunks)`, 'success');
      } else {
        showToast(res.error || 'Upload failed', 'error');
      }
    } catch (err) {
      showToast(`Upload error: ${err.message}`, 'error');
    }
  },

  removeAttachedFile(idx) {
    this.attachedChatFiles.splice(idx, 1);
    this._renderAttachmentChips();
    this.updateContextDrawer();
  },

  _renderAttachmentChips() {
    const container = document.getElementById('composer-attachments');
    if (!container) return;

    if (this.attachedChatFiles.length === 0) {
      container.classList.remove('has-files');
      container.innerHTML = '';
      return;
    }

    container.classList.add('has-files');
    container.innerHTML = this.attachedChatFiles.map((f, idx) => `
      <div class="attachment-chip">
        <i class="fa-solid fa-file" style="font-size:11px;color:var(--accent);"></i>
        <span>${this._escapeHtml(f.filename)}</span>
        <span style="font-size:11px;color:var(--text-3);">(${f.chunks} chunks)</span>
        <button class="remove-chip" onclick="ChatModule.removeAttachedFile(${idx})" title="Remove attachment">×</button>
      </div>
    `).join('');
  },

  _clearAttachmentChips() {
    this.attachedChatFiles = [];
    this._renderAttachmentChips();
  },

  /* ─── CONTEXT DRAWER UPDATE ─── */
  async updateContextDrawer() {
    const elModel    = document.getElementById('ctx-model');
    const elProject  = document.getElementById('ctx-project');
    const elKnow     = document.getElementById('ctx-knowledge');
    const elMode     = document.getElementById('ctx-mode');
    const elTokens   = document.getElementById('ctx-tokens');
    const elFiles    = document.getElementById('ctx-files-list');

    if (elModel)   elModel.textContent   = AppState.activeModel || 'Auto';
    if (elProject) elProject.textContent = AppState.activeProjectName || 'Default Workspace';
    if (elMode)    elMode.textContent    = AppState.activeMode || 'Learn';

    // Token estimate
    let totalChars = 0;
    this.messages.forEach(m => totalChars += (m.content || '').length);
    const estTokens = Math.round(totalChars / 4);
    if (elTokens) elTokens.textContent = `~${estTokens.toLocaleString()} / 32k`;

    // Knowledge files
    try {
      const docs = await VoltixAPI.getDocuments(AppState.activeProjectId);
      if (elKnow) elKnow.textContent = `${docs.length} document${docs.length !== 1 ? 's' : ''}`;
      if (elFiles) {
        elFiles.innerHTML = docs.length === 0
          ? '<div style="color:var(--text-3);font-size:12px;">No files indexed.</div>'
          : docs.map(d => `
            <div style="display:flex;align-items:center;justify-content:space-between;padding:5px 0;border-bottom:1px solid var(--border);">
              <span style="font-family:monospace;font-size:12px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;flex:1;">${this._escapeHtml(d.filename)}</span>
              <span style="font-size:11px;color:var(--accent);font-family:monospace;margin-left:8px;">${d.total_chunks}c</span>
            </div>`).join('');
      }
    } catch (e) {}
  },

  /* ─── EXPORT ─── */
  exportConversation(format = 'markdown') {
    if (!this.currentConversationId) {
      showToast('Start or select a conversation first', 'warning');
      return;
    }
    window.location.href = `/api/conversations/${this.currentConversationId}/export?format=${format}`;
    showToast('Exporting…', 'success', 2000);
  },

  /* ─── HEADER TITLE ─── */
  _setHeaderTitle(title, project) {
    const titleEl   = document.getElementById('header-conv-title');
    const projectEl = document.getElementById('header-project-badge');
    if (titleEl)   titleEl.textContent   = title;
    if (projectEl) {
      if (project && project !== 'Default Workspace') {
        projectEl.textContent = `· ${project}`;
        projectEl.classList.remove('hidden');
      } else {
        projectEl.classList.add('hidden');
      }
    }
  },

  /* ─── UTILITY: set prompt and send ─── */
  setPromptAndSend(prompt) {
    const input = document.getElementById('chat-input');
    if (input) {
      input.value = prompt;
      input.dispatchEvent(new Event('input'));
      this.sendMessage();
    }
  },

  /* ─── LEGACY COMPAT ─── */
  onShow() {
    this.loadConversations();
    this.updateContextDrawer();
    setTimeout(() => this.scrollToBottom(true), 80);
  },

  /* ─── PRIVATE: HTML escaping ─── */
  _escapeHtml(str) {
    if (typeof str !== 'string') return '';
    return str.replace(/[&<>"']/g,
      t => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[t] || t)
    );
  },

  _escapeAttr(str) {
    return (str || '').replace(/'/g, '&#39;').replace(/"/g, '&quot;');
  },

  // legacy alias
  escapeHtml(str) { return this._escapeHtml(str); }
};
