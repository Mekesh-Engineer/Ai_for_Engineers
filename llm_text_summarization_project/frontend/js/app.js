/**
 * Local LLM Studio - App Core State, Navigation, and Global Controller
 */

const AppState = {
  activeTab: 'dashboard',
  backendConnected: false,
  ollamaConnected: false,
  localModelAvailable: false,
  activeMode: 'ollama',
  activeOllamaModel: 'qwen2.5:7b',
  activeLocalModelPath: '',
  activeProfile: 'general',
  selectedProjectFiles: [],
  conversations: [],
  currentConversationId: null,
  sidebarCollapsed: false,
  recentActivities: []
};

// Theme Management Engine
function initTheme() {
  const saved = localStorage.getItem('studio_theme') || 'dark';
  if (saved === 'light') {
    document.documentElement.classList.remove('dark');
  } else {
    document.documentElement.classList.add('dark');
  }
}

function toggleTheme() {
  const isDark = document.documentElement.classList.contains('dark');
  if (isDark) {
    document.documentElement.classList.remove('dark');
    localStorage.setItem('studio_theme', 'light');
    showToast('Switched to Precision Light Theme', 'info', 2000);
  } else {
    document.documentElement.classList.add('dark');
    localStorage.setItem('studio_theme', 'dark');
    showToast('Switched to Obsidian Dark Theme', 'info', 2000);
  }
}

// Toast Notification Helper
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  const styles = {
    info: 'bg-[var(--color-surface)] border-indigo-500/40 text-[var(--color-text-primary)]',
    success: 'bg-[var(--color-surface)] border-emerald-500/40 text-[var(--color-text-primary)]',
    warning: 'bg-[var(--color-surface)] border-amber-500/40 text-[var(--color-text-primary)]',
    error: 'bg-[var(--color-surface)] border-rose-500/40 text-[var(--color-text-primary)]'
  };

  const icons = {
    info: '<svg class="w-4 h-4 text-indigo-400 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"></path></svg>',
    success: '<svg class="w-4 h-4 text-emerald-400 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>',
    warning: '<svg class="w-4 h-4 text-amber-400 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path></svg>',
    error: '<svg class="w-4 h-4 text-rose-400 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path></svg>'
  };

  toast.className = `flex items-center px-3 py-2 rounded-lg border shadow-lg transition-all duration-200 transform translate-y-2 opacity-0 text-xs font-medium ${styles[type] || styles.info}`;
  toast.innerHTML = `${icons[type] || icons.info}<span class="leading-snug">${message}</span>`;

  container.appendChild(toast);
  requestAnimationFrame(() => {
    toast.classList.remove('translate-y-2', 'opacity-0');
  });

  setTimeout(() => {
    toast.classList.add('opacity-0', '-translate-y-2');
    setTimeout(() => toast.remove(), 250);
  }, duration);
}

// Robust Clipboard Copy Helper with ExecCommand Fallback
async function copyToClipboard(text) {
  if (!text) return false;
  if (navigator.clipboard && navigator.clipboard.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch (err) {
      console.warn('Navigator clipboard failed, falling back:', err);
    }
  }
  
  // Fallback using temporary textarea
  try {
    const textArea = document.createElement('textarea');
    textArea.value = text;
    textArea.style.position = 'fixed';
    textArea.style.left = '-9999px';
    textArea.style.top = '0';
    document.body.appendChild(textArea);
    textArea.focus();
    textArea.select();
    const successful = document.execCommand('copy');
    document.body.removeChild(textArea);
    return successful;
  } catch (err) {
    console.error('Copy fallback failed:', err);
    return false;
  }
}

// Activity Feed Logger
function logActivity(text, type = 'info') {
  const item = {
    text,
    type,
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
  };
  AppState.recentActivities.unshift(item);
  if (AppState.recentActivities.length > 8) AppState.recentActivities.pop();
  renderActivityFeed();
}

function renderActivityFeed() {
  const container = document.getElementById('dashboard-activity-feed');
  if (!container) return;

  if (AppState.recentActivities.length === 0) {
    container.innerHTML = '<div class="text-xs text-[var(--color-text-muted)] py-2 italic">No recent activity yet. Start chatting or analyzing files.</div>';
    return;
  }

  container.innerHTML = AppState.recentActivities.map(a => `
    <div class="flex items-center justify-between text-xs py-1.5 border-b border-[var(--color-border)] last:border-0">
      <div class="flex items-center space-x-2 truncate">
        <span class="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
        <span class="text-[var(--color-text-primary)] truncate">${a.text}</span>
      </div>
      <span class="text-[10px] text-[var(--color-text-muted)] font-mono flex-shrink-0 ml-2">${a.time}</span>
    </div>
  `).join('');
}

// Sidebar Collapse / Expand Toggle
function toggleSidebar() {
  const sidebar = document.getElementById('app-sidebar');
  if (!sidebar) return;

  if (window.innerWidth < 1024) {
    const isClosed = sidebar.classList.contains('-translate-x-full');
    if (isClosed) {
      openMobileSidebar();
    } else {
      closeMobileSidebar();
    }
    return;
  }

  AppState.sidebarCollapsed = !AppState.sidebarCollapsed;
  sidebar.classList.toggle('sidebar-collapsed', AppState.sidebarCollapsed);
  localStorage.setItem('studio_sidebar_collapsed', AppState.sidebarCollapsed ? 'true' : 'false');
}

function initSidebarState() {
  const saved = localStorage.getItem('studio_sidebar_collapsed');
  if (saved === 'true') {
    AppState.sidebarCollapsed = true;
    const sidebar = document.getElementById('app-sidebar');
    if (sidebar) sidebar.classList.add('sidebar-collapsed');
  }
}

// Navigation & Tab Switching
function switchTab(tabId) {
  AppState.activeTab = tabId;

  // Update Nav Button active styling
  document.querySelectorAll('.nav-btn').forEach(btn => {
    const target = btn.getAttribute('data-tab');
    if (target === tabId) {
      btn.classList.add('bg-indigo-500/10', 'text-indigo-400', 'border-indigo-500/30');
      btn.classList.remove('text-[var(--color-text-secondary)]', 'hover:bg-[var(--color-elevated)]', 'hover:text-[var(--color-text-primary)]');
    } else {
      btn.classList.remove('bg-indigo-500/10', 'text-indigo-400', 'border-indigo-500/30');
      btn.classList.add('text-[var(--color-text-secondary)]', 'hover:bg-[var(--color-elevated)]', 'hover:text-[var(--color-text-primary)]');
    }
  });

  // Reveal Active Tab Panel
  document.querySelectorAll('.tab-panel').forEach(panel => {
    if (panel.id === `tab-${tabId}`) {
      panel.classList.remove('hidden');
    } else {
      panel.classList.add('hidden');
    }
  });

  // Module Lifecycle Hooks
  if (tabId === 'chat' && window.ChatModule) {
    window.ChatModule.onShow();
  } else if (tabId === 'project' && window.ProjectModule) {
    window.ProjectModule.onShow();
  } else if (tabId === 'settings' && window.SettingsModule) {
    window.SettingsModule.onShow();
  }

  closeMobileSidebar();
}

function openMobileSidebar() {
  const drawer = document.getElementById('app-sidebar');
  const backdrop = document.getElementById('mobile-sidebar-backdrop');
  if (drawer) drawer.classList.remove('hidden', '-translate-x-full');
  if (backdrop) backdrop.classList.remove('hidden');
}

function closeMobileSidebar() {
  const drawer = document.getElementById('app-sidebar');
  const backdrop = document.getElementById('mobile-sidebar-backdrop');
  if (window.innerWidth < 1024) {
    if (drawer) drawer.classList.add('-translate-x-full');
    if (backdrop) backdrop.classList.add('hidden');
  }
}

// System Health Verification
async function checkSystemHealth() {
  try {
    const res = await fetch('/api/health');
    if (res.ok) {
      const data = await res.json();
      AppState.backendConnected = true;
      AppState.ollamaConnected = data.ollama_connected;
      AppState.localModelAvailable = data.local_model_available;
      AppState.activeMode = data.active_mode;
      AppState.activeOllamaModel = data.ollama_model;

      updateStatusBadges();
      updateModeSelectors();
    } else {
      AppState.backendConnected = false;
      updateStatusBadges();
    }
  } catch (err) {
    AppState.backendConnected = false;
    updateStatusBadges();
  }
}

function updateStatusBadges() {
  // Backend Status Badge
  const backendBadge = document.getElementById('badge-backend');
  if (backendBadge) {
    backendBadge.innerHTML = AppState.backendConnected
      ? `<span class="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5"></span>Backend`
      : `<span class="w-1.5 h-1.5 rounded-full bg-rose-500 mr-1.5"></span>Backend`;
    backendBadge.className = `flex items-center text-[11px] font-medium px-2.5 py-1 rounded-md border cursor-pointer ${AppState.backendConnected ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-rose-500/10 text-rose-400 border-rose-500/30'}`;
  }

  // Ollama Status Badge
  const ollamaBadge = document.getElementById('badge-ollama');
  if (ollamaBadge) {
    ollamaBadge.innerHTML = AppState.ollamaConnected
      ? `<span class="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5"></span>Ollama (${AppState.activeOllamaModel})`
      : `<span class="w-1.5 h-1.5 rounded-full bg-amber-500 mr-1.5"></span>Ollama Offline`;
    ollamaBadge.className = `flex items-center text-[11px] font-medium px-2.5 py-1 rounded-md border cursor-pointer ${AppState.ollamaConnected ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30' : 'bg-amber-500/10 text-amber-400 border-amber-500/30'}`;
  }

  // Local Model Status Badge
  const localBadge = document.getElementById('badge-local-model');
  if (localBadge) {
    localBadge.innerHTML = AppState.localModelAvailable
      ? `<span class="w-1.5 h-1.5 rounded-full bg-indigo-400 mr-1.5"></span>Local Model: Ready`
      : `<span class="w-1.5 h-1.5 rounded-full bg-slate-500 mr-1.5"></span>Local Model: None`;
    localBadge.className = `hidden md:flex items-center text-[11px] font-medium px-2.5 py-1 rounded-md border cursor-pointer ${AppState.localModelAvailable ? 'bg-indigo-500/10 text-indigo-400 border-indigo-500/30' : 'bg-[var(--color-elevated)] text-[var(--color-text-muted)] border-[var(--color-border)]'}`;
  }

  // Active Mode Indicator
  const activeModePill = document.getElementById('badge-active-mode');
  if (activeModePill) {
    activeModePill.innerText = AppState.activeMode === 'ollama' ? `Mode: Ollama (${AppState.activeOllamaModel})` : `Mode: Local Model (Hugging Face)`;
  }
}

function updateModeSelectors() {
  const selects = document.querySelectorAll('.mode-selector-select');
  selects.forEach(select => {
    select.value = AppState.activeMode;
  });
}

// Mode Switch Handler
async function handleModeSwitch(newMode) {
  try {
    const res = await fetch('/api/models/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: newMode })
    });
    if (res.ok) {
      const data = await res.json();
      AppState.activeMode = data.active_mode;
      updateStatusBadges();
      updateModeSelectors();
      logActivity(`Switched active AI engine to ${newMode.toUpperCase()}`);
      showToast(`Switched active AI backend to ${newMode.toUpperCase()} mode.`, 'success');
      if (window.ChatModule) window.ChatModule.updateHeader();
    } else {
      showToast('Failed to switch model mode.', 'error');
    }
  } catch (err) {
    showToast(`Error switching mode: ${err.message}`, 'error');
  }
}

// System Connectivity Diagnostics
async function runSystemDiagnostics() {
  const modal = document.getElementById('diagnostics-modal');
  const resultsContainer = document.getElementById('diagnostics-results');
  if (modal) modal.classList.remove('hidden');
  if (resultsContainer) {
    resultsContainer.innerHTML = `<div class="flex items-center justify-center p-6"><div class="animate-spin rounded-full h-6 w-6 border-b-2 border-indigo-500"></div><span class="ml-2.5 text-xs text-[var(--color-text-secondary)]">Running diagnostic checks...</span></div>`;
  }

  try {
    const [ollamaRes, localRes] = await Promise.all([
      fetch('/api/models/test-ollama', { method: 'POST' }).then(r => r.json()),
      fetch('/api/models/test-local', { method: 'POST' }).then(r => r.json())
    ]);

    let html = `
      <div class="space-y-3 text-xs">
        <div class="p-3 rounded-lg border ${ollamaRes.test_generation ? 'bg-emerald-500/10 border-emerald-500/30' : 'bg-amber-500/10 border-amber-500/30'}">
          <div class="flex items-center justify-between font-semibold ${ollamaRes.test_generation ? 'text-emerald-400' : 'text-amber-400'}">
            <span>Ollama Backend (Qwen 2.5 7B)</span>
            <span class="font-mono text-[11px]">${ollamaRes.latency_seconds || 0}s</span>
          </div>
          <p class="text-[11px] text-[var(--color-text-secondary)] mt-1">${ollamaRes.message}</p>
          <div class="mt-2 grid grid-cols-2 gap-2 text-[11px] text-[var(--color-text-muted)]">
            <div>Reachable: <span class="${ollamaRes.ollama_reachable ? 'text-emerald-400' : 'text-rose-400'} font-medium">${ollamaRes.ollama_reachable ? 'Yes' : 'No'}</span></div>
            <div>Model Present: <span class="${ollamaRes.model_present ? 'text-emerald-400' : 'text-rose-400'} font-medium">${ollamaRes.model_present ? 'Yes' : 'No'}</span></div>
          </div>
        </div>

        <div class="p-3 rounded-lg border ${localRes.model_valid ? 'bg-indigo-500/10 border-indigo-500/30' : 'bg-[var(--color-canvas)] border-[var(--color-border)]'}">
          <div class="flex items-center justify-between font-semibold ${localRes.model_valid ? 'text-indigo-400' : 'text-[var(--color-text-muted)]'}">
            <span>Project Local Model (./models/)</span>
            <span class="font-mono text-[11px]">${localRes.latency_seconds || 0}s</span>
          </div>
          <p class="text-[11px] text-[var(--color-text-secondary)] mt-1">${localRes.message}</p>
          <div class="mt-2 grid grid-cols-2 gap-2 text-[11px] text-[var(--color-text-muted)]">
            <div>Detected: <span class="${localRes.model_detected ? 'text-emerald-400' : 'text-[var(--color-text-muted)]'} font-medium">${localRes.model_detected ? 'Yes' : 'No'}</span></div>
            <div>Verified: <span class="${localRes.test_generation ? 'text-emerald-400' : 'text-[var(--color-text-muted)]'} font-medium">${localRes.test_generation ? 'Yes' : 'No'}</span></div>
          </div>
        </div>
      </div>
    `;
    if (resultsContainer) resultsContainer.innerHTML = html;
  } catch (err) {
    if (resultsContainer) {
      resultsContainer.innerHTML = `<div class="p-3 bg-rose-500/10 border border-rose-500/30 text-rose-400 rounded-lg text-xs">Diagnostic error: ${err.message}</div>`;
    }
  }
}

// Command Palette Controller
const CommandPalette = {
  isOpen: false,
  commands: [
    { title: 'New AI Chat', category: 'Chat', action: () => { switchTab('chat'); if (window.ChatModule) window.ChatModule.newChat(); } },
    { title: 'Upload & Summarize File', category: 'Files', action: () => { switchTab('files'); document.getElementById('file-input')?.click(); } },
    { title: 'Switch to Ollama Mode (Qwen 2.5 7B)', category: 'Engine', action: () => handleModeSwitch('ollama') },
    { title: 'Switch to Project Local Model', category: 'Engine', action: () => handleModeSwitch('local_model') },
    { title: 'Audit Project Architecture', category: 'Project', action: () => { switchTab('project'); if (window.ProjectModule) window.ProjectModule.runAnalysis(); } },
    { title: 'Open AI Debugging Assistant', category: 'Project', action: () => { switchTab('project'); document.getElementById('debug-error-msg')?.focus(); } },
    { title: 'Run System Diagnostics', category: 'System', action: () => runSystemDiagnostics() },
    { title: 'Models & Settings Configuration', category: 'System', action: () => switchTab('settings') },
    { title: 'Toggle Light/Dark Theme', category: 'View', action: () => toggleTheme() },
    { title: 'Toggle Sidebar', category: 'View', action: () => toggleSidebar() }
  ],
  selectedIndex: 0,

  open() {
    const modal = document.getElementById('command-palette-modal');
    const input = document.getElementById('command-palette-input');
    if (!modal) return;

    this.isOpen = true;
    modal.classList.remove('hidden');
    this.render(this.commands);
    if (input) {
      input.value = '';
      setTimeout(() => input.focus(), 50);
    }
  },

  close() {
    const modal = document.getElementById('command-palette-modal');
    if (modal) modal.classList.add('hidden');
    this.isOpen = false;
  },

  render(filtered) {
    const container = document.getElementById('command-palette-results');
    if (!container) return;

    if (filtered.length === 0) {
      container.innerHTML = '<div class="p-4 text-center text-xs text-[var(--color-text-muted)]">No matching commands.</div>';
      return;
    }

    container.innerHTML = filtered.map((cmd, idx) => `
      <div onclick="CommandPalette.execute(${idx})" class="command-palette-item flex items-center justify-between px-3 py-2 rounded-lg text-xs cursor-pointer transition ${idx === this.selectedIndex ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' : 'text-[var(--color-text-primary)] hover:bg-[var(--color-elevated)]'}">
        <span class="font-medium">${cmd.title}</span>
        <span class="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-[var(--color-canvas)] text-[var(--color-text-muted)] border border-[var(--color-border)]">${cmd.category}</span>
      </div>
    `).join('');
  },

  filter(query) {
    const q = query.toLowerCase().trim();
    const filtered = this.commands.filter(c => c.title.toLowerCase().includes(q) || c.category.toLowerCase().includes(q));
    this.selectedIndex = 0;
    this.render(filtered);
  },

  execute(idx) {
    const input = document.getElementById('command-palette-input')?.value || '';
    const q = input.toLowerCase().trim();
    const filtered = this.commands.filter(c => c.title.toLowerCase().includes(q) || c.category.toLowerCase().includes(q));
    const target = filtered[idx] || this.commands[0];
    if (target && target.action) {
      this.close();
      target.action();
    }
  }
};

// Global Keyboard Shortcuts
window.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    if (CommandPalette.isOpen) CommandPalette.close();
    else CommandPalette.open();
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'b') {
    e.preventDefault();
    toggleSidebar();
  }
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'n') {
    e.preventDefault();
    switchTab('chat');
    if (window.ChatModule) window.ChatModule.newChat();
  }
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'o') {
    e.preventDefault();
    handleModeSwitch('ollama');
  }
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === 'l') {
    e.preventDefault();
    handleModeSwitch('local_model');
  }
  if ((e.ctrlKey || e.metaKey) && !e.shiftKey && e.key.toLowerCase() === 'u') {
    e.preventDefault();
    switchTab('files');
    document.getElementById('file-input')?.click();
  }
  if (e.key === 'Escape') {
    closeModal('diagnostics-modal');
    closeModal('custom-profile-modal');
    closeModal('status-detail-modal');
    closeModal('keyboard-shortcuts-modal');
    CommandPalette.close();
    const drawer = document.getElementById('context-files-drawer');
    if (drawer) drawer.classList.add('hidden');
    closeMobileSidebar();
  }
});

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.classList.add('hidden');
}

// Global Drag & Drop Overlay Handler
function initGlobalDragAndDrop() {
  const overlay = document.getElementById('global-drag-overlay');
  if (!overlay) return;

  window.addEventListener('dragover', (e) => {
    e.preventDefault();
    overlay.classList.remove('hidden');
  });

  overlay.addEventListener('dragleave', (e) => {
    e.preventDefault();
    overlay.classList.add('hidden');
  });

  overlay.addEventListener('drop', (e) => {
    e.preventDefault();
    overlay.classList.add('hidden');
    if (e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      switchTab('files');
      if (window.FilesModule) window.FilesModule.handleFileUpload(file);
    }
  });
}

// Lifecycle Initialization
document.addEventListener('DOMContentLoaded', () => {
  if (typeof marked !== 'undefined' && marked.setOptions) {
    marked.setOptions({ breaks: true, gfm: true });
  }

  initTheme();
  initSidebarState();
  initGlobalDragAndDrop();

  // Bind status badges to diagnostics modal
  ['badge-backend', 'badge-ollama', 'badge-local-model'].forEach(id => {
    const badge = document.getElementById(id);
    if (badge) {
      badge.addEventListener('click', () => runSystemDiagnostics());
    }
  });

  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tabId = btn.getAttribute('data-tab');
      switchTab(tabId);
    });
  });

  const cmdInput = document.getElementById('command-palette-input');
  if (cmdInput) {
    cmdInput.addEventListener('input', (e) => CommandPalette.filter(e.target.value));
    cmdInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        CommandPalette.execute(CommandPalette.selectedIndex);
      }
    });
  }

  checkSystemHealth();
  setInterval(checkSystemHealth, 10000);
});