/**
 * Local LLM Studio (Grammar Correction & Text Rewriting) - App Core State & Controller
 */

const AppState = {
  activeTab: 'dashboard',
  backendConnected: false,
  ollamaConnected: false,
  localModelAvailable: false,
  activeMode: 'ollama',
  activeOllamaModel: 'qwen2.5:7b',
  activeLocalModelPath: '',
  activeProfile: 'standard_corrector',
  selectedProjectFiles: [],
  conversations: [],
  currentConversationId: null,
  sidebarCollapsed: false,
  recentActivities: []
};

// Toast notification helper
function showToast(message, type = 'info', duration = 4000) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  const colors = {
    info: 'bg-indigo-950/90 border-indigo-500/60 text-indigo-200',
    success: 'bg-emerald-950/90 border-emerald-500/60 text-emerald-200',
    warning: 'bg-amber-950/90 border-amber-500/60 text-amber-200',
    error: 'bg-rose-950/90 border-rose-500/60 text-rose-200'
  };

  const icons = {
    info: '<svg class="w-4 h-4 text-indigo-400 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"></path></svg>',
    success: '<svg class="w-4 h-4 text-emerald-400 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>',
    warning: '<svg class="w-4 h-4 text-amber-400 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path></svg>',
    error: '<svg class="w-4 h-4 text-rose-400 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path></svg>'
  };

  toast.className = `flex items-center px-3.5 py-2.5 rounded-xl border backdrop-blur-md shadow-2xl transition-all duration-300 transform translate-y-2 opacity-0 text-xs font-medium ${colors[type] || colors.info}`;
  toast.innerHTML = `${icons[type] || icons.info}<span class="leading-snug">${message}</span>`;

  container.appendChild(toast);
  requestAnimationFrame(() => {
    toast.classList.remove('translate-y-2', 'opacity-0');
  });

  setTimeout(() => {
    toast.classList.add('opacity-0', '-translate-y-2');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// Activity Logging
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
    container.innerHTML = '<div class="text-xs text-slate-500 py-2 italic">No recent activity yet. Start proofreading or chatting.</div>';
    return;
  }

  container.innerHTML = AppState.recentActivities.map(a => `
    <div class="flex items-center justify-between text-xs py-1.5 border-b border-slate-800/60 last:border-0">
      <div class="flex items-center space-x-2 truncate">
        <span class="w-1.5 h-1.5 rounded-full bg-indigo-500"></span>
        <span class="text-slate-300 truncate">${a.text}</span>
      </div>
      <span class="text-[10px] text-slate-500 font-mono flex-shrink-0 ml-2">${a.time}</span>
    </div>
  `).join('');
}

// Sidebar Collapse / Expand Toggle
function toggleSidebar() {
  const sidebar = document.getElementById('app-sidebar');
  if (!sidebar) return;

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

  // Update nav buttons
  document.querySelectorAll('.nav-btn').forEach(btn => {
    const target = btn.getAttribute('data-tab');
    if (target === tabId) {
      btn.classList.add('bg-indigo-600/20', 'text-indigo-300', 'border-indigo-500/40');
      btn.classList.remove('text-slate-400', 'hover:bg-slate-800/60', 'hover:text-slate-200');
    } else {
      btn.classList.remove('bg-indigo-600/20', 'text-indigo-300', 'border-indigo-500/40');
      btn.classList.add('text-slate-400', 'hover:bg-slate-800/60', 'hover:text-slate-200');
    }
  });

  // Show active tab panel
  document.querySelectorAll('.tab-panel').forEach(panel => {
    if (panel.id === `tab-${tabId}`) {
      panel.classList.remove('hidden');
    } else {
      panel.classList.add('hidden');
    }
  });

  // On-show module hooks
  if (tabId === 'chat' && window.ChatModule) {
    window.ChatModule.onShow();
  } else if (tabId === 'project' && window.ProjectModule) {
    window.ProjectModule.onShow();
  } else if (tabId === 'settings' && window.SettingsModule) {
    window.SettingsModule.onShow();
  }

  // Close mobile sidebar if open
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

// Check Backend Health and Model Status
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
  // Backend badge
  const backendBadge = document.getElementById('badge-backend');
  if (backendBadge) {
    backendBadge.innerHTML = AppState.backendConnected
      ? `<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse mr-1.5"></span>Backend`
      : `<span class="w-2 h-2 rounded-full bg-rose-500 mr-1.5"></span>Backend`;
    backendBadge.className = `flex items-center text-xs font-medium px-2.5 py-1 rounded-full border cursor-pointer ${AppState.backendConnected ? 'bg-emerald-950/40 text-emerald-300 border-emerald-800/50' : 'bg-rose-950/40 text-rose-300 border-rose-800/50'}`;
  }

  // Ollama badge
  const ollamaBadge = document.getElementById('badge-ollama');
  if (ollamaBadge) {
    ollamaBadge.innerHTML = AppState.ollamaConnected
      ? `<span class="w-2 h-2 rounded-full bg-emerald-400 mr-1.5"></span>Ollama (${AppState.activeOllamaModel})`
      : `<span class="w-2 h-2 rounded-full bg-amber-500 mr-1.5"></span>Ollama Offline`;
    ollamaBadge.className = `flex items-center text-xs font-medium px-2.5 py-1 rounded-full border cursor-pointer ${AppState.ollamaConnected ? 'bg-emerald-950/40 text-emerald-300 border-emerald-800/50' : 'bg-amber-950/40 text-amber-300 border-amber-800/50'}`;
  }

  // Local model badge
  const localBadge = document.getElementById('badge-local-model');
  if (localBadge) {
    localBadge.innerHTML = AppState.localModelAvailable
      ? `<span class="w-2 h-2 rounded-full bg-indigo-400 mr-1.5"></span>Local Model`
      : `<span class="w-2 h-2 rounded-full bg-slate-500 mr-1.5"></span>Local Model`;
    localBadge.className = `flex items-center text-xs font-medium px-2.5 py-1 rounded-full border cursor-pointer ${AppState.localModelAvailable ? 'bg-indigo-950/40 text-indigo-300 border-indigo-800/50' : 'bg-slate-900 text-slate-400 border-slate-800'}`;
  }
}

function updateModeSelectors() {
  document.querySelectorAll('.active-mode-select').forEach(sel => {
    sel.value = AppState.activeMode;
  });
}

// Global Switch Active LLM Mode
async function switchActiveMode(mode) {
  try {
    const res = await fetch('/api/models/switch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode })
    });

    if (res.ok) {
      AppState.activeMode = mode;
      updateStatusBadges();
      updateModeSelectors();
      if (window.ChatModule) window.ChatModule.updateHeader();
      showToast(`Switched active mode to ${mode === 'ollama' ? 'Ollama (Qwen 2.5 7B)' : 'Project Local Model'}`, 'success');
      logActivity(`Switched active mode: ${mode}`);
    } else {
      showToast(`Failed to switch mode to ${mode}`, 'error');
    }
  } catch (err) {
    showToast(`Error switching mode: ${err.message}`, 'error');
  }
}

// Command Palette (⌘K / Ctrl+K)
function openCommandPalette() {
  const modal = document.getElementById('command-palette-modal');
  const input = document.getElementById('command-palette-input');
  if (modal && input) {
    modal.classList.remove('hidden');
    input.value = '';
    input.focus();
    renderCommandSuggestions('');
  }
}

function closeCommandPalette() {
  const modal = document.getElementById('command-palette-modal');
  if (modal) modal.classList.add('hidden');
}

const COMMANDS = [
  { title: 'Proofread & Grammar Correction', desc: 'Open Document Proofreader', action: () => switchTab('files') },
  { title: 'Interactive AI Chat', desc: 'Open Grammar Chat Workstation', action: () => switchTab('chat') },
  { title: 'Repository Doc & Docstring Audit', desc: 'Analyze Workspace Documentation', action: () => switchTab('project') },
  { title: 'Settings & Model Workbench', desc: 'Configure LLM Backends', action: () => switchTab('settings') },
  { title: 'Switch to Ollama Mode (Qwen 2.5 7B)', desc: 'Run via Ollama REST API', action: () => switchActiveMode('ollama') },
  { title: 'Switch to Project Local Model Mode', desc: 'Run offline Seq2Seq model', action: () => switchActiveMode('local_model') },
  { title: 'Run Automated Grammar Benchmark', desc: 'Execute Experiment 8 validation suite', action: () => runBenchmarkPipeline() }
];

function renderCommandSuggestions(query) {
  const list = document.getElementById('command-palette-list');
  if (!list) return;

  const filtered = COMMANDS.filter(c => 
    c.title.toLowerCase().includes(query.toLowerCase()) || 
    c.desc.toLowerCase().includes(query.toLowerCase())
  );

  if (filtered.length === 0) {
    list.innerHTML = '<div class="p-3 text-xs text-slate-500 italic">No matching actions found.</div>';
    return;
  }

  list.innerHTML = filtered.map((c, idx) => `
    <div onclick="COMMANDS[${COMMANDS.indexOf(c)}].action(); closeCommandPalette();" 
         class="p-2.5 rounded-xl border border-transparent hover:border-indigo-500/40 hover:bg-slate-800/80 cursor-pointer flex items-center justify-between transition group">
      <div>
        <div class="text-xs font-medium text-slate-200 group-hover:text-indigo-300">${c.title}</div>
        <div class="text-[11px] text-slate-500">${c.desc}</div>
      </div>
      <span class="text-[10px] text-slate-600 group-hover:text-indigo-400 font-mono">↵ Select</span>
    </div>
  `).join('');
}

async function runBenchmarkPipeline() {
  showToast('Running Experiment 8 Grammar Correction benchmark...', 'info');
  switchTab('dashboard');
}

// Global Drag & Drop for Files
function initGlobalDragAndDrop() {
  const overlay = document.getElementById('global-drag-overlay');
  if (!overlay) return;

  window.addEventListener('dragover', (e) => {
    e.preventDefault();
    overlay.classList.remove('hidden');
  });

  overlay.addEventListener('dragleave', (e) => {
    overlay.classList.add('hidden');
  });

  overlay.addEventListener('drop', (e) => {
    e.preventDefault();
    overlay.classList.add('hidden');
    if (e.dataTransfer.files.length > 0) {
      switchTab('files');
      if (window.FilesModule) {
        window.FilesModule.handleFileUpload(e.dataTransfer.files[0]);
      }
    }
  });
}

// Keyboard shortcuts (Cmd+K, Escape)
function initKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      openCommandPalette();
    } else if (e.key === 'Escape') {
      closeCommandPalette();
    }
  });

  const cmdInput = document.getElementById('command-palette-input');
  if (cmdInput) {
    cmdInput.addEventListener('input', (e) => {
      renderCommandSuggestions(e.target.value);
    });
  }
}

// Initialize Application on DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  initSidebarState();
  initGlobalDragAndDrop();
  initKeyboardShortcuts();

  // Mode select listeners
  document.querySelectorAll('.active-mode-select').forEach(sel => {
    sel.addEventListener('change', (e) => {
      switchActiveMode(e.target.value);
    });
  });

  // Nav clicks
  document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.getAttribute('data-tab');
      if (tab) switchTab(tab);
    });
  });

  // Initialize modules
  if (window.ChatModule) window.ChatModule.init();
  if (window.FilesModule) window.FilesModule.init();
  if (window.ProjectModule) window.ProjectModule.init();
  if (window.SettingsModule) window.SettingsModule.init();

  // Polling health check
  checkSystemHealth();
  setInterval(checkSystemHealth, 8000);
});
