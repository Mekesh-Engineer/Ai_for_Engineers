/**
 * VOLTIX — App Core: State, Theme, Sidebar, Command Palette, Diagnostics
 * ChatGPT-style architecture — single-page conversational AI shell
 */

/* ─────────────────────────────────────────────
   GLOBAL STATE
   ───────────────────────────────────────────── */
const AppState = {
  // Conversation
  activeConversationId: null,
  conversations: [],
  currentConversationTitle: 'VOLTIX',

  // Project / workspace
  activeProjectId: null,
  activeProjectName: 'Default Workspace',
  projects: [],

  // Model & inference
  activeModel: 'auto',
  activeMode: 'Learn',
  ragEnabled: true,
  temperature: 0.1,
  topP: 0.9,

  // UI
  sidebarOpen: true,       // desktop
  mobileSidebarOpen: false, // mobile

  // Status
  ollamaOnline: false,
  ollamaLatency: null,

  // Legacy aliases (keep JS modules happy)
  get currentConversationId() { return this.activeConversationId; },
  set currentConversationId(v) { this.activeConversationId = v; },
  backendConnected: true,
  ollamaConnected: false,
  sidebarCollapsed: false,
  recentActivities: []
};

/* ─────────────────────────────────────────────
   THEME ENGINE
   ───────────────────────────────────────────── */
function initTheme() {
  const saved = localStorage.getItem('voltix_theme')
             || localStorage.getItem('studio_theme')
             || 'dark';
  applyTheme(saved, false);
}

function applyTheme(theme, toast = true) {
  const isDark = theme === 'dark';
  document.documentElement.classList.toggle('dark', isDark);
  document.documentElement.classList.toggle('light', !isDark);
  localStorage.setItem('voltix_theme', theme);
  localStorage.setItem('studio_theme', theme); // legacy alias

  // Theme icons
  const sun  = document.getElementById('theme-icon-sun');
  const moon = document.getElementById('theme-icon-moon');
  if (sun)  sun.style.display  = isDark ? 'block' : 'none';
  if (moon) moon.style.display = isDark ? 'none'  : 'block';

  // Settings select
  const sel = document.getElementById('settings-theme-select');
  if (sel) sel.value = theme;

  if (toast) showToast(isDark ? 'Obsidian Dark' : 'Precision Light', 'info', 1800);
}

let _lastToggle = 0;
function toggleTheme() {
  const now = Date.now();
  if (now - _lastToggle < 200) return;
  _lastToggle = now;
  const isDark = document.documentElement.classList.contains('dark');
  applyTheme(isDark ? 'light' : 'dark');
}

// Legacy alias
function updateThemeIcons(isDark) {
  const sun  = document.getElementById('theme-icon-sun');
  const moon = document.getElementById('theme-icon-moon');
  if (sun)  sun.style.display  = isDark ? 'block' : 'none';
  if (moon) moon.style.display = isDark ? 'none'  : 'block';
}

/* ─────────────────────────────────────────────
   TOAST NOTIFICATIONS
   ───────────────────────────────────────────── */
function showToast(message, type = 'info', duration = 3200) {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const typeIcons = {
    info:    'fa-circle-info',
    success: 'fa-circle-check',
    warning: 'fa-triangle-exclamation',
    error:   'fa-circle-exclamation'
  };

  const toast = document.createElement('div');
  toast.className = 'toast';
  toast.innerHTML = `
    <i class="fa-solid ${typeIcons[type] || typeIcons.info} toast-icon ${type}"></i>
    <span style="flex:1;line-height:1.4;">${message}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('out');
    setTimeout(() => toast.remove(), 200);
  }, duration);
}

/* ─────────────────────────────────────────────
   CLIPBOARD
   ───────────────────────────────────────────── */
async function copyToClipboard(text) {
  if (!text) return false;
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (e) { /* fall through */ }
  try {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;left:-9999px;top:-9999px;';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    ta.remove();
    return ok;
  } catch (e) { return false; }
}

/* ─────────────────────────────────────────────
   ACTIVITY LOG (legacy compat)
   ───────────────────────────────────────────── */
function logActivity(text, type = 'info') {
  AppState.recentActivities.unshift({ text, type, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) });
  if (AppState.recentActivities.length > 12) AppState.recentActivities.pop();
}

/* ─────────────────────────────────────────────
   SIDEBAR
   ───────────────────────────────────────────── */
function initSidebarState() {
  const isMobile = window.innerWidth < 1024;
  const saved = localStorage.getItem('voltix_sidebar_collapsed');

  if (isMobile) {
    AppState.sidebarOpen = false;
    _setSidebarMobile(false);
  } else {
    const collapsed = saved === 'true';
    AppState.sidebarCollapsed = collapsed;
    AppState.sidebarOpen = !collapsed;
    _setSidebarDesktop(collapsed);
  }
}

function toggleSidebar() {
  const isMobile = window.innerWidth < 1024;
  if (isMobile) {
    AppState.mobileSidebarOpen = !AppState.mobileSidebarOpen;
    _setSidebarMobile(AppState.mobileSidebarOpen);
  } else {
    AppState.sidebarCollapsed = !AppState.sidebarCollapsed;
    AppState.sidebarOpen = !AppState.sidebarCollapsed;
    _setSidebarDesktop(AppState.sidebarCollapsed);
    localStorage.setItem('voltix_sidebar_collapsed', AppState.sidebarCollapsed ? 'true' : 'false');
  }
}

function _setSidebarDesktop(collapsed) {
  const sb = document.getElementById('sidebar');
  if (!sb) return;
  sb.classList.toggle('collapsed', collapsed);
}

function _setSidebarMobile(open) {
  const sb      = document.getElementById('sidebar');
  const overlay = document.getElementById('sidebar-overlay');
  if (!sb) return;
  sb.classList.toggle('open', open);
  if (overlay) overlay.classList.toggle('visible', open);
}

function openMobileSidebar() { AppState.mobileSidebarOpen = true;  _setSidebarMobile(true); }
function closeMobileSidebar() { AppState.mobileSidebarOpen = false; _setSidebarMobile(false); }

/* ─────────────────────────────────────────────
   MODEL HANDLING
   ───────────────────────────────────────────── */
function handleModelSwitch(value) {
  AppState.activeModel = value;

  const display = document.getElementById('header-model-display');
  const compName = document.getElementById('composer-model-name');
  const dot    = document.getElementById('composer-model-dot');

  // Find friendly name from dropdown
  const option = document.querySelector(`#model-dropdown-menu [data-value="${value}"]`);
  const friendlyName = option ? option.querySelector('.model-option-name')?.textContent?.trim() : value;
  const isCloud = option?.classList.contains('cloud') || value.includes(':cloud');
  const isLocal = option?.classList.contains('local') || value.includes('qwen') || value.includes('llama');

  if (display) display.textContent = friendlyName || value;
  if (compName) compName.textContent = friendlyName || value;
  if (dot) {
    dot.classList.remove('online', 'offline', 'checking');
    if (value === 'auto') dot.style.background = 'var(--accent)';
    else if (isCloud)     dot.style.background = 'var(--info)';
    else                  dot.style.background = 'var(--success)';
  }

  // Mark active in dropdown
  document.querySelectorAll('#model-dropdown-menu .model-option').forEach(el => {
    el.classList.toggle('active', el.dataset.value === value);
  });

  // Legacy select compatibility
  const sel = document.getElementById('global-model-select');
  if (sel && sel.value !== value) sel.value = value;

  showToast(`Model: ${friendlyName || value}`, 'info', 1800);
}

/* ─────────────────────────────────────────────
   MODAL SYSTEM
   ───────────────────────────────────────────── */
function openModal(id) {
  const m = document.getElementById(id);
  if (m) m.classList.add('open');
}

function closeModal(id) {
  const m = document.getElementById(id);
  if (m) m.classList.remove('open');
}

// Legacy alias (ChatModule uses this)
function _legacyCloseModal(modalId) {
  const m = document.getElementById(modalId);
  if (!m) return;
  // Handle both class-based (new) and hidden-based (legacy)
  m.classList.remove('open');
  m.classList.add('hidden');
}

/* ─────────────────────────────────────────────
   SETTINGS TABS
   ───────────────────────────────────────────── */
function initSettingsTabs() {
  document.querySelectorAll('.settings-nav-item').forEach(item => {
    item.addEventListener('click', () => {
      const section = item.dataset.section;
      // Activate nav item
      document.querySelectorAll('.settings-nav-item').forEach(i => i.classList.remove('active'));
      item.classList.add('active');
      // Show section
      document.querySelectorAll('.settings-pane').forEach(p => p.classList.add('hidden'));
      const pane = document.getElementById(`settings-section-${section}`);
      if (pane) pane.classList.remove('hidden');
    });
  });
}

/* ─────────────────────────────────────────────
   DIAGNOSTICS
   ───────────────────────────────────────────── */
async function runSystemDiagnostics(targetId = 'diagnostics-results') {
  const results = document.getElementById(targetId);
  if (!results) return;

  results.innerHTML = '<div style="color:var(--text-3);font-style:italic;font-size:13px;display:flex;align-items:center;gap:8px;"><span style="display:inline-block;width:14px;height:14px;border:2px solid var(--border-md);border-top-color:var(--accent);border-radius:50%;animation:spin 0.8s linear infinite;"></span>Running diagnostics…</div>';
  openModal('modal-diagnostics');

  try {
    const [health, models, projects] = await Promise.all([
      VoltixAPI.checkHealth(),
      VoltixAPI.getModels(),
      VoltixAPI.getProjects()
    ]);

    const ollamaOk = health.status === 'online' || health.ollama_reachable;
    AppState.ollamaOnline = ollamaOk;
    AppState.ollamaLatency = health.latency_ms;

    results.innerHTML = `
      <div style="display:flex;flex-direction:column;gap:6px;">
        <div class="diag-row">
          <span>Flask API Backend</span>
          <span class="diag-status ok"><i class="fa-solid fa-circle-check" style="font-size:11px;"></i> 200 OK</span>
        </div>
        <div class="diag-row">
          <span>Ollama Inference Engine</span>
          <span class="diag-status ${ollamaOk ? 'ok' : 'error'}">
            ${ollamaOk ? `<i class="fa-solid fa-circle-check" style="font-size:11px;"></i> Online (${health.latency_ms}ms)` : '<i class="fa-solid fa-circle-xmark" style="font-size:11px;"></i> Offline'}
          </span>
        </div>
        <div class="diag-row">
          <span>Available Models</span>
          <span class="diag-status ok font-mono">${models.length || 0} loaded</span>
        </div>
        <div class="diag-row">
          <span>Project Workspaces</span>
          <span class="diag-status ok font-mono">${projects.length || 0} workspaces</span>
        </div>
      </div>
    `;

    updateOllamaStatus(ollamaOk, health.latency_ms);
  } catch (err) {
    results.innerHTML = `<div style="color:var(--danger);font-size:13px;"><i class="fa-solid fa-circle-xmark"></i> Diagnostics failed: ${err.message}</div>`;
  }
}

/* ─────────────────────────────────────────────
   STATUS INDICATORS
   ───────────────────────────────────────────── */
function updateOllamaStatus(online, latency) {
  const dot  = document.getElementById('status-dot-sidebar');
  const text = document.getElementById('status-text-sidebar');

  if (dot) {
    dot.classList.remove('online', 'offline', 'checking');
    dot.classList.add(online ? 'online' : 'offline');
  }
  if (text) {
    text.textContent = online ? `Online (${latency}ms)` : 'Ollama offline';
    text.style.color = online ? 'var(--success)' : 'var(--danger)';
  }
  AppState.ollamaConnected = online;
}

/* ─────────────────────────────────────────────
   COMMAND PALETTE
   ───────────────────────────────────────────── */
const CommandPalette = {
  commands: [
    { title: 'New chat',                  category: 'Chat',      icon: 'fa-plus',                    action: () => ChatModule.newChat() },
    { title: 'Engineering solvers',       category: 'Tools',     icon: 'fa-calculator',              action: () => openModal('modal-calc') },
    { title: 'Project workspaces',        category: 'Tools',     icon: 'fa-folder-tree',             action: () => openModal('modal-project') },
    { title: 'Settings',                  category: 'System',    icon: 'fa-sliders',                 action: () => openModal('modal-settings') },
    { title: 'Toggle dark / light theme', category: 'System',    icon: 'fa-moon',                    action: () => toggleTheme() },
    { title: 'Export conversation',       category: 'Chat',      icon: 'fa-download',                action: () => ChatModule.exportConversation('markdown') },
    { title: 'Run system diagnostics',    category: 'System',    icon: 'fa-stethoscope',             action: () => runSystemDiagnostics() },
    { title: 'Open context panel',        category: 'Chat',      icon: 'fa-circle-info',             action: () => openContextDrawer() },
    { title: '3-Phase Power solver',      category: 'Solvers',   icon: 'fa-bolt',                    action: () => { openModal('modal-calc'); SolversModule.switchSolverPane('form-3phase'); } },
    { title: 'Ohm\'s Law solver',         category: 'Solvers',   icon: 'fa-bolt-lightning',          action: () => { openModal('modal-calc'); SolversModule.switchSolverPane('form-ohms'); } },
    { title: 'Motor Slip solver',         category: 'Solvers',   icon: 'fa-gears',                   action: () => { openModal('modal-calc'); SolversModule.switchSolverPane('form-motor'); } },
    { title: 'SymPy symbolic math',       category: 'Solvers',   icon: 'fa-square-root-variable',    action: () => { openModal('modal-calc'); SolversModule.switchSolverPane('form-sympy'); } },
    { title: 'MATLAB code generator',     category: 'Solvers',   icon: 'fa-code',                    action: () => { openModal('modal-calc'); SolversModule.switchSolverPane('form-matlab'); } },
  ],

  open() {
    const backdrop = document.getElementById('modal-command-palette');
    const input    = document.getElementById('cmd-palette-input');
    if (!backdrop) return;
    backdrop.classList.add('open');
    if (input) { input.value = ''; input.focus(); }
    this.render('');
  },

  close() {
    document.getElementById('modal-command-palette')?.classList.remove('open');
  },

  render(query) {
    const container = document.getElementById('cmd-palette-results');
    if (!container) return;

    const q = query.toLowerCase().trim();
    const filtered = this.commands.filter(c =>
      !q || c.title.toLowerCase().includes(q) || c.category.toLowerCase().includes(q)
    );

    if (filtered.length === 0) {
      container.innerHTML = '<div class="cmd-empty">No matching actions found.</div>';
      return;
    }

    // Group by category
    const groups = {};
    filtered.forEach(c => {
      if (!groups[c.category]) groups[c.category] = [];
      groups[c.category].push(c);
    });

    container.innerHTML = Object.entries(groups).map(([cat, cmds]) => `
      <div class="cmd-group-label">${cat}</div>
      ${cmds.map((c, i) => `
        <div class="cmd-item" data-idx="${this.commands.indexOf(c)}" tabindex="0">
          <span class="cmd-icon"><i class="fa-solid ${c.icon}"></i></span>
          <span>${c.title}</span>
          <span class="cmd-category">${c.category}</span>
        </div>
      `).join('')}
    `).join('');

    container.querySelectorAll('.cmd-item').forEach(el => {
      el.addEventListener('click', () => {
        const idx = parseInt(el.dataset.idx);
        if (this.commands[idx]) {
          this.close();
          this.commands[idx].action();
        }
      });
    });
  }
};

/* ─────────────────────────────────────────────
   CONTEXT DRAWER
   ───────────────────────────────────────────── */
function openContextDrawer() {
  document.getElementById('context-drawer')?.classList.add('open');
  const ol = document.getElementById('context-drawer-overlay');
  if (ol) ol.style.display = 'block';
  ChatModule.updateContextDrawer();
}

function closeContextDrawer() {
  document.getElementById('context-drawer')?.classList.remove('open');
  const ol = document.getElementById('context-drawer-overlay');
  if (ol) ol.style.display = 'none';
}

/* ─────────────────────────────────────────────
   PROJECTS (sidebar rendering)
   ───────────────────────────────────────────── */
async function loadProjectsSidebar() {
  const list = document.getElementById('projects-list');
  if (!list) return;

  try {
    const projects = await VoltixAPI.getProjects();
    AppState.projects = projects || [];

    if (projects.length === 0) {
      list.innerHTML = '<div style="font-size:12px;color:var(--text-3);padding:6px 4px;">No workspaces yet.</div>';
      return;
    }

    list.innerHTML = projects.map(p => `
      <div class="sidebar-item ${p.id === AppState.activeProjectId ? 'active' : ''}"
           onclick="switchProject('${p.id}', ${JSON.stringify(p.name).replace(/"/g, '&quot;')})"
           title="${p.name}">
        <i class="item-icon fa-solid fa-folder" style="color:var(--warning);"></i>
        <span class="sidebar-item-text">${p.name}</span>
      </div>
    `).join('');
  } catch (err) {
    console.warn('Failed to load projects sidebar:', err);
  }
}

function switchProject(id, name) {
  AppState.activeProjectId = id;
  AppState.activeProjectName = name;
  showToast(`Workspace: ${name}`, 'info', 2000);
  loadProjectsSidebar();
  ChatModule.loadConversations();
  ChatModule.updateContextDrawer();
}

/* ─────────────────────────────────────────────
   OVERFLOW MENU (header ⋯)
   ───────────────────────────────────────────── */
function bindOverflowMenu() {
  const btn  = document.getElementById('btn-overflow-menu');
  const menu = document.getElementById('overflow-menu');
  if (!btn || !menu) return;

  btn.addEventListener('click', e => {
    e.stopPropagation();
    const open = menu.style.display === 'block';
    menu.style.display = open ? 'none' : 'block';
  });

  document.addEventListener('click', e => {
    if (!menu.contains(e.target) && !btn.contains(e.target)) {
      menu.style.display = 'none';
    }
  });

  document.getElementById('btn-open-context')?.addEventListener('click', () => {
    menu.style.display = 'none';
    openContextDrawer();
  });

  document.getElementById('btn-export-chat')?.addEventListener('click', () => {
    menu.style.display = 'none';
    ChatModule.exportConversation('markdown');
  });

  document.getElementById('btn-rename-chat')?.addEventListener('click', () => {
    menu.style.display = 'none';
    if (ChatModule.currentConversationId) {
      const current = AppState.conversations.find(c => c.id === ChatModule.currentConversationId);
      ChatModule.renameConversation(ChatModule.currentConversationId, current?.title || '');
    } else {
      showToast('No conversation selected', 'warning');
    }
  });

  document.getElementById('btn-delete-chat')?.addEventListener('click', () => {
    menu.style.display = 'none';
    if (ChatModule.currentConversationId) {
      // Show delete confirmation modal
      openModal('modal-delete-confirm');
      document.getElementById('btn-confirm-delete').onclick = async () => {
        closeModal('modal-delete-confirm');
        await ChatModule.deleteConversation(ChatModule.currentConversationId);
      };
    } else {
      showToast('No conversation selected', 'warning');
    }
  });

  document.getElementById('btn-run-diagnostics')?.addEventListener('click', () => {
    menu.style.display = 'none';
    runSystemDiagnostics();
  });

  document.getElementById('btn-open-cmd-palette')?.addEventListener('click', () => {
    menu.style.display = 'none';
    CommandPalette.open();
  });
}

/* ─────────────────────────────────────────────
   GLOBAL KEYBOARD SHORTCUTS
   ───────────────────────────────────────────── */
function bindKeyboardShortcuts() {
  document.addEventListener('keydown', e => {
    const ctrl = e.ctrlKey || e.metaKey;

    if (ctrl && e.key.toLowerCase() === 'k') {
      e.preventDefault();
      const cp = document.getElementById('modal-command-palette');
      if (cp?.classList.contains('open')) CommandPalette.close();
      else CommandPalette.open();
      return;
    }

    if (ctrl && e.key.toLowerCase() === 'b') {
      e.preventDefault();
      toggleSidebar();
      return;
    }

    if (ctrl && e.key.toLowerCase() === 'n' && !e.shiftKey) {
      e.preventDefault();
      ChatModule.newChat();
      return;
    }

    if (e.altKey && e.key.toLowerCase() === 't') {
      e.preventDefault();
      toggleTheme();
      return;
    }

    if (e.altKey && e.key.toLowerCase() === 's') {
      e.preventDefault();
      openModal('modal-calc');
      return;
    }

    if (e.key === 'Escape') {
      CommandPalette.close();
      closeModal('modal-command-palette');
      closeModal('modal-calc');
      closeModal('modal-project');
      closeModal('modal-settings');
      closeModal('modal-sources');
      closeModal('modal-delete-confirm');
      closeModal('modal-diagnostics');
      closeContextDrawer();
      // Also close model dropdown
      document.getElementById('model-dropdown-menu')?.classList.remove('open');
      document.getElementById('overflow-menu') && (document.getElementById('overflow-menu').style.display = 'none');
      document.getElementById('composer-popover-menu')?.classList.remove('open');
      // Mobile sidebar
      if (AppState.mobileSidebarOpen) closeMobileSidebar();
    }
  });
}

/* ─────────────────────────────────────────────
   MODEL DROPDOWN BINDING
   ───────────────────────────────────────────── */
function bindModelDropdown() {
  const btn  = document.getElementById('btn-model-dropdown');
  const menu = document.getElementById('model-dropdown-menu');
  if (!btn || !menu) return;

  btn.addEventListener('click', e => {
    e.stopPropagation();
    menu.classList.toggle('open');
  });

  document.addEventListener('click', e => {
    if (!menu.contains(e.target) && !btn.contains(e.target)) {
      menu.classList.remove('open');
    }
  });

  menu.querySelectorAll('.model-option[data-value]').forEach(opt => {
    opt.addEventListener('click', () => {
      handleModelSwitch(opt.dataset.value);
      menu.classList.remove('open');
    });
  });
}

/* ─────────────────────────────────────────────
   PROJECTS SECTION TOGGLE (sidebar)
   ───────────────────────────────────────────── */
function bindProjectsToggle() {
  const btn     = document.getElementById('btn-toggle-projects');
  const list    = document.getElementById('projects-list');
  const chevron = document.getElementById('projects-chevron');
  let open = true;

  btn?.addEventListener('click', () => {
    open = !open;
    if (list) list.style.display = open ? 'block' : 'none';
    if (chevron) chevron.style.transform = open ? 'rotate(0deg)' : 'rotate(-90deg)';
  });
}

/* ─────────────────────────────────────────────
   SETTINGS BINDINGS
   ───────────────────────────────────────────── */
function bindSettingsActions() {
  // Theme select in settings
  document.getElementById('settings-theme-select')?.addEventListener('change', e => {
    applyTheme(e.target.value, false);
  });

  // Compact mode
  document.getElementById('setting-compact-mode')?.addEventListener('change', e => {
    document.body.classList.toggle('compact-mode', e.target.checked);
    localStorage.setItem('voltix_compact', e.target.checked ? 'true' : 'false');
  });

  // Reduce motion
  document.getElementById('setting-reduce-motion')?.addEventListener('change', e => {
    document.body.classList.toggle('reduced-motion', e.target.checked);
    localStorage.setItem('voltix_reduced_motion', e.target.checked ? 'true' : 'false');
  });

  // Temperature slider
  const tempSlider = document.getElementById('setting-temp');
  const tempVal    = document.getElementById('temp-val');
  if (tempSlider && tempVal) {
    tempSlider.addEventListener('input', e => {
      tempVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  // Top-P slider
  const toppSlider = document.getElementById('setting-topp');
  const toppVal    = document.getElementById('topp-val');
  if (toppSlider && toppVal) {
    toppSlider.addEventListener('input', e => {
      toppVal.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  // Diagnostics inline (About section)
  document.getElementById('btn-run-diagnostics-inline')?.addEventListener('click', () => {
    runSystemDiagnostics('diag-results-settings');
    document.getElementById('modal-diagnostics')?.classList.remove('open');
  });

  // Load saved preferences
  try {
    const temp = localStorage.getItem('voltix_temp') || '0.1';
    const topp = localStorage.getItem('voltix_topp') || '0.9';
    if (tempSlider) tempSlider.value = temp;
    if (tempVal)    tempVal.textContent = parseFloat(temp).toFixed(2);
    if (toppSlider) toppSlider.value = topp;
    if (toppVal)    toppVal.textContent = parseFloat(topp).toFixed(2);

    const compact = localStorage.getItem('voltix_compact') === 'true';
    const reduced = localStorage.getItem('voltix_reduced_motion') === 'true';
    const compactCb = document.getElementById('setting-compact-mode');
    const reducedCb = document.getElementById('setting-reduce-motion');
    if (compactCb) compactCb.checked = compact;
    if (reducedCb) reducedCb.checked = reduced;
    if (compact) document.body.classList.add('compact-mode');
    if (reduced) document.body.classList.add('reduced-motion');
  } catch (e) {}
}

/* ─────────────────────────────────────────────
   MODAL CLOSE BUTTONS (universal .btn-close-modal)
   ───────────────────────────────────────────── */
function bindModalCloseButtons() {
  document.querySelectorAll('.btn-close-modal').forEach(btn => {
    btn.addEventListener('click', () => {
      const modal = btn.closest('.modal-backdrop');
      if (modal) modal.classList.remove('open');
    });
  });

  // Close on backdrop click
  document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
    backdrop.addEventListener('click', e => {
      if (e.target === backdrop) backdrop.classList.remove('open');
    });
  });
}

/* ─────────────────────────────────────────────
   SIDEBAR STATUS (status dot)
   ───────────────────────────────────────────── */
document.getElementById('btn-system-status')?.addEventListener('click', () => {
  runSystemDiagnostics();
});

/* ─────────────────────────────────────────────
   OLLAMA HEALTH CHECK
   ───────────────────────────────────────────── */
async function checkOllamaHealth() {
  const dot  = document.getElementById('status-dot-sidebar');
  const text = document.getElementById('status-text-sidebar');

  try {
    const health = await VoltixAPI.checkHealth();
    const online = health.status === 'online' || health.ollama_reachable;
    updateOllamaStatus(online, health.latency_ms);

    // Populate model list in header if available
    if (health.available_models?.length) {
      AppState.availableModels = health.available_models;
    }
  } catch (err) {
    updateOllamaStatus(false, null);
  }
}

/* ─────────────────────────────────────────────
   LEGACY COMPATIBILITY STUBS
   (kept for backward compat with older modules)
   ───────────────────────────────────────────── */
function switchTab(tabId) {
  // In the new layout there are no tabs; map tab IDs to modal/actions
  if (tabId === 'chat')     { closeMobileSidebar(); return; }
  if (tabId === 'solvers')  { openModal('modal-calc'); return; }
  if (tabId === 'files')    { openModal('modal-project'); return; }
  if (tabId === 'project')  { openModal('modal-project'); return; }
  if (tabId === 'settings') { openModal('modal-settings'); return; }
}

/* ─────────────────────────────────────────────
   GLOBAL APP INIT
   ───────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  initSidebarState();
  initSettingsTabs();
  bindModelDropdown();
  bindOverflowMenu();
  bindKeyboardShortcuts();
  bindModalCloseButtons();
  bindProjectsToggle();
  bindSettingsActions();

  // Status button click
  document.getElementById('btn-system-status')?.addEventListener('click', runSystemDiagnostics);

  // Sidebar toggle (hamburger in header)
  document.getElementById('btn-toggle-sidebar')?.addEventListener('click', toggleSidebar);

  // Sidebar collapse button
  document.getElementById('btn-collapse-sidebar')?.addEventListener('click', toggleSidebar);

  // Mobile sidebar overlay dismissal
  document.getElementById('sidebar-overlay')?.addEventListener('click', closeMobileSidebar);

  // Context drawer close
  document.getElementById('btn-close-context')?.addEventListener('click', closeContextDrawer);
  document.getElementById('context-drawer-overlay')?.addEventListener('click', closeContextDrawer);

  // Settings modal open/close
  document.getElementById('btn-open-settings')?.addEventListener('click', () => openModal('modal-settings'));

  // Theme toggle button
  document.getElementById('btn-theme-toggle')?.addEventListener('click', toggleTheme);

  // Command palette input
  const cmdInput = document.getElementById('cmd-palette-input');
  if (cmdInput) {
    cmdInput.addEventListener('input', e => CommandPalette.render(e.target.value));
  }

  // Command palette backdrop click close
  document.getElementById('modal-command-palette')?.addEventListener('click', e => {
    if (e.target === document.getElementById('modal-command-palette')) CommandPalette.close();
  });

  // Composer popover menu
  const composerPlusBtn = document.getElementById('btn-composer-menu');
  const composerPopover = document.getElementById('composer-popover-menu');
  if (composerPlusBtn && composerPopover) {
    composerPlusBtn.addEventListener('click', e => {
      e.stopPropagation();
      composerPopover.classList.toggle('open');
    });
    document.addEventListener('click', e => {
      if (!composerPopover.contains(e.target) && !composerPlusBtn.contains(e.target)) {
        composerPopover.classList.remove('open');
      }
    });

    const fileInput = document.getElementById('file-upload-input');
    document.getElementById('btn-action-upload')?.addEventListener('click', () => {
      composerPopover.classList.remove('open');
      fileInput?.click();
    });

    document.getElementById('btn-action-knowledge')?.addEventListener('click', () => {
      composerPopover.classList.remove('open');
      openModal('modal-project');
    });

    document.getElementById('btn-action-solvers')?.addEventListener('click', () => {
      composerPopover.classList.remove('open');
      openModal('modal-calc');
    });

    document.getElementById('btn-action-workspaces')?.addEventListener('click', () => {
      composerPopover.classList.remove('open');
      openModal('modal-project');
    });
  }

  // New project button (sidebar)
  document.getElementById('btn-new-project')?.addEventListener('click', () => {
    openModal('modal-project');
  });

  // Drag & drop overlay
  const mainArea = document.getElementById('main-viewport');
  const dragOverlay = document.getElementById('drag-overlay');
  if (mainArea && dragOverlay) {
    let dragCounter = 0;
    mainArea.addEventListener('dragenter', e => {
      if (e.dataTransfer.types.includes('Files')) {
        dragCounter++;
        dragOverlay.classList.add('active');
      }
    });
    mainArea.addEventListener('dragleave', () => {
      dragCounter--;
      if (dragCounter <= 0) {
        dragCounter = 0;
        dragOverlay.classList.remove('active');
      }
    });
    mainArea.addEventListener('dragover', e => e.preventDefault());
    mainArea.addEventListener('drop', e => {
      e.preventDefault();
      dragCounter = 0;
      dragOverlay.classList.remove('active');
      const file = e.dataTransfer.files?.[0];
      if (file && typeof ChatModule !== 'undefined') {
        ChatModule.handleChatFileAttachment(file);
      }
    });
  }

  // Initialize sub-modules (in order — no tab switching needed)
  if (typeof ChatModule    !== 'undefined') ChatModule.init();
  if (typeof SolversModule !== 'undefined') SolversModule.init();
  if (typeof FilesModule   !== 'undefined') FilesModule.init();
  if (typeof ProjectModule !== 'undefined') ProjectModule.init();
  if (typeof SettingsModule !== 'undefined') SettingsModule.init();

  // Health check
  await checkOllamaHealth();

  // Load sidebar projects
  await loadProjectsSidebar();
});
