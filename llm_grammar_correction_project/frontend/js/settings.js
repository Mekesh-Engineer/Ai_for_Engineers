/**
 * Local LLM Studio - Settings, Models Configuration & Workbench (Grammar Studio)
 */

const SettingsModule = {
  settingsData: null,
  discoveredOllamaModels: [],
  discoveredLocalModels: [],

  init() {
    this.bindEvents();
    this.loadAppearancePreferences();
  },

  onShow() {
    this.loadSettings();
    this.refreshModels();
  },

  bindEvents() {
    document.querySelectorAll('.mode-toggle-radio').forEach(radio => {
      radio.addEventListener('change', (e) => {
        switchActiveMode(e.target.value);
        this.updateVisibleConfigPanels(e.target.value);
      });
    });

    const testOllamaBtn = document.getElementById('test-ollama-btn');
    const refreshOllamaBtn = document.getElementById('refresh-ollama-btn');
    const testLocalBtn = document.getElementById('test-local-btn');
    const saveSettingsBtn = document.getElementById('save-settings-btn');
    const createProfileBtn = document.getElementById('save-custom-profile-btn');
    const compareBtn = document.getElementById('run-model-comparison-btn');

    if (testOllamaBtn) {
      testOllamaBtn.addEventListener('click', () => this.testOllama());
    }

    if (refreshOllamaBtn) {
      refreshOllamaBtn.addEventListener('click', () => this.refreshModels());
    }

    if (testLocalBtn) {
      testLocalBtn.addEventListener('click', () => this.testLocalModel());
    }

    if (saveSettingsBtn) {
      saveSettingsBtn.addEventListener('click', () => this.saveSettings());
    }

    if (createProfileBtn) {
      createProfileBtn.addEventListener('click', () => this.createCustomProfile());
    }

    if (compareBtn) {
      compareBtn.addEventListener('click', () => this.runModelComparison());
    }

    // Appearance toggles
    const compactToggle = document.getElementById('pref-compact-mode');
    const glassToggle = document.getElementById('pref-glass-effects');
    const motionToggle = document.getElementById('pref-reduced-motion');

    if (compactToggle) {
      compactToggle.addEventListener('change', (e) => this.setAppearancePref('compact', e.target.checked));
    }
    if (glassToggle) {
      glassToggle.addEventListener('change', (e) => this.setAppearancePref('glass', e.target.checked));
    }
    if (motionToggle) {
      motionToggle.addEventListener('change', (e) => this.setAppearancePref('reduced_motion', e.target.checked));
    }
  },

  loadAppearancePreferences() {
    const compact = localStorage.getItem('studio_pref_compact') === 'true';
    const glass = localStorage.getItem('studio_pref_glass') !== 'false';
    const motion = localStorage.getItem('studio_pref_motion') === 'true';

    const compactToggle = document.getElementById('pref-compact-mode');
    const glassToggle = document.getElementById('pref-glass-effects');
    const motionToggle = document.getElementById('pref-reduced-motion');

    if (compactToggle) compactToggle.checked = compact;
    if (glassToggle) glassToggle.checked = glass;
    if (motionToggle) motionToggle.checked = motion;

    if (compact) document.body.classList.add('compact-mode');
    if (motion) document.body.classList.add('reduced-motion');
  },

  setAppearancePref(pref, enabled) {
    if (pref === 'compact') {
      localStorage.setItem('studio_pref_compact', enabled ? 'true' : 'false');
      document.body.classList.toggle('compact-mode', enabled);
    } else if (pref === 'glass') {
      localStorage.setItem('studio_pref_glass', enabled ? 'true' : 'false');
    } else if (pref === 'reduced_motion') {
      localStorage.setItem('studio_pref_motion', enabled ? 'true' : 'false');
      document.body.classList.toggle('reduced-motion', enabled);
    }
    showToast(`Saved preference: ${pref}`, 'info', 2000);
  },

  updateVisibleConfigPanels(mode) {
    const ollamaPanel = document.getElementById('ollama-config-panel');
    const localPanel = document.getElementById('local-config-panel');

    if (ollamaPanel) {
      ollamaPanel.classList.toggle('border-indigo-500/80', mode === 'ollama');
    }
    if (localPanel) {
      localPanel.classList.toggle('border-violet-500/80', mode === 'local_model');
    }
  },

  async loadSettings() {
    try {
      const res = await fetch('/api/settings');
      if (res.ok) {
        const data = await res.json();
        this.settingsData = data;
        
        const s = data.settings;
        const urlInput = document.getElementById('settings-ollama-url');
        const dirInput = document.getElementById('settings-local-dir');
        
        if (urlInput) urlInput.value = s.ollama.base_url;
        if (dirInput) dirInput.value = s.local_model.directory;
        
        document.querySelectorAll('.mode-toggle-radio').forEach(r => {
          r.checked = (r.value === s.active_mode);
        });

        this.updateVisibleConfigPanels(s.active_mode);
        this.renderProfilesList(data.profiles);
      }
    } catch (err) {
      showToast(`Error loading settings: ${err.message}`, 'error');
    }
  },

  renderProfilesList(profiles) {
    const container = document.getElementById('settings-profiles-container');
    if (!container) return;

    container.innerHTML = '';
    for (const [key, profile] of Object.entries(profiles)) {
      const card = document.createElement('div');
      card.className = 'glass-panel p-4 rounded-xl border border-slate-700/60 flex flex-col justify-between';
      card.innerHTML = `
        <div>
          <div class="flex items-center justify-between mb-1">
            <h4 class="text-sm font-semibold text-white">${profile.name}</h4>
            <span class="text-[10px] uppercase font-mono px-2 py-0.5 rounded ${key === AppState.activeProfile ? 'bg-indigo-600 text-white' : 'bg-slate-800 text-slate-400'}">${key}</span>
          </div>
          <p class="text-xs text-slate-400 mb-3">${profile.description || ''}</p>
          <pre class="text-[11px] font-mono text-slate-300 bg-slate-900/80 p-2.5 rounded-lg border border-slate-800 overflow-x-auto whitespace-pre-wrap max-h-24">${profile.prompt}</pre>
        </div>
        <div class="mt-3 pt-2 border-t border-slate-800/80 flex items-center justify-between">
          <button onclick="SettingsModule.applyProfile('${key}')" class="text-xs font-medium text-indigo-400 hover:text-indigo-300">
            ${key === AppState.activeProfile ? '✓ Active Profile' : 'Set as Active'}
          </button>
        </div>
      `;
      container.appendChild(card);
    }
  },

  applyProfile(key) {
    AppState.activeProfile = key;
    if (window.ChatModule) window.ChatModule.loadProfiles();
    this.loadSettings();
    showToast(`Active profile set to '${key}'`, 'success');
  },

  async refreshModels() {
    try {
      const res = await fetch('/api/models');
      if (res.ok) {
        const data = await res.json();
        
        // Ollama models
        const ollamaList = document.getElementById('ollama-models-list');
        if (ollamaList && data.ollama.available_models) {
          this.discoveredOllamaModels = data.ollama.available_models;
          ollamaList.innerHTML = data.ollama.available_models.map(m => `
            <div class="p-2.5 rounded-xl border border-slate-800 bg-slate-900/60 flex items-center justify-between text-xs">
              <div>
                <span class="font-medium text-slate-200 font-mono">${m.name}</span>
                <span class="text-[10px] text-slate-500 block">${m.size_gb} GB • ${m.family}</span>
              </div>
              <button onclick="SettingsModule.selectOllamaModel('${m.name}')" class="px-2.5 py-1 rounded bg-indigo-600/30 text-indigo-300 hover:bg-indigo-600 hover:text-white transition text-xs font-medium">
                ${m.name === AppState.activeOllamaModel ? '✓ Active' : 'Select'}
              </button>
            </div>
          `).join('');
        }

        // Local models
        const localList = document.getElementById('local-models-list');
        if (localList && data.local_model.discovered_models) {
          this.discoveredLocalModels = data.local_model.discovered_models;
          localList.innerHTML = data.local_model.discovered_models.map(m => `
            <div class="p-2.5 rounded-xl border border-slate-800 bg-slate-900/60 flex items-center justify-between text-xs">
              <div>
                <span class="font-medium text-slate-200 font-mono">${m.name}</span>
                <span class="text-[10px] text-slate-500 block">${m.size_mb} MB • ${m.format}</span>
              </div>
              <button onclick="SettingsModule.selectLocalModel('${m.path}')" class="px-2.5 py-1 rounded bg-violet-600/30 text-violet-300 hover:bg-violet-600 hover:text-white transition text-xs font-medium">
                Select
              </button>
            </div>
          `).join('');
        }
      }
    } catch (err) {
      console.warn('Error refreshing models:', err);
    }
  },

  async selectOllamaModel(modelName) {
    try {
      await fetch('/api/models/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'ollama', model_name_or_path: modelName })
      });
      AppState.activeOllamaModel = modelName;
      checkSystemHealth();
      this.refreshModels();
      showToast(`Selected Ollama model: ${modelName}`, 'success');
    } catch (err) {
      showToast(`Failed to select model: ${err.message}`, 'error');
    }
  },

  async selectLocalModel(modelPath) {
    try {
      await fetch('/api/models/switch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: 'local_model', model_name_or_path: modelPath })
      });
      checkSystemHealth();
      this.refreshModels();
      showToast(`Selected Local Model checkpoint`, 'success');
    } catch (err) {
      showToast(`Failed to select local model: ${err.message}`, 'error');
    }
  },

  async testOllama() {
    const outputEl = document.getElementById('ollama-test-output');
    showToast('Running diagnostic test on Ollama service...', 'info');

    try {
      const res = await fetch('/api/models/test-ollama', { method: 'POST' });
      const data = await res.json();

      if (outputEl) {
        outputEl.classList.remove('hidden');
        outputEl.innerHTML = `
          <div class="p-3 rounded-xl border ${data.ollama_reachable ? 'border-emerald-500/40 bg-emerald-950/20 text-emerald-300' : 'border-rose-500/40 bg-rose-950/20 text-rose-300'} text-xs">
            <div class="font-bold mb-1">${data.message}</div>
            <div class="text-[11px] text-slate-400 font-mono">Latency: ${data.latency_seconds}s | Models: ${data.details?.installed_models?.join(', ') || 'None'}</div>
          </div>
        `;
      }
      if (data.ollama_reachable) showToast('Ollama service passed diagnostic tests!', 'success');
      else showToast('Ollama connection test failed', 'error');
    } catch (err) {
      showToast(`Test error: ${err.message}`, 'error');
    }
  },

  async testLocalModel() {
    const outputEl = document.getElementById('local-test-output');
    showToast('Testing local Transformer model checkpoint...', 'info');

    try {
      const res = await fetch('/api/models/test-local', { method: 'POST' });
      const data = await res.json();

      if (outputEl) {
        outputEl.classList.remove('hidden');
        outputEl.innerHTML = `
          <div class="p-3 rounded-xl border ${data.model_valid ? 'border-indigo-500/40 bg-indigo-950/20 text-indigo-300' : 'border-rose-500/40 bg-rose-950/20 text-rose-300'} text-xs">
            <div class="font-bold mb-1">${data.message}</div>
            <div class="text-[11px] text-slate-400 font-mono">Latency: ${data.latency_seconds}s | Sample Output: ${data.details?.sample_output || 'N/A'}</div>
          </div>
        `;
      }
      if (data.model_valid) showToast('Local model passed validation!', 'success');
      else showToast('Local model check failed', 'error');
    } catch (err) {
      showToast(`Test error: ${err.message}`, 'error');
    }
  },

  async saveSettings() {
    const ollamaUrl = document.getElementById('settings-ollama-url')?.value.trim();
    const localDir = document.getElementById('settings-local-dir')?.value.trim();

    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ollama_base_url: ollamaUrl,
          active_local_model_path: localDir
        })
      });

      if (res.ok) {
        showToast('Settings saved successfully', 'success');
      }
    } catch (err) {
      showToast(`Failed to save settings: ${err.message}`, 'error');
    }
  },

  async createCustomProfile() {
    const id = document.getElementById('custom-profile-id')?.value.trim();
    const name = document.getElementById('custom-profile-name')?.value.trim();
    const desc = document.getElementById('custom-profile-desc')?.value.trim();
    const prompt = document.getElementById('custom-profile-prompt')?.value.trim();

    if (!id || !name || !prompt) {
      showToast('Please fill in profile ID, name, and system prompt', 'warning');
      return;
    }

    try {
      const res = await fetch('/api/settings/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_id: id,
          name: name,
          description: desc,
          prompt: prompt
        })
      });

      if (res.ok) {
        const data = await res.json();
        this.renderProfilesList(data.profiles);
        showToast(`Profile '${name}' saved!`, 'success');
      }
    } catch (err) {
      showToast(`Error saving profile: ${err.message}`, 'error');
    }
  },

  async runModelComparison() {
    const testSentence = document.getElementById('model-comparison-input')?.value.trim() || 'He go to the laboratory yesterday for doing the experiment.';
    const outputOllama = document.getElementById('comparison-ollama-output');
    const outputLocal = document.getElementById('comparison-local-output');

    if (outputOllama) outputOllama.innerHTML = '<span class="text-indigo-400 animate-pulse">Running Ollama Qwen 2.5 7B inference...</span>';
    if (outputLocal) outputLocal.innerHTML = '<span class="text-violet-400 animate-pulse">Running Local Model Seq2Seq inference...</span>';

    showToast('Executing dual-mode side-by-side benchmark...', 'info');

    // Run Ollama
    try {
      const t0 = Date.now();
      const res1 = await fetch('/api/files/proofread', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: testSentence, mode: 'ollama', correction_level: 'standard' })
      });
      const d1 = await res1.json();
      const dur1 = ((Date.now() - t0) / 1000).toFixed(2);
      if (outputOllama) {
        outputOllama.innerHTML = `
          <div class="text-[11px] text-slate-400 font-mono mb-1">Latency: ${dur1}s | Edit Distance: ${d1.levenshtein_distance}</div>
          <div class="text-slate-200 text-xs font-medium">${d1.corrected_text}</div>
        `;
      }
    } catch (e1) {
      if (outputOllama) outputOllama.innerHTML = `<span class="text-rose-400">Ollama error: ${e1.message}</span>`;
    }

    // Run Local Model
    try {
      const t0 = Date.now();
      const res2 = await fetch('/api/files/proofread', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: testSentence, mode: 'local_model', correction_level: 'standard' })
      });
      const d2 = await res2.json();
      const dur2 = ((Date.now() - t0) / 1000).toFixed(2);
      if (outputLocal) {
        outputLocal.innerHTML = `
          <div class="text-[11px] text-slate-400 font-mono mb-1">Latency: ${dur2}s | Edit Distance: ${d2.levenshtein_distance}</div>
          <div class="text-slate-200 text-xs font-medium">${d2.corrected_text}</div>
        `;
      }
    } catch (e2) {
      if (outputLocal) outputLocal.innerHTML = `<span class="text-rose-400">Local Model error: ${e2.message}</span>`;
    }

    showToast('Dual-mode comparison complete!', 'success');
  }
};

window.SettingsModule = SettingsModule;
