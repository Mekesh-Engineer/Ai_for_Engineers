/**
 * Local LLM Studio - Settings, Models Configuration & Model Comparison Workbench
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
    // Mode radio / select buttons
    document.querySelectorAll('.mode-toggle-radio').forEach(radio => {
      radio.addEventListener('change', (e) => {
        handleModeSwitch(e.target.value);
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
    const glass = localStorage.getItem('studio_pref_glass') !== 'false'; // default true
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
        
        // Populate inputs
        const s = data.settings;
        const urlInput = document.getElementById('settings-ollama-url');
        const dirInput = document.getElementById('settings-local-dir');
        
        if (urlInput) urlInput.value = s.ollama.base_url;
        if (dirInput) dirInput.value = s.local_model.directory;
        
        // Select active radio
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
          <div class="bg-slate-900/80 p-2.5 rounded-lg border border-slate-800 text-[11px] text-slate-300 font-mono line-clamp-3">${profile.prompt}</div>
        </div>
        <div class="mt-3 pt-2 border-t border-slate-800 flex justify-end">
          <button onclick="SettingsModule.activateProfile('${key}')" class="text-xs px-2.5 py-1 bg-indigo-600/20 text-indigo-300 border border-indigo-500/40 rounded-lg hover:bg-indigo-600 hover:text-white transition">Select Profile</button>
        </div>
      `;
      container.appendChild(card);
    }
  },

  activateProfile(profileKey) {
    AppState.activeProfile = profileKey;
    const profileSelect = document.getElementById('chat-profile-select');
    if (profileSelect) profileSelect.value = profileKey;
    this.saveSettings();
    showToast(`Activated profile: ${profileKey}`, 'success');
  },

  async refreshModels() {
    try {
      const res = await fetch('/api/models');
      if (res.ok) {
        const data = await res.json();
        
        // Ollama models dropdown
        const ollamaSelect = document.getElementById('settings-ollama-model-select');
        if (ollamaSelect) {
          ollamaSelect.innerHTML = '';
          const models = data.ollama.available_models || [];
          if (models.length === 0) {
            ollamaSelect.innerHTML = '<option value="qwen2.5:7b">qwen2.5:7b (default)</option>';
          } else {
            models.forEach(m => {
              const opt = document.createElement('option');
              opt.value = m.name;
              opt.textContent = `${m.name} (${m.size_gb} GB - ${m.parameter_size})`;
              if (m.name === data.ollama.active_model) opt.selected = true;
              ollamaSelect.appendChild(opt);
            });
          }
        }

        // Local models container
        const localList = document.getElementById('discovered-local-models-list');
        if (localList) {
          const locals = data.local_model.discovered_models || [];
          if (locals.length === 0) {
            localList.innerHTML = '<div class="text-xs text-slate-500 p-2 italic">No compatible model checkpoints found in ./models/.</div>';
          } else {
            localList.innerHTML = locals.map(m => `
              <div class="p-2.5 rounded-lg bg-slate-900/60 border border-slate-700/60 flex items-center justify-between text-xs">
                <div>
                  <div class="font-medium text-indigo-300 font-mono">${m.name}</div>
                  <div class="text-[10px] text-slate-400">${m.architecture} | ${m.format} | ${m.size_mb} MB</div>
                </div>
                <span class="px-2 py-0.5 rounded bg-emerald-950/60 text-emerald-400 border border-emerald-800 text-[10px]">Valid Checkpoint</span>
              </div>
            `).join('');
          }
        }

        showToast('Refreshed available models.', 'info');
      }
    } catch (err) {
      showToast(`Refresh error: ${err.message}`, 'error');
    }
  },

  async testOllama() {
    const btn = document.getElementById('test-ollama-btn');
    const resultBox = document.getElementById('ollama-test-result');
    if (btn) btn.disabled = true;
    if (resultBox) {
      resultBox.classList.remove('hidden');
      resultBox.innerHTML = '<span class="text-xs text-slate-400">Testing connection to Ollama...</span>';
    }

    try {
      const res = await fetch('/api/models/test-ollama', { method: 'POST' });
      const data = await res.json();
      
      const isSuccess = data.test_generation;
      resultBox.innerHTML = `
        <div class="p-3 rounded-lg border text-xs ${isSuccess ? 'bg-emerald-950/40 border-emerald-800 text-emerald-300' : 'bg-rose-950/40 border-rose-800 text-rose-300'}">
          <div class="font-semibold mb-1">${isSuccess ? '✓ Ollama Test Passed' : '✗ Ollama Test Failed'} (${data.latency_seconds || 0}s)</div>
          <div>${data.message}</div>
          ${data.details?.sample_output ? `<div class="mt-1 font-mono text-[11px] opacity-80">Sample Output: "${data.details.sample_output}"</div>` : ''}
        </div>
      `;
    } catch (err) {
      resultBox.innerHTML = `<div class="p-3 rounded-lg border border-rose-800 bg-rose-950/40 text-rose-300 text-xs">Test failed: ${err.message}</div>`;
    } finally {
      if (btn) btn.disabled = false;
    }
  },

  async testLocalModel() {
    const btn = document.getElementById('test-local-btn');
    const resultBox = document.getElementById('local-test-result');
    if (btn) btn.disabled = true;
    if (resultBox) {
      resultBox.classList.remove('hidden');
      resultBox.innerHTML = '<span class="text-xs text-slate-400">Validating and loading local model...</span>';
    }

    try {
      const res = await fetch('/api/models/test-local', { method: 'POST' });
      const data = await res.json();
      
      const isSuccess = data.test_generation;
      resultBox.innerHTML = `
        <div class="p-3 rounded-lg border text-xs ${isSuccess ? 'bg-indigo-950/40 border-indigo-800 text-indigo-300' : 'bg-rose-950/40 border-rose-800 text-rose-300'}">
          <div class="font-semibold mb-1">${isSuccess ? '✓ Local Model Verified' : '✗ Verification Failed'} (${data.latency_seconds || 0}s)</div>
          <div>${data.message}</div>
          ${data.details?.sample_output ? `<div class="mt-1 font-mono text-[11px] opacity-80">Sample Output: "${data.details.sample_output}"</div>` : ''}
        </div>
      `;
    } catch (err) {
      resultBox.innerHTML = `<div class="p-3 rounded-lg border border-rose-800 bg-rose-950/40 text-rose-300 text-xs">Test failed: ${err.message}</div>`;
    } finally {
      if (btn) btn.disabled = false;
    }
  },

  async saveSettings() {
    const ollamaUrl = document.getElementById('settings-ollama-url')?.value.trim();
    const ollamaModel = document.getElementById('settings-ollama-model-select')?.value;

    const payload = {
      ollama_base_url: ollamaUrl || undefined,
      active_ollama_model: ollamaModel || undefined,
      active_system_profile: AppState.activeProfile
    };

    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        showToast('Settings saved successfully!', 'success');
        checkSystemHealth();
      }
    } catch (err) {
      showToast(`Failed to save settings: ${err.message}`, 'error');
    }
  },

  async createCustomProfile() {
    const name = document.getElementById('new-profile-name')?.value.trim();
    const id = document.getElementById('new-profile-id')?.value.trim();
    const desc = document.getElementById('new-profile-desc')?.value.trim();
    const prompt = document.getElementById('new-profile-prompt')?.value.trim();

    if (!name || !id || !prompt) {
      showToast('Please fill in profile name, identifier, and prompt instructions.', 'warning');
      return;
    }

    try {
      const res = await fetch('/api/settings/profile', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_id: id.toLowerCase().replace(/\s+/g, '_'),
          name: name,
          description: desc,
          prompt: prompt
        })
      });

      if (res.ok) {
        showToast(`Created profile '${name}'!`, 'success');
        closeModal('custom-profile-modal');
        this.loadSettings();
      }
    } catch (err) {
      showToast(`Error creating profile: ${err.message}`, 'error');
    }
  },

  async runModelComparison() {
    const promptInput = document.getElementById('compare-prompt-input');
    const prompt = promptInput?.value.trim() || 'Summarize the core difference between abstractive and extractive summarization in 3 concise bullet points.';
    
    const ollamaResultEl = document.getElementById('compare-ollama-output');
    const localResultEl = document.getElementById('compare-local-output');
    const spinner = document.getElementById('compare-spinner');
    const btn = document.getElementById('run-model-comparison-btn');

    if (btn) btn.disabled = true;
    if (spinner) spinner.classList.remove('hidden');

    if (ollamaResultEl) ollamaResultEl.innerHTML = '<div class="text-slate-400 italic">Streaming inference from Ollama (Qwen 2.5 7B)...</div>';
    if (localResultEl) localResultEl.innerHTML = '<div class="text-slate-400 italic">Executing inference on Local Model (Transformers)...</div>';

    const startTime = Date.now();

    try {
      // Execute Ollama Request
      const ollamaPromise = fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: prompt,
          mode: 'ollama',
          system_profile: 'general',
          stream: false
        })
      }).then(r => r.json());

      // Execute Local Model Request
      const localPromise = fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: prompt,
          mode: 'local_model',
          system_profile: 'general',
          stream: false
        })
      }).then(r => r.json());

      const [ollamaRes, localRes] = await Promise.allSettled([ollamaPromise, localPromise]);

      // Render Ollama Result
      if (ollamaResultEl) {
        if (ollamaRes.status === 'fulfilled' && !ollamaRes.value.error) {
          const resp = ollamaRes.value.response || ollamaRes.value.content || '';
          ollamaResultEl.innerHTML = window.marked ? marked.parse(resp) : resp;
        } else {
          const err = ollamaRes.status === 'rejected' ? ollamaRes.reason?.message : ollamaRes.value?.detail || 'Execution error';
          ollamaResultEl.innerHTML = `<div class="p-3 bg-rose-950/40 text-rose-300 rounded-lg text-xs">Ollama Error: ${err}</div>`;
        }
      }

      // Render Local Model Result
      if (localResultEl) {
        if (localRes.status === 'fulfilled' && !localRes.value.error) {
          const resp = localRes.value.response || localRes.value.content || '';
          localResultEl.innerHTML = window.marked ? marked.parse(resp) : resp;
        } else {
          const err = localRes.status === 'rejected' ? localRes.reason?.message : localRes.value?.detail || 'Local model error or unavailable';
          localResultEl.innerHTML = `<div class="p-3 bg-rose-950/40 text-rose-300 rounded-lg text-xs">Local Model Error: ${err}</div>`;
        }
      }

      const totalElapsed = ((Date.now() - startTime) / 1000).toFixed(2);
      showToast(`Model comparison completed in ${totalElapsed}s`, 'success');
      logActivity(`Ran dual model comparison: "${prompt.slice(0, 30)}..."`);
    } catch (err) {
      showToast(`Comparison error: ${err.message}`, 'error');
    } finally {
      if (btn) btn.disabled = false;
      if (spinner) spinner.classList.add('hidden');
    }
  }
};

window.SettingsModule = SettingsModule;
document.addEventListener('DOMContentLoaded', () => SettingsModule.init());
