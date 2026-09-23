/**
 * Local LLM Studio - Settings, Models Configuration & Dual-Model Comparison Workbench
 * Experiment 8: Automated Grammar Error Correction & Text Rewriting
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
    // Mode radio toggle buttons
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
    for (const [key, profile] of Object.entries(profiles || {})) {
      const isDefault = (key === (this.settingsData?.settings?.system_profile?.active_profile || 'standard_corrector'));
      const card = document.createElement('div');
      card.className = `glass-panel p-3.5 rounded-xl flex flex-col justify-between space-y-2 border ${isDefault ? 'border-indigo-500/40' : 'border-[var(--color-border)]'}`;
      card.innerHTML = `
        <div>
          <div class="flex items-center justify-between mb-1">
            <h4 class="font-medium text-xs text-[var(--color-text-primary)]">${profile.name}</h4>
            ${isDefault ? '<span class="text-[9px] bg-indigo-500/10 text-indigo-400 px-1.5 py-0.5 rounded border border-indigo-500/20 font-mono">Active</span>' : ''}
          </div>
          <p class="text-[11px] text-[var(--color-text-secondary)] line-clamp-2">${profile.description || ''}</p>
          <div class="mt-2 p-2 rounded bg-[var(--color-canvas)] text-[10px] font-mono text-[var(--color-text-muted)] line-clamp-3 border border-[var(--color-border)]">
            ${profile.prompt || ''}
          </div>
        </div>
        <div class="flex justify-end pt-1">
          <button onclick="SettingsModule.selectProfile('${key}')" class="text-xs px-2.5 py-1 rounded bg-[var(--color-elevated)] hover:bg-indigo-600 hover:text-white transition font-medium text-[var(--color-text-secondary)]">Use Profile</button>
        </div>
      `;
      container.appendChild(card);
    }
  },

  async selectProfile(profileKey) {
    AppState.activeProfile = profileKey;
    const select = document.getElementById('chat-profile-select');
    if (select) select.value = profileKey;
    showToast(`Active profile set to: ${profileKey}`, 'success');
  },

  async testOllama() {
    const output = document.getElementById('ollama-test-result') || document.getElementById('ollama-test-output');
    if (output) {
      output.classList.remove('hidden');
      output.innerHTML = `<div class="text-xs text-indigo-400 flex items-center"><span class="animate-spin rounded-full h-3 w-3 border-b-2 border-indigo-400 mr-2"></span>Testing Ollama service connectivity...</div>`;
    }

    try {
      const res = await fetch('/api/models/test-ollama', { method: 'POST' });
      const data = await res.json();
      
      if (output) {
        output.innerHTML = `
          <div class="p-2.5 rounded-lg border text-xs ${data.test_generation ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300' : 'bg-amber-500/10 border-amber-500/30 text-amber-300'}">
            <div class="font-semibold">${data.message}</div>
            <div class="text-[11px] mt-1 font-mono text-[var(--color-text-muted)]">Latency: ${data.latency_seconds || 0}s | Reachable: ${data.ollama_reachable} | Model: ${data.model_present}</div>
          </div>
        `;
      }
    } catch (err) {
      if (output) {
        output.innerHTML = `<div class="p-2.5 bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-lg text-xs">Test failed: ${err.message}</div>`;
      }
    }
  },

  async refreshModels() {
    const select = document.getElementById('settings-ollama-model-select');
    const localList = document.getElementById('discovered-local-models-list') || document.getElementById('local-models-list');
    const ollamaList = document.getElementById('ollama-models-list');

    try {
      const res = await fetch('/api/models/list');
      if (res.ok) {
        const data = await res.json();
        
        // Ollama Select & List
        if (select && data.ollama_models) {
          select.innerHTML = '';
          data.ollama_models.forEach(m => {
            const opt = document.createElement('option');
            opt.value = m;
            opt.textContent = m;
            if (m.includes('qwen2.5:7b') || m === AppState.activeOllamaModel) opt.selected = true;
            select.appendChild(opt);
          });
        }

        if (ollamaList && data.ollama_models) {
          ollamaList.innerHTML = data.ollama_models.map(m => `
            <div class="flex items-center justify-between p-2 rounded-lg bg-[var(--color-canvas)] border border-[var(--color-border)] text-xs font-mono text-indigo-300">
              <span>${m}</span>
              <span class="text-[10px] text-emerald-400 font-sans">Ready</span>
            </div>
          `).join('');
        }

        // Local Models List
        if (localList && data.local_models) {
          if (data.local_models.length === 0) {
            localList.innerHTML = `<div class="text-xs text-[var(--color-text-muted)] italic p-2">No model weights detected in ./models/.</div>`;
          } else {
            localList.innerHTML = data.local_models.map(m => `
              <div class="p-2 rounded-lg bg-[var(--color-canvas)] border border-[var(--color-border)] text-xs font-mono text-violet-300">
                <div class="font-medium truncate">${m.name || m.path}</div>
                <div class="text-[10px] text-[var(--color-text-muted)]">${m.format || 'Safetensors / PyTorch'} • ${(m.size_bytes ? (m.size_bytes / (1024*1024)).toFixed(1) + ' MB' : 'Local Checkpoint')}</div>
              </div>
            `).join('');
          }
        }

        showToast('Refreshed local model checkpoints and Ollama catalog.', 'info', 2000);
      }
    } catch (err) {
      console.warn('Failed to refresh models:', err);
    }
  },

  async testLocalModel() {
    const output = document.getElementById('local-test-result') || document.getElementById('local-test-output');
    if (output) {
      output.classList.remove('hidden');
      output.innerHTML = `<div class="text-xs text-violet-400 flex items-center"><span class="animate-spin rounded-full h-3 w-3 border-b-2 border-violet-400 mr-2"></span>Validating PyTorch / Transformers model checkpoint...</div>`;
    }

    try {
      const res = await fetch('/api/models/test-local', { method: 'POST' });
      const data = await res.json();
      
      if (output) {
        output.innerHTML = `
          <div class="p-2.5 rounded-lg border text-xs ${data.model_valid ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300' : 'bg-amber-500/10 border-amber-500/30 text-amber-300'}">
            <div class="font-semibold">${data.message}</div>
            <div class="text-[11px] mt-1 font-mono text-[var(--color-text-muted)]">Latency: ${data.latency_seconds || 0}s | Detected: ${data.model_detected} | Output: ${data.sample_output || 'N/A'}</div>
          </div>
        `;
      }
    } catch (err) {
      if (output) {
        output.innerHTML = `<div class="p-2.5 bg-rose-500/10 border border-rose-500/30 text-rose-300 rounded-lg text-xs">Test failed: ${err.message}</div>`;
      }
    }
  },

  async runModelComparison() {
    const input = document.getElementById('compare-prompt-input') || document.getElementById('model-comparison-input');
    const prompt = input ? input.value.trim() : '';
    const spinner = document.getElementById('compare-spinner');
    const ollamaOut = document.getElementById('compare-ollama-output') || document.getElementById('comparison-ollama-output');
    const localOut = document.getElementById('compare-local-output') || document.getElementById('comparison-local-output');

    if (!prompt) {
      showToast('Please enter an evaluation sentence to compare.', 'warning');
      return;
    }

    if (spinner) spinner.classList.remove('hidden');
    if (ollamaOut) ollamaOut.innerHTML = `<div class="flex items-center text-xs text-indigo-400"><span class="animate-spin rounded-full h-3 w-3 border-b-2 border-indigo-400 mr-2"></span>Inferring on Ollama Qwen 2.5 7B...</div>`;
    if (localOut) localOut.innerHTML = `<div class="flex items-center text-xs text-violet-400"><span class="animate-spin rounded-full h-3 w-3 border-b-2 border-violet-400 mr-2"></span>Inferring on Project Local Seq2Seq Model...</div>`;

    try {
      const res = await fetch('/api/models/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: prompt })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Comparison failed');
      }

      const result = await res.json();
      
      if (ollamaOut) {
        const ollamaText = result.ollama?.text || result.ollama_output || 'No response from Ollama.';
        const ollamaLat = result.ollama?.duration || result.ollama_latency || '0';
        ollamaOut.innerHTML = `
          <div class="space-y-2">
            <div>${(typeof marked !== 'undefined') ? marked.parse(ollamaText) : ollamaText}</div>
            <div class="text-[10px] text-[var(--color-text-muted)] pt-1 border-t border-[var(--color-border)] font-mono flex justify-between">
              <span>Latency: ${ollamaLat}s</span>
              <span>Model: qwen2.5:7b</span>
            </div>
          </div>
        `;
      }

      if (localOut) {
        const localText = result.local_model?.text || result.local_output || 'No response from Local Model.';
        const localLat = result.local_model?.duration || result.local_latency || '0';
        localOut.innerHTML = `
          <div class="space-y-2">
            <div>${(typeof marked !== 'undefined') ? marked.parse(localText) : localText}</div>
            <div class="text-[10px] text-[var(--color-text-muted)] pt-1 border-t border-[var(--color-border)] font-mono flex justify-between">
              <span>Latency: ${localLat}s</span>
              <span>Model: Local Seq2Seq</span>
            </div>
          </div>
        `;
      }

      logActivity('Executed Dual-Model Side-by-Side Comparison Benchmark');
      showToast('Dual-Model comparison completed successfully!', 'success');
    } catch (err) {
      showToast(`Comparison error: ${err.message}`, 'error');
      if (ollamaOut) ollamaOut.innerText = `Error: ${err.message}`;
      if (localOut) localOut.innerText = `Error: ${err.message}`;
    } finally {
      if (spinner) spinner.classList.add('hidden');
    }
  },

  async createCustomProfile() {
    const nameInput = document.getElementById('new-profile-name');
    const idInput = document.getElementById('new-profile-id');
    const descInput = document.getElementById('new-profile-desc');
    const promptInput = document.getElementById('new-profile-prompt');

    const name = nameInput ? nameInput.value.trim() : '';
    const id = idInput ? idInput.value.trim().toLowerCase().replace(/\s+/g, '_') : '';
    const desc = descInput ? descInput.value.trim() : '';
    const prompt = promptInput ? promptInput.value.trim() : '';

    if (!name || !id || !prompt) {
      showToast('Please fill out all required profile fields.', 'warning');
      return;
    }

    try {
      const res = await fetch('/api/settings/profiles', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          profile_id: id,
          name: name,
          description: desc,
          prompt: prompt
        })
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Failed to save profile');
      }

      showToast(`Created profile '${name}'!`, 'success');
      closeModal('custom-profile-modal');
      this.loadSettings();
      if (window.ChatModule) window.ChatModule.loadProfiles();
    } catch (err) {
      showToast(`Error creating profile: ${err.message}`, 'error');
    }
  },

  async saveSettings() {
    const urlInput = document.getElementById('settings-ollama-url');
    const modelSelect = document.getElementById('settings-ollama-model-select');

    try {
      const payload = {
        ollama_url: urlInput ? urlInput.value.trim() : undefined,
        ollama_model: modelSelect ? modelSelect.value : undefined
      };

      const res = await fetch('/api/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (res.ok) {
        showToast('Settings saved successfully!', 'success');
        checkSystemHealth();
      } else {
        showToast('Failed to save settings.', 'error');
      }
    } catch (err) {
      showToast(`Save error: ${err.message}`, 'error');
    }
  }
};

window.SettingsModule = SettingsModule;
