/**
 * VOLTIX Settings, Hyperparameters & Dual-Model Benchmark Workbench Module
 * Handles Ollama daemon connectivity, hyperparameter tuning, and concurrent comparative inference.
 */

const SettingsModule = {
    isComparing: false,

    init() {
        this.bindEvents();
        this.loadSavedPreferences();
    },

    // Called when settings modal opens (compatibility)
    onShow() {
        this.loadSavedPreferences();
    },

    bindEvents() {
        // Temperature Slider
        const tempSlider = document.getElementById('setting-temp');
        const tempVal = document.getElementById('temp-val');
        if (tempSlider && tempVal) {
            tempSlider.addEventListener('input', (e) => {
                tempVal.innerText = e.target.value;
            });
        }

        // Top-P Slider
        const toppSlider = document.getElementById('setting-topp');
        const toppVal = document.getElementById('topp-val');
        if (toppSlider && toppVal) {
            toppSlider.addEventListener('input', (e) => {
                toppVal.innerText = e.target.value;
            });
        }

        // Save Settings
        document.getElementById('save-settings-btn')?.addEventListener('click', () => this.savePreferences());

        // Test Ollama Connection
        document.getElementById('test-ollama-btn')?.addEventListener('click', () => this.testOllamaConnection());

        // Dual-Model Comparison
        document.getElementById('run-model-comparison-btn')?.addEventListener('click', () => this.runDualModelBenchmark());
    },

    loadSavedPreferences() {
        try {
            const temp = localStorage.getItem('voltix_temp') || '0.1';
            const topp = localStorage.getItem('voltix_topp') || '0.9';
            const rag = localStorage.getItem('voltix_rag') !== 'false';

            const tempSlider = document.getElementById('setting-temp');
            const tempVal = document.getElementById('temp-val');
            const toppSlider = document.getElementById('setting-topp');
            const toppVal = document.getElementById('topp-val');
            const ragCb = document.getElementById('setting-enable-rag');

            if (tempSlider) tempSlider.value = temp;
            if (tempVal) tempVal.innerText = temp;
            if (toppSlider) toppSlider.value = topp;
            if (toppVal) toppVal.innerText = topp;
            if (ragCb) ragCb.checked = rag;
        } catch (e) {}
    },

    savePreferences() {
        const temp = document.getElementById('setting-temp')?.value || '0.1';
        const topp = document.getElementById('setting-topp')?.value || '0.9';
        const rag = document.getElementById('setting-enable-rag')?.checked ?? true;

        localStorage.setItem('voltix_temp', temp);
        localStorage.setItem('voltix_topp', topp);
        localStorage.setItem('voltix_rag', rag ? 'true' : 'false');

        // Sync to AppState
        if (typeof AppState !== 'undefined') {
            AppState.temperature = parseFloat(temp);
            AppState.topP = parseFloat(topp);
            AppState.ragEnabled = rag;
        }

        showToast('Settings saved', 'success');
        if (typeof logActivity === 'function') {
            logActivity(`Updated: Temp=${parseFloat(temp).toFixed(2)}, Top-P=${parseFloat(topp).toFixed(2)}, RAG=${rag}`);
        }
    },

    async testOllamaConnection() {
        const resultBox = document.getElementById('ollama-test-result');
        const btn = document.getElementById('test-ollama-btn');

        if (btn) btn.disabled = true;
        if (resultBox) {
            resultBox.classList.remove('hidden');
            resultBox.innerHTML = '<span class="text-cyan-400 font-mono">Pinging Ollama daemon...</span>';
        }

        try {
            const data = await VoltixAPI.checkHealth();
            if (resultBox) {
                if (data.status === 'ok' || data.status === 'healthy' || data.ollama_reachable) {
                    resultBox.innerHTML = `
                        <div class="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-400">
                            <div class="font-semibold flex items-center">
                                <i class="fa-solid fa-circle-check mr-1.5"></i> Ollama Daemon Connected
                            </div>
                            <div class="text-[10px] text-[var(--color-text-secondary)] mt-1 font-mono">
                                Available Models: ${(data.available_models || data.models || ['qwen2.5:7b', 'deepseek-v4-flash:cloud']).join(', ')}
                            </div>
                        </div>
                    `;
                    showToast('Ollama service reachable and healthy', 'success');
                } else {
                    resultBox.innerHTML = `
                        <div class="p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400">
                            <div class="font-semibold flex items-center">
                                <i class="fa-solid fa-triangle-exclamation mr-1.5"></i> Service Status: ${data.status || 'Offline'}
                            </div>
                            <div class="text-[10px] text-[var(--color-text-secondary)] mt-1">
                                ${data.error || 'Check that `ollama serve` is running.'}
                            </div>
                        </div>
                    `;
                    showToast('Ollama check returned notice', 'warning');
                }
            }
        } catch (e) {
            if (resultBox) {
                resultBox.innerHTML = `
                    <div class="p-2.5 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400">
                        <div class="font-semibold flex items-center">
                            <i class="fa-solid fa-circle-xmark mr-1.5"></i> Connection Failed
                        </div>
                        <div class="text-[10px] text-[var(--color-text-secondary)] mt-1 font-mono">
                            ${e.message}
                        </div>
                    </div>
                `;
            }
            showToast(`Ollama check failed: ${e.message}`, 'error');
        } finally {
            if (btn) btn.disabled = false;
        }
    },

    // Dual-Model Side-by-Side EEE Benchmark
    async runDualModelBenchmark() {
        if (this.isComparing) return;

        const promptInput = document.getElementById('compare-prompt-input');
        const prompt = promptInput?.value.trim();
        if (!prompt) {
            showToast('Please enter a benchmark question or derivation', 'warning');
            return;
        }

        const btn = document.getElementById('run-model-comparison-btn');
        const spinner = document.getElementById('compare-spinner');
        const out1 = document.getElementById('compare-model1-output');
        const out2 = document.getElementById('compare-model2-output');

        this.isComparing = true;
        if (spinner) spinner.classList.remove('hidden');
        if (btn) btn.disabled = true;

        if (out1) out1.innerHTML = '<span class="text-cyan-400 italic font-mono">Streaming Qwen 2.5 7B derivation...</span>';
        if (out2) out2.innerHTML = '<span class="text-purple-400 italic font-mono">Streaming DeepSeek V4 Flash derivation...</span>';

        const streamModel = async (modelName, targetElement) => {
            try {
                const response = await fetch('/api/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: prompt,
                        model: modelName,
                        academic_mode: 'Exam'
                    })
                });

                if (!response.ok) throw new Error(`HTTP ${response.status}`);

                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullText = '';
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop();

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const raw = line.slice(6).trim();
                            if (raw === '[DONE]') continue;
                            try {
                                const parsed = JSON.parse(raw);
                                if (parsed.type === 'token' && parsed.content) {
                                    fullText += parsed.content;
                                    if (targetElement) {
                                        targetElement.innerHTML = typeof marked !== 'undefined' ? marked.parse(fullText) : fullText;
                                    }
                                }
                            } catch (e) {}
                        }
                    }
                }

                if (targetElement && typeof MathRenderer !== 'undefined') {
                    MathRenderer.render(targetElement);
                }
                if (targetElement && typeof CodeHighlighter !== 'undefined') {
                    CodeHighlighter.highlight(targetElement);
                }
            } catch (e) {
                if (targetElement) {
                    targetElement.innerHTML = `<span class="text-rose-400">Benchmark error: ${e.message}</span>`;
                }
            }
        };

        try {
            await Promise.all([
                streamModel('qwen2.5:7b', out1),
                streamModel('deepseek-v4-flash:cloud', out2)
            ]);
            showToast('Dual-model benchmark completed', 'success');
            if (typeof logActivity === 'function') {
                logActivity(`Ran benchmark on '${prompt.slice(0, 30)}...'`);
            }
        } finally {
            this.isComparing = false;
            if (spinner) spinner.classList.add('hidden');
            if (btn) btn.disabled = false;
        }
    }
};

// Note: SettingsModule.init() is called by app.js DOMContentLoaded handler.
if (typeof AppState === 'undefined') {
    document.addEventListener('DOMContentLoaded', () => SettingsModule.init());
}
