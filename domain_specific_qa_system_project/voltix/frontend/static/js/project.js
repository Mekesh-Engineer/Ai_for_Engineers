/**
 * VOLTIX Project Workspaces & AI Circuit/Code Debugger Module
 * Manages isolated engineering project domains and intelligent runtime diagnostics.
 */

const ProjectModule = {
    projects: [],
    isDebugging: false,

    init() {
        this.bindEvents();
        this.loadProjects();
    },

    onShow() {
        this.loadProjects();
    },

    bindEvents() {
        // Create Project
        document.getElementById('btn-create-project')?.addEventListener('click', () => this.createProject());

        // Run Debugger
        document.getElementById('run-debugger-btn')?.addEventListener('click', () => this.runDebugger());
    },

    async loadProjects() {
        try {
            const data = await VoltixAPI.getProjects();
            this.projects = Array.isArray(data) ? data : [];
            this.renderProjectsList();
        } catch (e) {
            console.error('Error loading workspaces:', e);
        }
    },

    renderProjectsList() {
        const container = document.getElementById('projects-list-container');
        if (!container) return;

        if (this.projects.length === 0) {
            container.innerHTML = `
                <div class="p-2 rounded-lg bg-[var(--color-surface)] border border-dashed border-[var(--color-border)] text-[11px] text-[var(--color-text-muted)] italic text-center">
                    Default Workspace Active
                </div>
            `;
            return;
        }

        container.innerHTML = this.projects.map(p => {
            const isActive = AppState.activeProject === p.id;
            return `
                <div onclick="ProjectModule.switchWorkspace('${p.id}', '${p.name}')" class="p-2.5 rounded-xl border transition cursor-pointer flex items-center justify-between text-xs ${isActive ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-300' : 'bg-[var(--color-surface)] border-[var(--color-border)] hover:bg-[var(--color-elevated)] text-[var(--color-text-primary)]'}">
                    <div class="truncate flex-1 pr-2">
                        <div class="font-medium truncate flex items-center">
                            <i class="fa-solid fa-folder-tree mr-1.5 text-xs ${isActive ? 'text-cyan-400' : 'text-[var(--color-text-muted)]'}"></i>
                            <span class="truncate">${p.name}</span>
                        </div>
                        <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5 truncate">${p.description || 'No description'}</div>
                    </div>
                    <div class="flex items-center space-x-1.5 flex-shrink-0">
                        ${isActive ? '<span class="text-[9px] px-1.5 py-0.2 rounded bg-cyan-500/20 text-cyan-300 border border-cyan-500/30 font-mono">Active</span>' : ''}
                        <button onclick="event.stopPropagation(); ProjectModule.deleteProject('${p.id}', '${p.name}')" class="text-[var(--color-text-muted)] hover:text-rose-400 p-1 transition" title="Delete workspace">
                            <i class="fa-solid fa-trash text-xs"></i>
                        </button>
                    </div>
                </div>
            `;
        }).join('');
    },

    async createProject() {
        const nameInput = document.getElementById('input-new-project-name');
        const descInput = document.getElementById('input-new-project-desc');
        const name = nameInput?.value.trim();
        const desc = descInput?.value.trim() || '';

        if (!name) {
            showToast('Please enter a workspace name', 'warning');
            return;
        }

        try {
            const data = await VoltixAPI.createProject(name, desc);
            if (data.error) throw new Error(data.error);
            showToast(`Workspace '${name}' created`, 'success');
            if (nameInput) nameInput.value = '';
            if (descInput) descInput.value = '';

            await this.loadProjects();
            if (data.id) {
                this.switchWorkspace(data.id, data.name);
            }
        } catch (e) {
            showToast(`Error creating workspace: ${e.message}`, 'error');
        }
    },

    switchWorkspace(id, name) {
        AppState.activeProjectId = id;
        AppState.activeProject = id;
        AppState.activeProjectName = name;

        this.renderProjectsList();
        showToast(`Workspace: ${name}`, 'info', 2000);

        // Reload sidebar project list
        if (typeof loadProjectsSidebar === 'function') loadProjectsSidebar();

        // Reload scoped resources
        if (typeof FilesModule !== 'undefined') FilesModule.loadDocuments();
        if (typeof ChatModule !== 'undefined') ChatModule.loadConversations?.();
    },

    async deleteProject(id, name) {
        if (!confirm(`Delete workspace '${name}' and its scoped documents?`)) return;
        try {
            await VoltixAPI.deleteProject(id);
            showToast(`Deleted workspace '${name}'`, 'info');
            if (AppState.activeProjectId === id || AppState.activeProject === id) {
                AppState.activeProjectId = null;
                AppState.activeProject = null;
                const badge = document.getElementById('badge-active-project');
                if (badge) badge.innerText = 'Default Workspace';
            }
            await this.loadProjects();
            if (typeof FilesModule !== 'undefined') {
                FilesModule.loadDocuments();
            }
            if (typeof ChatModule !== 'undefined') {
                ChatModule.loadConversations?.();
            }
        } catch (e) {
            showToast(`Error deleting workspace: ${e.message}`, 'error');
        }
    },

    // AI Circuit & Code Error Diagnostician
    async runDebugger() {
        if (this.isDebugging) return;

        const errorMsg = document.getElementById('debug-error-msg')?.value.trim();
        const framework = document.getElementById('debug-framework-select')?.value || 'matlab';
        const stackTrace = document.getElementById('debug-stack-trace')?.value.trim();

        if (!errorMsg && !stackTrace) {
            showToast('Please enter an error symptom or paste a stack trace', 'warning');
            return;
        }

        const spinner = document.getElementById('debugger-spinner');
        const resultCard = document.getElementById('debugger-result-card');
        const output = document.getElementById('debugger-output');
        const btn = document.getElementById('run-debugger-btn');

        this.isDebugging = true;
        if (spinner) spinner.classList.remove('hidden');
        if (btn) btn.disabled = true;
        if (resultCard) resultCard.classList.remove('hidden');
        if (output) output.innerHTML = '<span class="text-cyan-400 italic">Analyzing stack trace and generating root cause diagnosis...</span>';

        const prompt = `[DIAGNOSTIC TASK]
Environment/Framework: ${framework.toUpperCase()}
Error Symptom: ${errorMsg || 'None specified'}

Code Snippet / Stack Trace:
\`\`\`
${stackTrace || 'No stack trace provided'}
\`\`\`

Please provide:
1. **Root Cause Analysis**: Why this error occurred.
2. **Mathematical / Circuit / Syntax Correction**: The exact corrected formulas or code.
3. **Prevention & Best Practices**: Recommendations for avoiding similar pitfalls in MATLAB/Simulink/Python.`;

        try {
            const response = await fetch('/api/chat/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: prompt,
                    model: AppState.activeModel === 'auto' ? 'qwen2.5:7b' : AppState.activeModel,
                    academic_mode: 'Practice'
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
                                if (output) {
                                    output.innerHTML = typeof marked !== 'undefined' ? marked.parse(fullText) : fullText;
                                }
                            }
                        } catch (e) {}
                    }
                }
            }

            if (output && typeof MathRenderer !== 'undefined') {
                MathRenderer.render(output);
            }
            if (output && typeof CodeHighlighter !== 'undefined') {
                CodeHighlighter.highlight(output);
            }

            showToast('Diagnosis completed', 'success');
            if (typeof logActivity === 'function') {
                logActivity(`Ran AI Diagnostician for ${framework.toUpperCase()}`);
            }
        } catch (e) {
            if (output) output.innerHTML = `<span class="text-rose-400">Diagnosis error: ${e.message}</span>`;
            showToast(`Diagnosis failed: ${e.message}`, 'error');
        } finally {
            this.isDebugging = false;
            if (spinner) spinner.classList.add('hidden');
            if (btn) btn.disabled = false;
        }
    }
};

// Note: ProjectModule.init() is called by app.js DOMContentLoaded handler.
if (typeof AppState === 'undefined') {
    document.addEventListener('DOMContentLoaded', () => ProjectModule.init());
}
