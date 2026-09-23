/**
 * VOLTIX Document Upload and Management Controller (ChatGPT Redesign)
 * Handles drag-and-drop document upload, FAISS vector indexing notifications, and deletion lifecycle.
 */
const VoltixDocs = {
    init() {
        const fileInput = document.getElementById('file-upload-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.upload(e.target.files[0]);
                }
            });
        }

        const openBtn = document.getElementById('btn-open-upload');
        if (openBtn && fileInput) {
            openBtn.addEventListener('click', () => fileInput.click());
        }

        this.bindDragAndDrop();
        this.refreshSummary();
    },

    bindDragAndDrop() {
        const chatWindow = document.querySelector('main');
        if (!chatWindow) return;

        ['dragenter', 'dragover'].forEach(eventName => {
            chatWindow.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                chatWindow.classList.add('bg-neutral-800/20');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            chatWindow.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                chatWindow.classList.remove('bg-neutral-800/20');
            });
        });

        chatWindow.addEventListener('drop', (e) => {
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                this.upload(e.dataTransfer.files[0]);
            }
        });
    },

    async upload(file) {
        const container = document.getElementById('attached-file-container');
        const badge = document.getElementById('attached-file-name');
        if (container) container.classList.remove('hidden');
        if (badge) {
            badge.innerHTML = `
                <i class="fa-solid fa-spinner fa-spin text-cyan-400 text-xs"></i>
                <span class="truncate max-w-xs font-mono">Indexing ${this.escapeHtml(file.name)}...</span>
            `;
            badge.classList.remove('hidden');
        }
        VoltixUI.toast(`Indexing "${file.name}" into FAISS vector store...`, 'info', 3000);

        try {
            const res = await VoltixAPI.uploadDocument(file, VoltixUI.activeProjectId);
            if (res.document) {
                if (badge) {
                    badge.innerHTML = `
                        <i class="fa-regular fa-file text-cyan-400 text-xs"></i>
                        <span class="truncate max-w-xs font-mono">${this.escapeHtml(file.name)} (${res.document.total_chunks} chunks)</span>
                    `;
                }
                VoltixUI.toast(`Indexed "${file.name}" (${res.document.total_chunks} chunks)`, 'success', 3500);
                this.refreshSummary();
                setTimeout(() => {
                    if (container) container.classList.add('hidden');
                }, 4000);
            } else {
                if (badge) {
                    badge.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-rose-400 text-xs"></i> <span>${this.escapeHtml(res.error || 'Failed')}</span>`;
                }
                VoltixUI.toast(res.error || 'Document indexing failed', 'error');
            }
        } catch (e) {
            console.error('Document upload error:', e);
            if (badge) {
                badge.innerHTML = `<i class="fa-solid fa-triangle-exclamation text-rose-400 text-xs"></i> <span>Upload failed</span>`;
            }
            VoltixUI.toast(`Upload failed: ${e.message}`, 'error');
        }
    },

    async refreshSummary() {
        const summary = document.getElementById('doc-list-summary');
        if (!summary) return;

        try {
            const docs = await VoltixAPI.getDocuments(VoltixUI.activeProjectId);
            if (docs && docs.length > 0) {
                const totalChunks = docs.reduce((acc, d) => acc + (d.total_chunks || 0), 0);
                summary.innerHTML = `
                    <div class="flex items-center justify-between">
                        <span class="truncate font-medium flex items-center gap-1 text-neutral-200">
                            <i class="fa-regular fa-file-pdf text-cyan-400"></i> ${docs.length} Doc(s) (${totalChunks} chunks)
                        </span>
                        <button class="btn-clear-doc text-neutral-400 hover:text-rose-400 p-0.5 rounded hover:bg-white/5 transition-colors" data-id="${docs[0].id}" title="Delete latest doc">
                            <i class="fa-solid fa-trash-can text-[10px]"></i>
                        </button>
                    </div>
                `;

                const clearBtn = summary.querySelector('.btn-clear-doc');
                if (clearBtn) {
                    clearBtn.addEventListener('click', async (e) => {
                        e.stopPropagation();
                        if (confirm(`Delete document "${docs[0].filename}" and un-index its chunks?`)) {
                            await VoltixAPI.deleteDocument(docs[0].id);
                            VoltixDocs.refreshSummary();
                            VoltixUI.toast(`Deleted "${docs[0].filename}"`, 'info');
                        }
                    });
                }
            } else {
                summary.innerHTML = '<span class="text-neutral-500">No active documents.</span>';
            }
        } catch (e) {
            summary.innerHTML = '<span class="text-neutral-500">Knowledge base ready.</span>';
        }
    },

    escapeHtml(str) {
        if (typeof str !== 'string') return '';
        return str.replace(/[&<>'"]/g, 
            tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
        );
    }
};
