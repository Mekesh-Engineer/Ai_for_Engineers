/**
 * VOLTIX RAG Knowledge Base & Document Management Module
 * Manages FAISS vector document indexing, drag-and-drop ingestion, and document preview.
 */

const FilesModule = {
    documents: [],
    selectedDoc: null,

    init() {
        this.bindEvents();
        this.loadDocuments();
    },

    bindEvents() {
        const dropZone = document.getElementById('file-drop-zone');
        const fileInput = document.getElementById('file-input');

        if (dropZone && fileInput) {
            dropZone.addEventListener('click', () => fileInput.click());

            dropZone.addEventListener('dragover', (e) => {
                e.preventDefault();
                dropZone.classList.add('border-cyan-500', 'bg-cyan-950/20');
            });

            dropZone.addEventListener('dragleave', () => {
                dropZone.classList.remove('border-cyan-500', 'bg-cyan-950/20');
            });

            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('border-cyan-500', 'bg-cyan-950/20');
                if (e.dataTransfer.files.length > 0) {
                    this.handleFileUpload(e.dataTransfer.files[0]);
                }
            });

            fileInput.addEventListener('change', (e) => {
                if (e.target.files.length > 0) {
                    this.handleFileUpload(e.target.files[0]);
                    fileInput.value = '';
                }
            });
        }

        // Global Drag Overlay (new ID: 'drag-overlay')
        const globalOverlay = document.getElementById('drag-overlay') ||
                              document.getElementById('global-drag-overlay');
        if (globalOverlay) {
            window.addEventListener('dragenter', (e) => {
                if (e.dataTransfer.types && Array.from(e.dataTransfer.types).includes('Files')) {
                    globalOverlay.classList.add('active');
                }
            });

            globalOverlay.addEventListener('dragover', (e) => {
                e.preventDefault();
            });

            globalOverlay.addEventListener('dragleave', (e) => {
                if (e.relatedTarget === null) {
                    globalOverlay.classList.remove('active');
                }
            });

            globalOverlay.addEventListener('drop', (e) => {
                e.preventDefault();
                globalOverlay.classList.remove('active');
                if (e.dataTransfer.files.length > 0) {
                    this.handleFileUpload(e.dataTransfer.files[0]);
                }
            });
        }
    },

    classifyFile(filename) {
        const ext = (filename || '').split('.').pop().toLowerCase();
        if (['pdf'].includes(ext)) return { label: 'PDF Document', badge: '📕 PDF', color: 'text-rose-400 bg-rose-500/10 border-rose-500/30' };
        if (['docx', 'doc'].includes(ext)) return { label: 'Word Document', badge: '📘 DOCX', color: 'text-blue-400 bg-blue-500/10 border-blue-500/30' };
        if (['py', 'm', 'c', 'cpp'].includes(ext)) return { label: 'Code Script', badge: '💻 Code', color: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30' };
        if (['csv', 'xlsx'].includes(ext)) return { label: 'Dataset', badge: '📊 CSV', color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/30' };
        return { label: 'Text File', badge: '📝 Text', color: 'text-amber-400 bg-amber-500/10 border-amber-500/30' };
    },

    onShow() {
        this.loadDocuments();
    },

    async handleFileUpload(file) {
        showToast(`Uploading and vectorizing '${file.name}'...`, 'info');
        try {
            const projectId = AppState.activeProjectId || AppState.activeProject || null;
            const data = await VoltixAPI.uploadDocument(file, projectId);
            if (data.error) {
                throw new Error(data.error);
            }
            showToast(`Indexed '${file.name}' (${data.chunk_count || 1} vector chunks)`, 'success');
            if (typeof logActivity === 'function') {
                logActivity(`Indexed RAG document: ${file.name}`);
            }
            await this.loadDocuments();
        } catch (e) {
            showToast(`Upload failed: ${e.message}`, 'error');
        }
    },

    async loadDocuments() {
        try {
            const projectId = AppState.activeProjectId || AppState.activeProject || null;
            const docs = await VoltixAPI.getDocuments(projectId);
            this.documents = Array.isArray(docs) ? docs : [];
            this.renderDocumentList();
            this.updateBadges();
        } catch (e) {
            console.error('Error fetching documents:', e);
        }
    },

    renderDocumentList() {
        const container = document.getElementById('uploaded-files-history-list');
        const drawerContainer = document.getElementById('context-files-list');

        if (container) {
            if (this.documents.length === 0) {
                container.innerHTML = '<div class="text-[11px] text-[var(--color-text-muted)] p-2 italic">No documents indexed in this workspace.</div>';
            } else {
                container.innerHTML = this.documents.map((doc, idx) => {
                    const cls = this.classifyFile(doc.filename);
                    const isSelected = this.selectedDoc && this.selectedDoc.id === doc.id;
                    return `
                        <div onclick="FilesModule.selectDocument(${idx})" class="p-2.5 rounded-xl border transition cursor-pointer flex items-center justify-between text-xs ${isSelected ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-300' : 'bg-[var(--color-surface)] border-[var(--color-border)] hover:bg-[var(--color-elevated)] text-[var(--color-text-primary)]'}">
                            <div class="truncate flex-1 pr-2">
                                <div class="font-medium truncate">${doc.filename}</div>
                                <div class="text-[10px] text-[var(--color-text-muted)] mt-0.5">${doc.chunk_count || 1} chunks • ${(doc.file_size / 1024).toFixed(1)} KB</div>
                            </div>
                            <div class="flex items-center space-x-1.5 flex-shrink-0">
                                <span class="text-[9px] px-1.5 py-0.5 rounded border ${cls.color} font-mono">${cls.badge}</span>
                                <button onclick="event.stopPropagation(); FilesModule.deleteDocument('${doc.id}', '${doc.filename}')" class="text-[var(--color-text-muted)] hover:text-rose-400 p-1 transition" title="Delete document">
                                    <i class="fa-solid fa-trash text-xs"></i>
                                </button>
                            </div>
                        </div>
                    `;
                }).join('');
            }
        }

        if (drawerContainer) {
            if (this.documents.length === 0) {
                drawerContainer.innerHTML = '<div class="text-[11px] text-[var(--color-text-muted)] col-span-2 italic">No workspace documents attached.</div>';
            } else {
                drawerContainer.innerHTML = this.documents.map(doc => `
                    <div class="flex items-center space-x-1.5 p-1.5 rounded-lg bg-[var(--color-elevated)] border border-[var(--color-border)] text-xs truncate">
                        <i class="fa-solid fa-file-lines text-cyan-400 text-[10px]"></i>
                        <span class="truncate flex-1 font-mono">${doc.filename}</span>
                        <span class="text-[9px] text-[var(--color-text-muted)]">${(doc.file_size / 1024).toFixed(1)} KB</span>
                    </div>
                `).join('');
            }
        }
    },

    selectDocument(idx) {
        const doc = this.documents[idx];
        if (!doc) return;
        this.selectedDoc = doc;
        this.renderDocumentList();

        const preview = document.getElementById('file-content-preview');
        if (preview) {
            preview.innerText = `Document: ${doc.filename}
Document ID: ${doc.id}
File Size: ${(doc.file_size / 1024).toFixed(2)} KB
Chunks Indexed: ${doc.chunk_count || 1}
Project Scope: ${doc.project_id || 'Default Workspace'}
Indexed At: ${doc.created_at || 'Just now'}

Vector Embedding: Dense Sentence-Transformers
FAISS Index Status: Active and Grounded for Q&A

Snippet / Content Preview:
${doc.content_preview || doc.summary || 'Content vectorized and indexed into FAISS knowledge base.'}`;
        }
    },

    async deleteDocument(docId, filename) {
        if (!confirm(`Delete indexed document '${filename}'?`)) return;
        try {
            await VoltixAPI.deleteDocument(docId);
            showToast(`Deleted document '${filename}'`, 'info');
            if (this.selectedDoc && this.selectedDoc.id === docId) {
                this.selectedDoc = null;
                const preview = document.getElementById('file-content-preview');
                if (preview) preview.innerText = 'Upload a document to inspect its extracted chunks and FAISS indexing stats.';
            }
            await this.loadDocuments();
        } catch (e) {
            showToast(`Error deleting document: ${e.message}`, 'error');
        }
    },

    updateBadges() {
        const sidebarCount = document.getElementById('sidebar-project-docs-count');
        const inspectorCount = document.getElementById('inspector-files-count');
        const contextBadge = document.getElementById('context-count-badge');

        const countText = `${this.documents.length} docs`;
        if (sidebarCount) sidebarCount.innerText = countText;
        if (inspectorCount) inspectorCount.innerText = `${this.documents.length} docs indexed`;
        if (contextBadge) {
            contextBadge.innerText = this.documents.length;
            contextBadge.classList.toggle('hidden', this.documents.length === 0);
        }
    }
};

// Note: FilesModule.init() is called by app.js DOMContentLoaded handler.
if (typeof AppState === 'undefined') {
    document.addEventListener('DOMContentLoaded', () => FilesModule.init());
}
