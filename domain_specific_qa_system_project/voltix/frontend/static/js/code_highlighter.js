/**
 * VOLTIX Highlight.js Code Highlighting & Copy Button Module
 */
const VoltixCode = {
    highlight(container) {
        if (!container || typeof hljs === 'undefined') return;
        container.querySelectorAll('pre code').forEach((block) => {
            if (!block.dataset.highlighted) {
                hljs.highlightElement(block);
                block.dataset.highlighted = "true";
                this.addCopyButton(block.parentElement);
            }
        });
    },

    addCopyButton(preElement) {
        if (preElement.querySelector('.copy-code-btn')) return;
        preElement.style.position = 'relative';

        const btn = document.createElement('button');
        btn.className = 'copy-code-btn absolute top-2 right-2 px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-slate-300 text-[10px] rounded border border-slate-700 font-mono transition-colors';
        btn.innerHTML = '<i class="fa-regular fa-copy mr-1"></i>Copy';

        btn.addEventListener('click', () => {
            const codeText = preElement.querySelector('code').innerText;
            navigator.clipboard.writeText(codeText);
            btn.innerHTML = '<i class="fa-solid fa-check text-green-400 mr-1"></i>Copied!';
            setTimeout(() => {
                btn.innerHTML = '<i class="fa-regular fa-copy mr-1"></i>Copy';
            }, 2000);
        });

        preElement.appendChild(btn);
    }
};
