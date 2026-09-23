/**
 * VOLTIX KaTeX Math Renderer Module
 */
const VoltixMath = {
    render(element) {
        if (!element || typeof renderMathInElement !== 'function') return;
        try {
            renderMathInElement(element, {
                delimiters: [
                    { left: '$$', right: '$$', display: true },
                    { left: '$', right: '$', display: false },
                    { left: '\\(', right: '\\)', display: false },
                    { left: '\\[', right: '\\]', display: true }
                ],
                throwOnError: false
            });
        } catch (e) {
            console.warn('KaTeX rendering error:', e);
        }
    }
};
