/**
 * VOLTIX Complete API Client
 * Wraps REST and Streaming endpoints for Projects, Conversations, Documents, Models, and Engineering Solvers.
 */
const VoltixAPI = {
    // --- Projects ---
    async getProjects() {
        const res = await fetch('/api/projects/');
        return await res.json();
    },

    async createProject(name, description = "", notes = "") {
        const res = await fetch('/api/projects/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name, description, notes })
        });
        return await res.json();
    },

    async deleteProject(projectId) {
        const res = await fetch(`/api/projects/${projectId}`, { method: 'DELETE' });
        return await res.json();
    },

    // --- Conversations ---
    async getConversations(projectId = null) {
        const url = projectId ? `/api/conversations/?project_id=${projectId}` : '/api/conversations/';
        const res = await fetch(url);
        return await res.json();
    },

    async searchConversations(query) {
        const res = await fetch(`/api/conversations/search?q=${encodeURIComponent(query)}`);
        return await res.json();
    },

    async createConversation(title = "New Conversation", model = "qwen2.5:7b", projectId = null, academicMode = "Learn") {
        const res = await fetch('/api/conversations/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title, model, project_id: projectId, academic_mode: academicMode })
        });
        return await res.json();
    },

    async getConversationDetail(convId) {
        const res = await fetch(`/api/conversations/${convId}`);
        return await res.json();
    },

    async renameConversation(convId, title) {
        const res = await fetch(`/api/conversations/${convId}`, {
            method: 'PATCH',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title })
        });
        return await res.json();
    },

    async deleteConversation(convId) {
        const res = await fetch(`/api/conversations/${convId}`, { method: 'DELETE' });
        return await res.json();
    },

    // --- Documents ---
    async getDocuments(projectId = null) {
        const url = projectId ? `/api/documents/?project_id=${projectId}` : '/api/documents/';
        const res = await fetch(url);
        return await res.json();
    },

    async uploadDocument(file, projectId = null) {
        const formData = new FormData();
        formData.append('file', file);
        if (projectId) formData.append('project_id', projectId);

        const res = await fetch('/api/documents/upload', {
            method: 'POST',
            body: formData
        });
        return await res.json();
    },

    async deleteDocument(docId) {
        const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
        return await res.json();
    },

    // --- Models & Health ---
    async getModels() {
        const res = await fetch('/api/models/');
        return await res.json();
    },

    async checkHealth() {
        try {
            const res = await fetch('/api/models/health');
            return await res.json();
        } catch (e) {
            return { status: "offline", error: e.message };
        }
    },

    // --- Engineering Tools & Solvers ---
    async calculate3Phase(params) {
        const res = await fetch('/api/tools/three-phase-power', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        return await res.json();
    },

    async calculateOhmsLaw(params) {
        const res = await fetch('/api/tools/ohms-law', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        return await res.json();
    },

    async calculateMotor(params) {
        const slipRes = await fetch('/api/tools/slip', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ns: params.ns, n: params.n })
        });
        const torqueRes = await fetch('/api/tools/motor-torque', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ power_w: params.power_w, speed_rpm: params.n })
        });
        return { slip: await slipRes.json(), torque: await torqueRes.json() };
    },

    async calculateConverter(params) {
        const res = await fetch('/api/tools/dc-dc-converter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        return await res.json();
    },

    async calculatePFCorrection(params) {
        const res = await fetch('/api/tools/power-factor-correction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        return await res.json();
    },

    async calculateBatteryEV(params) {
        const res = await fetch('/api/tools/battery-ev', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        return await res.json();
    },

    async calculateSolarPV(params) {
        const res = await fetch('/api/tools/solar-pv', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        return await res.json();
    },

    async runSymPy(operation, params) {
        let endpoint = '/api/tools/sympy/solve';
        if (operation === 'diff') endpoint = '/api/tools/sympy/diff';
        else if (operation === 'integrate') endpoint = '/api/tools/sympy/integrate';
        else if (operation === 'laplace') endpoint = '/api/tools/sympy/laplace';
        else if (operation === 'inv_laplace') endpoint = '/api/tools/sympy/inverse-laplace';
        else if (operation === 'tf') endpoint = '/api/tools/sympy/transfer-function';

        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        return await res.json();
    },

    async convertUnit(value, from_unit, to_unit) {
        const res = await fetch('/api/tools/unit-convert', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ value, from_unit, to_unit })
        });
        return await res.json();
    },

    async generateMatlab(type, params = {}) {
        let endpoint = '/api/tools/matlab/bode';
        if (type === 'step') endpoint = '/api/tools/matlab/step';
        else if (type === 'torque_speed') endpoint = '/api/tools/matlab/torque-speed';
        else if (type === 'simulink') endpoint = '/api/tools/matlab/simulink-blueprint';

        const res = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });
        return await res.json();
    }
};
