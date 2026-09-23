/**
 * VOLTIX EEE Deterministic Solvers & Calculators Module
 * Provides interactive UI handlers for 10 engineering calculation engines.
 */

const SolversModule = {
    activeForm: 'form-3phase',
    lastResult: null,

    init() {
        this.bindTabButtons();
        this.bindActionButtons();
    },

    bindTabButtons() {
        document.querySelectorAll('.solver-tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                const target = btn.dataset.target;
                if (!target) return;
                this.switchSolverPane(target, btn);
            });
        });
    },

    switchSolverPane(targetId, activeBtn) {
        this.activeForm = targetId;

        // Update tab buttons
        document.querySelectorAll('.solver-tab-btn').forEach(b => {
            b.classList.remove('active');
        });
        if (activeBtn) {
            activeBtn.classList.add('active');
        } else {
            // Find it by data-target
            const btn = document.querySelector(`.solver-tab-btn[data-target="${targetId}"]`);
            if (btn) btn.classList.add('active');
        }

        // Hide all panes & show target
        document.querySelectorAll('.solver-form-pane').forEach(p => p.classList.remove('active'));
        const pane = document.getElementById(targetId);
        if (pane) pane.classList.add('active');
    },

    bindActionButtons() {
        // 1. 3-Phase Power
        document.getElementById('btn-run-3phase')?.addEventListener('click', () => this.run3Phase());

        // 2. Ohm's Law
        document.getElementById('btn-run-ohms')?.addEventListener('click', () => this.runOhmsLaw());

        // 3. Motor Slip & Torque
        document.getElementById('btn-run-motor')?.addEventListener('click', () => this.runMotor());

        // 4. DC-DC Converter
        document.getElementById('btn-run-conv')?.addEventListener('click', () => this.runConverter());

        // 5. Power Factor Correction
        document.getElementById('btn-run-pf')?.addEventListener('click', () => this.runPF());

        // 6. Battery & EV
        document.getElementById('btn-run-battery')?.addEventListener('click', () => this.runBattery());

        // 7. Solar PV
        document.getElementById('btn-run-pv')?.addEventListener('click', () => this.runSolarPV());

        // 8. SymPy Math
        document.getElementById('btn-run-sympy')?.addEventListener('click', () => this.runSymPy());

        // 9. Unit Converter
        document.getElementById('btn-run-unit')?.addEventListener('click', () => this.runUnitConverter());

        // 10. MATLAB Generator
        document.getElementById('btn-run-matlab')?.addEventListener('click', () => this.runMatlab());

        // Copy button
        document.getElementById('btn-copy-solver')?.addEventListener('click', () => {
            if (this.lastResult) {
                const text = typeof this.lastResult === 'string' ? this.lastResult : JSON.stringify(this.lastResult, null, 2);
                navigator.clipboard.writeText(text);
                showToast('Results copied to clipboard', 'success');
            }
        });
    },

    displayResult(data, title = 'Calculation Output') {
        this.lastResult = data;
        const box = document.getElementById('solver-result-box');
        const body = document.getElementById('solver-result-body');
        if (!box || !body) return;

        box.classList.add('visible');

        if (typeof data === 'string') {
            body.innerText = data;
        } else {
            body.innerText = JSON.stringify(data, null, 2);
        }

        if (typeof logActivity === 'function') logActivity(`Executed ${title}`);
        showToast(`${title} completed`, 'success');
        box.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    },

    displayError(err, title = 'Solver Error') {
        const box = document.getElementById('solver-result-box');
        const body = document.getElementById('solver-result-body');
        if (!box || !body) return;

        box.classList.add('visible');
        body.innerText = `Error: ${err.message || err}`;
        showToast(`${title}: ${err.message || err}`, 'error');
    },

    // 1. 3-Phase Power
    async run3Phase() {
        try {
            const pVal = document.getElementById('calc-3p-power')?.value;
            const vlVal = document.getElementById('calc-3p-voltage')?.value;
            const ilVal = document.getElementById('calc-3p-current')?.value;
            const pfVal = document.getElementById('calc-3p-pf')?.value;

            const params = {
                p: pVal !== '' ? parseFloat(pVal) : null,
                vl: vlVal !== '' ? parseFloat(vlVal) : null,
                il: ilVal !== '' ? parseFloat(ilVal) : null,
                pf: pfVal !== '' ? parseFloat(pfVal) : null
            };

            const data = await VoltixAPI.calculate3Phase(params);
            this.displayResult(data, '3-Phase Power Solver');
        } catch (e) {
            this.displayError(e, '3-Phase Power');
        }
    },

    // 2. Ohm's Law
    async runOhmsLaw() {
        try {
            const vVal = document.getElementById('calc-v')?.value;
            const iVal = document.getElementById('calc-i')?.value;
            const rVal = document.getElementById('calc-r')?.value;

            const params = {
                v: vVal !== '' ? parseFloat(vVal) : null,
                i: iVal !== '' ? parseFloat(iVal) : null,
                r: rVal !== '' ? parseFloat(rVal) : null
            };

            const data = await VoltixAPI.calculateOhmsLaw(params);
            this.displayResult(data, "Ohm's Law Solver");
        } catch (e) {
            this.displayError(e, "Ohm's Law");
        }
    },

    // 3. Motor Slip & Torque
    async runMotor() {
        try {
            const f = parseFloat(document.getElementById('calc-motor-freq')?.value || 50);
            const p = parseInt(document.getElementById('calc-motor-poles')?.value || 4);
            const n = parseFloat(document.getElementById('calc-motor-speed')?.value || 1440);
            const power = parseFloat(document.getElementById('calc-motor-power')?.value || 7500);

            const ns = (120 * f) / p;
            const data = await VoltixAPI.calculateMotor({ ns, n, power_w: power });
            this.displayResult(data, 'Motor Slip & Torque Analysis');
        } catch (e) {
            this.displayError(e, 'Motor Analysis');
        }
    },

    // 4. DC-DC Converter
    async runConverter() {
        try {
            const topology = document.getElementById('calc-conv-topology')?.value || 'buck';
            const freq_hz = parseFloat(document.getElementById('calc-conv-freq')?.value || 50000);
            const vin = parseFloat(document.getElementById('calc-conv-vin')?.value || 24);
            const vout = parseFloat(document.getElementById('calc-conv-vout')?.value || 5);

            const params = { topology, freq_hz, vin, vout, iout: 2 };
            const data = await VoltixAPI.calculateConverter(params);
            this.displayResult(data, 'DC-DC Converter Design');
        } catch (e) {
            this.displayError(e, 'DC-DC Converter');
        }
    },

    // 5. Power Factor Correction
    async runPF() {
        try {
            const active_power_kw = parseFloat(document.getElementById('calc-pf-kw')?.value || 100);
            const voltage_v = parseFloat(document.getElementById('calc-pf-volt')?.value || 415);
            const initial_pf = parseFloat(document.getElementById('calc-pf-init')?.value || 0.75);
            const target_pf = parseFloat(document.getElementById('calc-pf-target')?.value || 0.98);

            const params = { active_power_kw, voltage_v, initial_pf, target_pf, freq_hz: 50 };
            const data = await VoltixAPI.calculatePFCorrection(params);
            this.displayResult(data, 'Power Factor Correction Capacitor Sizing');
        } catch (e) {
            this.displayError(e, 'PF Correction');
        }
    },

    // 6. Battery & EV
    async runBattery() {
        try {
            const voltage_v = parseFloat(document.getElementById('calc-bat-volt')?.value || 400);
            const capacity_ah = parseFloat(document.getElementById('calc-bat-ah')?.value || 100);

            const params = { voltage_v, capacity_ah, c_rate: 1.0, wh_per_km: 150 };
            const data = await VoltixAPI.calculateBatteryEV(params);
            this.displayResult(data, 'Battery EV Energy & Range Calculation');
        } catch (e) {
            this.displayError(e, 'Battery EV');
        }
    },

    // 7. Solar PV
    async runSolarPV() {
        try {
            const voc = parseFloat(document.getElementById('calc-pv-voc')?.value || 45);
            const target_kw = parseFloat(document.getElementById('calc-pv-target')?.value || 5.0);

            const params = { voc, isc: 10, vmp: 36, imp: 9.2, target_kw };
            const data = await VoltixAPI.calculateSolarPV(params);
            this.displayResult(data, 'Solar PV Array Sizing');
        } catch (e) {
            this.displayError(e, 'Solar PV');
        }
    },

    // 8. SymPy Math
    async runSymPy() {
        try {
            const op = document.getElementById('calc-sympy-op')?.value || 'solve';
            const expr = document.getElementById('calc-sympy-expr')?.value || '';

            const params = { expression: expr, variable: 't' };
            const data = await VoltixAPI.runSymPy(op, params);
            this.displayResult(data, `SymPy Symbolic ${op.toUpperCase()}`);
        } catch (e) {
            this.displayError(e, 'SymPy Math');
        }
    },

    // 9. Unit Converter
    async runUnitConverter() {
        try {
            const val = parseFloat(document.getElementById('calc-unit-val')?.value || 0);
            const from_u = document.getElementById('calc-unit-from')?.value || 'kW';
            const to_u = document.getElementById('calc-unit-to')?.value || 'W';

            const data = await VoltixAPI.convertUnit(val, from_u, to_u);
            this.displayResult(data, 'Engineering Unit Conversion');
        } catch (e) {
            this.displayError(e, 'Unit Converter');
        }
    },

    // 10. MATLAB Generator
    async runMatlab() {
        try {
            const type = document.getElementById('calc-matlab-type')?.value || 'torque_speed';
            const data = await VoltixAPI.generateMatlab(type, {});
            const scriptContent = data.script || data.blueprint || JSON.stringify(data, null, 2);
            this.displayResult(scriptContent, 'MATLAB/Simulink Generator');
        } catch (e) {
            this.displayError(e, 'MATLAB Generator');
        }
    }
};

// Note: SolversModule.init() is called by app.js DOMContentLoaded handler.
// Duplicate init guard for standalone use:
if (typeof AppState === 'undefined') {
    document.addEventListener('DOMContentLoaded', () => SolversModule.init());
}
