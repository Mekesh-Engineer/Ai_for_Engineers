from typing import List, Dict, Any

class MatlabScriptGenerator:
    """Synthesizes valid, executable MATLAB scripts and comprehensive Simulink modeling blueprints."""

    @staticmethod
    def generate_bode_plot_script(num: List[float], den: List[float], title: str = "Bode Plot Analysis") -> str:
        num_str = " ".join(map(str, num))
        den_str = " ".join(map(str, den))

        script = f"""%% VOLTIX MATLAB Code Generator — {title}
clc; clear; close all;

%% Define Transfer Function Coefficients
num = [{num_str}];
den = [{den_str}];
sys = tf(num, den);

%% Frequency Response & Stability Margins
figure('Color', 'w');
bode(sys);
grid on;
title('{title}');
[gm, pm, wcg, wcp] = margin(sys);

fprintf('--- Stability Margins ---\\n');
fprintf('Gain Margin: %.2f dB at %.2f rad/s\\n', 20*log10(gm), wcg);
fprintf('Phase Margin: %.2f deg at %.2f rad/s\\n', pm, wcp);
"""
        return script.strip()

    @staticmethod
    def generate_step_response_script(num: List[float], den: List[float], title: str = "Step Response Analysis") -> str:
        num_str = " ".join(map(str, num))
        den_str = " ".join(map(str, den))

        script = f"""%% VOLTIX MATLAB Code Generator — {title}
clc; clear; close all;

num = [{num_str}];
den = [{den_str}];
sys = tf(num, den);

figure('Color', 'w');
step(sys);
grid on;
title('{title}');

info = stepinfo(sys);
fprintf('--- Step Response Performance Metrics ---\\n');
fprintf('Rise Time: %.4f s\\n', info.RiseTime);
fprintf('Settling Time: %.4f s\\n', info.SettlingTime);
fprintf('Peak Time: %.4f s\\n', info.PeakTime);
fprintf('Percentage Overshoot: %.2f %%\\n', info.Overshoot);
"""
        return script.strip()

    @staticmethod
    def generate_induction_motor_torque_speed_script(
        v_phase: float = 230.0,
        f: float = 50.0,
        poles: int = 4,
        r1: float = 0.641,
        x1: float = 1.106,
        r2: float = 0.332,
        x2: float = 0.464
    ) -> str:
        script = f"""%% VOLTIX MATLAB Code Generator — Induction Motor Torque-Speed Characteristic
clc; clear; close all;

%% Motor Parameters (Equivalent Circuit)
V_phase = {v_phase};      % Phase Voltage (V)
f = {f};                  % Stator Frequency (Hz)
P = {poles};              % Number of Poles
R1 = {r1};               % Stator Resistance (Ohm)
X1 = {x1};               % Stator Leakage Reactance (Ohm)
R2 = {r2};               % Referred Rotor Resistance (Ohm)
X2 = {x2};               % Referred Rotor Reactance (Ohm)

%% Synchronous Speed Calculations
Ns = (120 * f) / P;       % Synchronous Speed in RPM
ws_sync = 2 * pi * Ns / 60;  % Synchronous Angular Velocity in rad/s

%% Slip Vector (from 1 down to 0)
s = linspace(1, 0.001, 500);
Nr = (1 - s) * Ns;        % Rotor Speed in RPM

%% Thevenin Equivalent & Torque Calculation
R_th = R1;
X_th = X1;
V_th = V_phase;

T = zeros(size(s));
for i = 1:length(s)
    R2_eff = R2 / s(i);
    I2 = V_th / sqrt((R_th + R2_eff)^2 + (X_th + X2)^2);
    T(i) = (3 * (I2^2) * R2_eff) / ws_sync;
end

%% Plot Characteristics
figure('Color', 'w');
plot(Nr, T, 'b-', 'LineWidth', 2);
grid on;
xlabel('Rotor Speed (RPM)', 'FontSize', 12);
ylabel('Developed Torque (N\\cdot m)', 'FontSize', 12);
title('Three-Phase Induction Motor Torque-Speed Characteristic', 'FontSize', 14);

[T_max, idx_max] = max(T);
fprintf('--- Induction Motor Characteristics ---\\n');
fprintf('Synchronous Speed: %.1f RPM\\n', Ns);
fprintf('Breakdown Torque (T_max): %.2f N*m at %.1f RPM (Slip = %.3f)\\n', T_max, Nr(idx_max), s(idx_max));
fprintf('Starting Torque (s=1): %.2f N*m\\n', T(1));
"""
        return script.strip()

    @staticmethod
    def generate_simulink_model_blueprint(topic: str) -> Dict[str, Any]:
        """Returns structured Simulink architecture blueprint."""
        return {
            "model_topic": topic,
            "solver_configuration": {
                "solver_type": "Variable-step (ode23tb or ode15s for power electronics / ode45 for control systems)",
                "max_step_size": "1e-5",
                "simulation_time": "0.1 to 1.0 seconds"
            },
            "core_library_blocks": [
                "Simscape / Electrical / Specialized Power Systems / Power Electronics (MOSFET, IGBT, Diode)",
                "Simscape / Electrical / Specialized Power Systems / Fundamental Blocks (powergui set to Continuous/Discrete)",
                "Simscape / Electrical / Passives (Series RLC Branch, Parallel RLC Load)",
                "Simulink / Sources (Pulse Generator, Repeating Sequence, DC Voltage Source)",
                "Simulink / Sinks (Scope, Display, To Workspace)",
                "Simulink / Math Operations (Gain, Sum, Product, Integrator)"
            ],
            "step_by_step_assembly": [
                "1. Open MATLAB and type `simulink` to create a blank model.",
                "2. Drag the essential `powergui` block into the canvas and configure simulation type.",
                "3. Place power sources and switches according to circuit topology.",
                "4. Wire gate trigger signals using the Pulse Generator block (set frequency fs and duty cycle D).",
                "5. Connect voltage and current measurement sensors across components of interest.",
                "6. Feed measured signals into Scope blocks for transient and steady-state inspection."
            ]
        }
