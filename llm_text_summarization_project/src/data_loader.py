# -*- coding: utf-8 -*-
"""
data_loader.py
--------------
Handles data acquisition, loading, synthetic corpus generation, split partitioning,
and JSON/CSV export for Experiment 7: LLM Text Summarization.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional

_DEFAULT_CORPUS = [
    {
        "id": "DOC_001",
        "category": "Artificial Intelligence & Civil Engineering",
        "title": "UAV Computer Vision for Bridge Structural Health Monitoring",
        "document_text": (
            "Artificial Intelligence and Computer Vision algorithms have revolutionized structural health monitoring "
            "in civil infrastructure. Traditional bridge inspection workflows rely on manual visual checks and scaffolding, "
            "which are labor-intensive, hazardous, and prone to subjective error. In recent deployments, autonomous unmanned aerial "
            "vehicles (UAVs) equipped with high-resolution 4K optical cameras and LiDAR sensors traverse civil structures along "
            "pre-programmed GPS coordinates. Convolutional neural networks (CNNs), specifically YOLOv8 and Mask R-CNN architectures, "
            "process aerial imagery in real time to identify concrete micro-cracks, surface spalling, rebar corrosion, and structural "
            "deflections. The automated system segments defect boundaries down to 0.1-millimeter crack widths with 94.8% classification "
            "precision. Edge-computing microprocessors mounted directly on drone airframes enable onboard inference, transmitting "
            "flagged anomaly coordinates instantly to engineering teams. This automated aerial pipeline cuts inspection durations by 75% "
            "and substantially lowers maintenance overhead while drastically improving worker safety on high-span bridges."
        ),
        "reference_summary": (
            "Autonomous UAVs equipped with optical cameras and CNN computer vision architectures automate structural bridge inspections. "
            "The system detects concrete cracks and corrosion down to 0.1 mm with 94.8% precision, cutting inspection times by 75% and "
            "improving safety."
        ),
        "source": "Journal of Civil Structural Health & AI Monitoring"
    },
    {
        "id": "DOC_002",
        "category": "Energy & Power Systems",
        "title": "Recurrent Deep Learning for Smart Grid Load Forecasting",
        "document_text": (
            "Modern electrical power grids face severe grid stability challenges due to the intermittent nature of renewable energy "
            "generation from wind farms and distributed solar photovoltaic installations. To mitigate sudden supply-demand mismatches, "
            "utility operators are deploying deep sequence learning models to forecast hourly energy consumption and generation profiles. "
            "Recurrent Neural Networks (RNNs), Long Short-Term Memory (LSTM) units, and Gated Recurrent Units (GRUs) model multi-variate "
            "time-series containing historical power demand, ambient temperature, solar irradiance, wind speed, and consumer holiday schedules. "
            "By capturing non-linear diurnal cycles and weather correlations, LSTM models achieve a Mean Absolute Percentage Error (MAPE) "
            "of 2.84%, outperforming traditional ARIMA statistical baselines by over 38%. The forecasted demand trajectories feed into "
            "automated automated dynamic battery energy storage systems (BESS) and peaker power plants, facilitating smooth grid frequency "
            "regulation and minimizing fossil-fuel spinning reserve costs."
        ),
        "reference_summary": (
            "Deep recurrent models such as LSTMs accurately forecast hourly smart grid power demand with a 2.84% MAPE by modeling weather "
            "and consumption patterns, enabling efficient renewable energy integration and battery storage optimization."
        ),
        "source": "IEEE Transactions on Power Systems & Smart Grids"
    },
    {
        "id": "DOC_003",
        "category": "Automotive & Mechanical Systems",
        "title": "Deep Reinforcement Learning in EV Battery Thermal Management",
        "document_text": (
            "Electric vehicle (EV) battery packs require stringent temperature regulation to ensure safety, prevent thermal runaway, and "
            "maximize lithium-ion cell cycle longevity. Operating temperatures must remain strictly between 20°C and 35°C during both rapid "
            "DC fast charging and aggressive highway discharge cycles. Conventional rule-based thermostatic controllers often over-cool or "
            "lag behind thermal spikes, wasting auxiliary power and accelerating battery degradation. Automotive engineers have developed "
            "Deep Deterministic Policy Gradient (DDPG) and Proximal Policy Optimization (PPO) reinforcement learning agents to control "
            "variable-speed coolant pumps, electronic expansion valves, and thermoelectric cooling loops. Trained on high-fidelity electro-thermal "
            "digital twin models, the RL controller anticipates temperature surges based on driver throttle inputs and terrain elevation. "
            "Experimental tests show an 18.5% reduction in cooling energy consumption and a 22% reduction in spatial cell temperature variance."
        ),
        "reference_summary": (
            "Deep reinforcement learning agents optimize electric vehicle battery thermal cooling systems, maintaining optimal cell temperatures "
            "while reducing auxiliary cooling energy consumption by 18.5% and mitigating thermal runaway risks."
        ),
        "source": "SAE International Journal of Electrified Vehicles"
    },
    {
        "id": "DOC_004",
        "category": "Biomedical Engineering",
        "title": "Transformer Attention Networks for Arrhythmia Detection from ECG",
        "document_text": (
            "Cardiovascular diseases represent a leading cause of global mortality, with cardiac arrhythmias requiring rapid, accurate "
            "clinical diagnosis. Traditional 12-lead electrocardiogram (ECG) interpretation by cardiologists is constrained by fatigue and "
            "inter-observer variability. Deep learning architectures have advanced automated ECG analysis, but standard convolutional models "
            "often struggle to capture long-range rhythmic dependencies spanning several cardiac cycles. Researchers developed a multi-head "
            "Self-Attention Transformer network capable of processing raw 12-lead voltage streams at 500 Hz sampling rates. The model "
            "tokenizes continuous waveform segments using temporal convolutional embeddings before passing them through six Transformer "
            "encoder layers. Evaluating on the MIT-BIH Arrhythmia benchmark, the model achieved a 99.1% sensitivity for ventricular premature "
            "contractions and atrial fibrillation. The self-attention heatmaps highlight clinically salient P-wave and QRS complex morphology, "
            "providing interpretable visual explanations for clinical decision support."
        ),
        "reference_summary": (
            "A multi-head self-attention Transformer network analyzes 12-lead ECG signals to detect cardiac arrhythmias with 99.1% sensitivity, "
            "providing interpretable attention heatmaps over P-wave and QRS complexes for clinical support."
        ),
        "source": "Nature Digital Medicine & Healthcare Engineering"
    },
    {
        "id": "DOC_005",
        "category": "Cybersecurity & Cloud Systems",
        "title": "Graph Neural Networks for Zero-Day Cloud Intrusion Detection",
        "document_text": (
            "Cloud computing environments host distributed microservices that generate massive volumes of heterogeneous network flow telemetry "
            "and system call logs. Traditional signature-based Intrusion Detection Systems (IDS) fail against zero-day exploits and multi-stage "
            "Advanced Persistent Threats (APTs) that camouflage malicious lateral movement across virtual private clouds. Security engineers "
            "constructed a spatial-temporal Graph Neural Network (GNN) framework that models cloud infrastructure as a dynamic graph. In this graph, "
            "compute nodes, containers, and databases serve as vertices, while TCP/IP connections, API calls, and IAM permissions constitute edges. "
            "Message passing layers compute topological embeddings that identify anomalous subgraph patterns indicative of privilege escalation "
            "and data exfiltration. The framework achieved an Area Under the ROC Curve (AUC) of 0.982 on enterprise benchmark datasets with an "
            "inference latency under 15 milliseconds, enabling proactive automated container isolation."
        ),
        "reference_summary": (
            "A dynamic Graph Neural Network framework detects zero-day cloud intrusions by analyzing network telemetry and IAM permissions as "
            "graph embeddings, achieving a 0.982 AUC and enabling sub-15ms automated threat response."
        ),
        "source": "ACM Transactions on Privacy and Cloud Security"
    },
    {
        "id": "DOC_006",
        "category": "Robotics & Manufacturing",
        "title": "Vision-Language-Action Models for Industrial Robot Assembly",
        "document_text": (
            "Industrial manufacturing requires flexible robotic manipulation capable of adapting to diverse assembly tasks without weeks of "
            "manual reprogramming. Vision-Language-Action (VLA) models combine multimodal transformer foundation models with continuous robotic "
            "motor trajectory generation. The system ingests natural language operator instructions such as 'fasten the bracket to the chassis "
            "using M4 bolts' along with stereoscopic RGB-D camera feeds of the assembly workstation. The transformer aligns visual affordances "
            "with language semantics to output 7-degree-of-freedom end-effector waypoints and gripper torque profiles. During extensive factory "
            "trials, the VLA model achieved a 96.4% success rate across 45 novel mechanical parts, reducing reconfiguration downtime from 48 hours "
            "to under 10 minutes. Closed-loop visual feedback enables real-time compensation for part misalignments and worker hand interference."
        ),
        "reference_summary": (
            "Vision-Language-Action models translate natural language assembly commands and 3D visual feeds into robotic motor trajectories, "
            "achieving a 96.4% task success rate and cutting assembly line reconfiguration time from 48 hours to 10 minutes."
        ),
        "source": "IEEE Robotics and Automation Letters"
    },
    {
        "id": "DOC_007",
        "category": "Environmental Science & Remote Sensing",
        "title": "Satellite Multispectral AI for Deforestation and Carbon Sink Mapping",
        "document_text": (
            "Monitoring global deforestation and carbon sequestration dynamics requires continuous, wide-area remote sensing analytics. "
            "European Space Agency Sentinel-2 and NASA Landsat satellites capture terabytes of optical and near-infrared multispectral imagery daily. "
            "Analyzing this deluge of satellite data requires robust deep learning segmentation models that remain resilient against heavy cloud cover "
            "and seasonal phenological shifts. A hierarchical U-Net architecture integrated with temporal attention was deployed to segment rainforest "
            "canopy density, illegal logging roads, and wildfire burns across 5 million square kilometers of the Amazon basin. The model cross-references "
            "synthetic aperture radar (SAR) data from Sentinel-1 to penetrate cloud cover during the monsoon season. Validated against airborne LiDAR "
            "field surveys, the model accurately quantified above-ground biomass with an R-squared of 0.91, allowing international carbon credit "
            "registries to verify forestry conservation covenants in near real time."
        ),
        "reference_summary": (
            "Hierarchical temporal U-Nets analyze multispectral satellite imagery and SAR data to monitor deforestation and quantify above-ground "
            "biomass with R2=0.91, providing near-real-time verification for carbon credit registries."
        ),
        "source": "Remote Sensing of Environment Journal"
    },
    {
        "id": "DOC_008",
        "category": "Quantum Computing",
        "title": "Machine Learning for Quantum Circuit Optimization and Error Mitigation",
        "document_text": (
            "Noisy Intermediate-Scale Quantum (NISQ) processors are severely limited by decoherence, gate infidelities, and cross-talk noise. "
            "Executing complex quantum algorithms like the Variational Quantum Eigensolver (VQE) requires transpiling abstract quantum circuits "
            "into minimal depth hardware-native pulse sequences. Researchers formulated quantum circuit compilation as a reinforcement learning "
            "Markov Decision Process. The RL agent discovers optimal gate cancellation rules, qubit routing topologies, and dynamical decoupling "
            "sequences tailored to instantaneous hardware calibration matrices. Across 54-qubit superconducting quantum processors, the ML "
            "compilation framework reduced two-qubit CNOT gate counts by 42% and enhanced algorithmic fidelity by a factor of 2.3. This breakthrough "
            "enables quantum simulations of molecular electronic structures that were previously intractable due to noise accumulation."
        ),
        "reference_summary": (
            "Reinforcement learning optimizes quantum circuit compilation on NISQ superconducting processors, reducing CNOT gate counts by 42% "
            "and doubling algorithmic fidelity for molecular quantum simulations."
        ),
        "source": "Physical Review Applied & Quantum Information"
    },
    {
        "id": "DOC_009",
        "category": "Aerospace & Autonomous Navigation",
        "title": "Neuromorphic Event Cameras and Spiking Neural Networks for High-Speed Drone Flight",
        "document_text": (
            "Autonomous quadcopters navigating dense forest canopies and indoor environments at speeds exceeding 15 meters per second require "
            "ultra-low-latency perception systems. Standard CMOS frame-based cameras suffer from motion blur and frame rate bottlenecks (30-60 FPS), "
            "limiting reaction times during dynamic obstacle avoidance. Engineers integrated bio-inspired neuromorphic event cameras that asynchronously "
            "record microsecond-level brightness changes per pixel with a dynamic range of 120 dB. The asynchronous event stream is ingested by a "
            "Spiking Neural Network (SNN) executed on an ultra-low-power neuromorphic processor consuming under 1.5 Watts. The SNN estimates optical flow "
            "and obstacle proximity with 2-millisecond latency. In flight trials through obstacle courses, the neuromorphic quadcopter avoided 98.7% "
            "of incoming projectiles and thin wires, proving the viability of neuromorphic AI for agile autonomous robotics."
        ),
        "reference_summary": (
            "Neuromorphic event cameras coupled with Spiking Neural Networks provide 2 ms obstacle perception at 1.5W power, enabling autonomous "
            "drones to avoid high-speed obstacles and navigate challenging environments."
        ),
        "source": "Science Robotics & Aerospace Navigation"
    },
    {
        "id": "DOC_010",
        "category": "Materials Science & Nanotechnology",
        "title": "Generative Diffusion Models for Novel Solid-State Electrolyte Discovery",
        "document_text": (
            "The development of next-generation all-solid-state lithium batteries requires solid electrolytes exhibiting high ionic conductivity, "
            "broad electrochemical stability windows, and mechanical ductility. Traditional trial-and-error laboratory synthesis is costly and slow, "
            "exploring only a tiny fraction of the infinite chemical space. Materials scientists introduced 3D Crystal Generative Diffusion Models "
            "(DiffCSP) trained on 150,000 DFT-calculated inorganic crystal structures. The generative model denoises atom positions, space group "
            "symmetries, and unit cell vectors simultaneously. The model proposed 28 previously uncatalogued lithium thiophosphate crystal "
            "structures, which were then evaluated using automated ab-initio molecular dynamics simulations. Four synthesized candidates demonstrated "
            "room-temperature lithium-ion conductivities exceeding 12 mS/cm, matching liquid electrolytes while eliminating flammability hazards."
        ),
        "reference_summary": (
            "3D Crystal Generative Diffusion Models accelerated solid-state electrolyte discovery, generating novel lithium thiophosphate crystals "
            "with ionic conductivity over 12 mS/cm to enable safe, high-energy solid-state batteries."
        ),
        "source": "Advanced Energy Materials & AI Synthesis"
    }
]


def load_config(config_path: str) -> Dict[str, Any]:
    """Load JSON configuration from path."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_or_create_raw_articles(raw_csv_path: str) -> pd.DataFrame:
    """
    Load raw articles CSV if present; otherwise generate the curated
    multi-domain scientific and engineering corpus.
    """
    os.makedirs(os.path.dirname(os.path.abspath(raw_csv_path)), exist_ok=True)
    if os.path.exists(raw_csv_path):
        df = pd.read_csv(raw_csv_path)
        if not df.empty:
            return df
    
    df = pd.DataFrame(_DEFAULT_CORPUS)
    df.to_csv(raw_csv_path, index=False, encoding="utf-8")
    return df


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataset deterministically into train, validation, and test subsets."""
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n = len(df_shuffled)
    
    if n <= 10:
        # For a 10-document corpus, partition into 6 train, 1 validation, 3 test
        train_df = df_shuffled.iloc[:6].reset_index(drop=True)
        val_df = df_shuffled.iloc[6:7].reset_index(drop=True)
        test_df = df_shuffled.iloc[7:].reset_index(drop=True)
        return train_df, val_df, test_df

    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    train_df = df_shuffled.iloc[:n_train].reset_index(drop=True)
    val_df = df_shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df_shuffled.iloc[n_train + n_val:].reset_index(drop=True)
    
    if test_df.empty:
        test_df = df_shuffled.iloc[-2:].reset_index(drop=True)
    
    return train_df, val_df, test_df


def save_processed_data(
    cleaned_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cleaned_csv_path: str,
    test_docs_json_path: str,
    reference_summaries_json_path: str
) -> None:
    """Save processed artifacts, test documents JSON, and reference summaries JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(cleaned_csv_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(test_docs_json_path)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(reference_summaries_json_path)), exist_ok=True)

    cleaned_df.to_csv(cleaned_csv_path, index=False, encoding="utf-8")

    test_docs = test_df[["id", "category", "title", "document_text"]].to_dict(orient="records")
    with open(test_docs_json_path, "w", encoding="utf-8") as f:
        json.dump(test_docs, f, indent=2, ensure_ascii=False)

    ref_summaries = {row["id"]: row["reference_summary"] for _, row in test_df.iterrows()}
    with open(reference_summaries_json_path, "w", encoding="utf-8") as f:
        json.dump(ref_summaries, f, indent=2, ensure_ascii=False)


def load_test_documents(test_docs_json_path: str) -> List[Dict[str, Any]]:
    """Load test documents from JSON."""
    if not os.path.exists(test_docs_json_path):
        raise FileNotFoundError(f"Test documents not found: {test_docs_json_path}")
    with open(test_docs_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_reference_summaries(reference_summaries_json_path: str) -> Dict[str, str]:
    """Load ground-truth reference summaries mapping doc_id -> reference_summary."""
    if not os.path.exists(reference_summaries_json_path):
        raise FileNotFoundError(f"Reference summaries not found: {reference_summaries_json_path}")
    with open(reference_summaries_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_generated_summaries(
    generated_summaries: List[Dict[str, Any]],
    output_path: str
) -> None:
    """Save model-generated summaries with metadata to JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(generated_summaries, f, indent=2, ensure_ascii=False)
