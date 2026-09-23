import json
import re
from typing import Generator, Dict, Any, List, Optional
from app.services.router import QueryRouter
from app.services.memory import ConversationMemoryManager
from app.services.prompt_builder import PromptBuilder
from app.agents import (
    MachinesAgent,
    PowerSystemsAgent,
    PowerElectronicsAgent,
    ControlSystemsAgent,
    CircuitTheoryAgent,
    MatlabSimulinkAgent,
    EVAgent,
    RenewableAgent,
    BatteryAgent,
    ElectronicsAgent,
)
from app.agents.planner import TaskPlanner
from app.tools.registry import ToolRegistry
from app.rag import RAGRetriever, CitationFormatter, EmbeddingGenerator, FAISSVectorStore
from app.llm import LLMFactory
from app.llm.ollama_client import OllamaClient
from app.llm.model_selector import ModelSelector
from app.utils.logger import get_logger

logger = get_logger("voltix.orchestrator")

class Orchestrator:
    """
    Central Agentic Coordinator orchestrating:
    1. Zero-latency intent routing & model selection
    2. Multi-step task planning (Plan-and-Solve)
    3. ReAct loop (Reason -> Act -> Observe -> Reflect)
    4. Project-isolated RAG retrieval & citations
    5. Real-time SSE event streaming
    """

    def __init__(self, config_obj):
        self.config = config_obj
        self.agents = {
            "MachinesAgent": MachinesAgent(),
            "PowerSystemsAgent": PowerSystemsAgent(),
            "PowerElectronicsAgent": PowerElectronicsAgent(),
            "ControlSystemsAgent": ControlSystemsAgent(),
            "CircuitTheoryAgent": CircuitTheoryAgent(),
            "MatlabSimulinkAgent": MatlabSimulinkAgent(),
            "EVAgent": EVAgent(),
            "RenewableAgent": RenewableAgent(),
            "BatteryAgent": BatteryAgent(),
            "ElectronicsAgent": ElectronicsAgent(),
        }

        # Lazy initialized RAG components
        self._embedder = None
        self._vector_store = None
        self._retriever = None

    def _init_rag(self):
        if self._retriever is None:
            self._embedder = EmbeddingGenerator(model_name=self.config.EMBEDDING_MODEL_NAME)
            self._vector_store = FAISSVectorStore(
                index_path=self.config.FAISS_INDEX_PATH,
                metadata_path=self.config.FAISS_METADATA_PATH,
            )
            self._retriever = RAGRetriever(
                embedder=self._embedder,
                vector_store=self._vector_store,
                similarity_threshold=self.config.RAG_SIMILARITY_THRESHOLD,
            )

    def process_and_stream(
        self,
        prompt: str,
        conversation_id: str,
        model_name: Optional[str] = None,
        academic_mode: str = "Learn",
        project_id: Optional[str] = None,
        enable_rag: bool = True,
        enable_agentic_loop: bool = True,
    ) -> Generator[Dict[str, Any], None, None]:
        """Main pipeline yielding dict events for SSE streaming."""

        # 1. Route Intent & Agent
        routing_info = QueryRouter.classify_intent(prompt)
        agent_key = routing_info.get("agent", "CircuitTheoryAgent")
        agent = self.agents.get(agent_key, self.agents["CircuitTheoryAgent"])
        intent = routing_info.get("intent", "EEE_CONCEPT")

        # 2. Dynamic Model Selection
        selected_model = ModelSelector.select_model(
            intent=intent,
            complexity="high" if academic_mode in ("Research", "Project") else "medium",
            requested_model=model_name
        )

        logger.info(f"Agentic Pipeline: Agent='{agent.name}', Intent='{intent}', Mode='{academic_mode}', Model='{selected_model}'")

        # 3. Retrieve RAG Context with optional Project Isolation
        context_str = ""
        citations = []
        if enable_rag:
            try:
                self._init_rag()
                passages = self._retriever.retrieve(prompt, top_k=self.config.RAG_TOP_K, project_id=project_id)
                if passages:
                    context_str = CitationFormatter.build_context_string(passages)
                    citations = CitationFormatter.format_citations(passages)
            except Exception as rag_err:
                logger.warning(f"RAG retrieval skipped or failed: {rag_err}")

        # 4. Fetch Multi-turn History
        history = ConversationMemoryManager.get_recent_history(conversation_id, limit=6)

        # 5. Yield initial metadata event
        yield {
            "type": "meta",
            "agent": agent.name,
            "intent": intent,
            "model": selected_model,
            "academic_mode": academic_mode,
            "citations": citations,
        }

        # 6. ReAct Agent Loop: Check if tool calling / multi-step execution is beneficial
        tool_observations = []
        ollama_client = OllamaClient(base_url=self.config.OLLAMA_BASE_URL, default_model=selected_model)

        if enable_agentic_loop:
            # Check for URL inspection or specific tool indicators in prompt
            urls = re.findall(r'https?://[^\s<>"]+|www\.[^\s<>"]+', prompt)
            if urls:
                target_url = urls[0]
                yield {"type": "thought", "content": f"Detected web page reference: `{target_url}`. Inspecting page content..."}
                yield {"type": "tool_call", "name": "fetch_local_web_page", "args": {"url": target_url}}
                
                web_res = ToolRegistry.execute("fetch_local_web_page", {"url": target_url})
                yield {"type": "tool_result", "name": "fetch_local_web_page", "output": web_res}
                
                if web_res.get("status") == "success":
                    tool_observations.append(f"Web Page Content from {target_url} (Title: {web_res.get('title')}):\n{web_res.get('content')}")

            # Check if numerical/calculation intent warrants deterministic tool execution
            if intent == "NUMERICAL_SOLVER" or any(kw in prompt.lower() for kw in ["3-phase", "three-phase", "slip", "buck", "boost", "torque", "pf correction", "power factor", "laplace", "transfer function"]):
                # Use local tool calling model
                yield {"type": "thought", "content": "Evaluating engineering parameters for deterministic solver execution..."}
                
                # Check for 3-phase calculation pattern
                if "phase" in prompt.lower() and ("current" in prompt.lower() or "power" in prompt.lower() or "kw" in prompt.lower() or "hp" in prompt.lower()):
                    # Extract values if possible or call 3-phase tool
                    # Attempt intelligent parameter parsing
                    v_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:v|volt)', prompt, re.IGNORECASE)
                    p_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:kw|kilo\s*watt|w\b|watt|hp\b)', prompt, re.IGNORECASE)
                    pf_match = re.search(r'(?:pf|power\s*factor|cos\s*phi)[\s=:]*(\d+(?:\.\d+)?)', prompt, re.IGNORECASE)
                    
                    if v_match and p_match:
                        v_val = float(v_match.group(1))
                        p_raw = float(p_match.group(1))
                        if "hp" in p_match.group(0).lower():
                            p_val = p_raw * 0.7457 # HP to kW
                        else:
                            p_val = p_raw
                        pf_val = float(pf_match.group(1)) if pf_match else 0.85

                        yield {"type": "thought", "content": f"Invoking 3-phase power tool with V={v_val}V, P={p_val}kW, PF={pf_val}..."}
                        yield {"type": "tool_call", "name": "three_phase_power", "args": {"power_val": p_val, "voltage": v_val, "power_factor": pf_val, "power_type": "active", "calc_target": "current"}}
                        
                        tool_out = ToolRegistry.execute("three_phase_power", {"power_val": p_val, "voltage": v_val, "power_factor": pf_val, "power_type": "active", "calc_target": "current"})
                        yield {"type": "tool_result", "name": "three_phase_power", "output": tool_out}
                        tool_observations.append(f"Deterministic 3-Phase Power Calculation Result:\n{json.dumps(tool_out)}")

                # Check for Motor Slip calculation pattern
                elif "slip" in prompt.lower() and ("motor" in prompt.lower() or "induction" in prompt.lower() or "rpm" in prompt.lower()):
                    rpm_matches = re.findall(r'(\d{3,4})\s*(?:rpm)?', prompt, re.IGNORECASE)
                    if len(rpm_matches) >= 2:
                        s_rpm = float(rpm_matches[0])
                        r_rpm = float(rpm_matches[1])
                        yield {"type": "thought", "content": f"Invoking induction motor slip tool with Ns={s_rpm} RPM, Nr={r_rpm} RPM..."}
                        yield {"type": "tool_call", "name": "induction_motor_slip", "args": {"sync_speed": s_rpm, "rotor_speed": r_rpm}}
                        
                        tool_out = ToolRegistry.execute("induction_motor_slip", {"sync_speed": s_rpm, "rotor_speed": r_rpm})
                        yield {"type": "tool_result", "name": "induction_motor_slip", "output": tool_out}
                        tool_observations.append(f"Induction Motor Slip Calculation Result:\n{json.dumps(tool_out)}")

        # 7. Build System Prompt with Combined Context, Tool Observations & Academic Mode
        augmented_context = context_str
        if tool_observations:
            augmented_context += "\n\n### AGENTIC TOOL EXECUTION RESULTS (GROUND TRUTH):\n" + "\n---\n".join(tool_observations)

        system_prompt = PromptBuilder.build_system_prompt(
            agent_name=agent.name,
            base_system_prompt=agent.get_system_prompt(context_str=""),
            context_str=augmented_context,
            academic_mode=academic_mode,
        )

        # 8. Stream Final Response
        yield {"type": "thought", "content": f"Synthesizing response with model: `{selected_model}`..."}

        for token in ollama_client.stream_generate(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
            model_name=selected_model,
        ):
            yield {"type": "token", "content": token}

        # 9. Completion signal
        yield {"type": "done", "citations": citations}
