from app.llm.model_selector import ModelSelector

def test_model_selector_explicit_model():
    selected = ModelSelector.select_model(intent="EEE_CONCEPT", requested_model="gpt-oss:120b-cloud")
    assert selected == "gpt-oss:120b-cloud"

def test_model_selector_proof_intent():
    selected = ModelSelector.select_model(intent="MATHEMATICAL_PROOF", requested_model=None)
    assert selected == "gpt-oss:120b-cloud"

def test_model_selector_code_intent():
    selected = ModelSelector.select_model(intent="MATLAB_SCRIPT", requested_model=None)
    assert selected == "deepseek-v4-flash:cloud"

def test_model_selector_numerical_intent():
    selected = ModelSelector.select_model(intent="NUMERICAL_SOLVER", requested_model=None)
    assert selected == "qwen2.5:7b"

def test_model_selector_research_high_complexity():
    selected = ModelSelector.select_model(intent="RESEARCH_INQUIRY", complexity="high")
    assert selected == "gemma4:31b-cloud"

def test_model_metadata():
    metadata = ModelSelector.get_all_models_metadata()
    assert "qwen2.5:3b" in metadata
    assert "qwen2.5:7b" in metadata
    assert "deepseek-v4-flash:cloud" in metadata
    assert "gemma4:cloud" in metadata
    assert "gemma4:31b-cloud" in metadata
    assert "gpt-oss:120b-cloud" in metadata
