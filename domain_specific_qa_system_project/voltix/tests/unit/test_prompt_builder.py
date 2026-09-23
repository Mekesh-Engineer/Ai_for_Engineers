import pytest
from app.services.prompt_builder import PromptBuilder

def test_prompt_builder_academic_modes():
    p_learn = PromptBuilder.build_system_prompt("MachinesAgent", "", academic_mode="Learn")
    assert "ACADEMIC MODE: LEARN" in p_learn

    p_viva = PromptBuilder.build_system_prompt("MachinesAgent", "", academic_mode="Viva")
    assert "ACADEMIC MODE: VIVA VOCE" in p_viva

    p_lab = PromptBuilder.build_system_prompt("MachinesAgent", "", academic_mode="Laboratory")
    assert "ACADEMIC MODE: LABORATORY" in p_lab
    assert "APPARATUS" in p_lab
