from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _install_fake_auxiliary_client(call_llm_impl):
    """Inject a fake agent.auxiliary_client so extraction.py's lazy import
    resolves offline, without a real Hermes checkout or network call."""
    agent_pkg = sys.modules.setdefault("agent", types.ModuleType("agent"))
    aux_mod = types.ModuleType("agent.auxiliary_client")
    aux_mod.call_llm = call_llm_impl
    aux_mod.extract_content_or_reasoning = lambda response: response.choices[0].message.content
    sys.modules["agent.auxiliary_client"] = aux_mod
    agent_pkg.auxiliary_client = aux_mod


def _load_extraction_module():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "lancedb_extraction_under_test", root / "src" / "extraction.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _fake_response(content: str):
    message = types.SimpleNamespace(content=content)
    choice = types.SimpleNamespace(message=message)
    return types.SimpleNamespace(choices=[choice])


def test_extract_passes_response_format_via_extra_body_not_top_level():
    captured = {}

    def fake_call_llm(**kwargs):
        captured.update(kwargs)
        return _fake_response('{"facts": []}')

    _install_fake_auxiliary_client(fake_call_llm)
    extraction = _load_extraction_module()

    extraction.extract([{"role": "user", "content": "hello"}])

    # call_llm() has no top-level response_format kwarg (only extra_body) --
    # passing it directly raises TypeError.
    assert "response_format" not in captured
    assert captured["extra_body"] == {"response_format": {"type": "json_object"}}


def test_extract_parses_choices_message_content():
    def fake_call_llm(**kwargs):
        return _fake_response(
            '{"facts": [{"content": "user likes tea", "category": "preference"}]}'
        )

    _install_fake_auxiliary_client(fake_call_llm)
    extraction = _load_extraction_module()

    facts = extraction.extract([{"role": "user", "content": "I love tea"}])

    assert facts == [
        {
            "content": "user likes tea",
            "abstract": "",
            "category": "preference",
            "tags": [],
            "evidence": [],
        }
    ]


def test_extract_returns_empty_list_when_call_llm_raises():
    def fake_call_llm(**kwargs):
        raise TypeError("unexpected keyword argument 'response_format'")

    _install_fake_auxiliary_client(fake_call_llm)
    extraction = _load_extraction_module()

    assert extraction.extract([{"role": "user", "content": "hello"}]) == []


def test_extract_returns_empty_on_no_messages():
    extraction = _load_extraction_module()
    assert extraction.extract([]) == []
