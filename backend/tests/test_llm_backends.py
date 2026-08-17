"""What the Ollama backend actually puts on the wire.

`helper` and `reason` are separate flags because the internal calls in this project
want different combinations, and getting the combination wrong is invisible: the
call still succeeds, it just answers badly (a critique that cannot judge) or slowly
(a router that thinks). Neither shows up as an error, so it is asserted here.

No network — the request is intercepted and inspected.
"""

import json
import urllib.request

import pytest

from core import llm
from core.config import settings


@pytest.fixture
def sent_payload(monkeypatch: pytest.MonkeyPatch):
    """Capture the JSON body of the next Ollama request instead of sending it."""
    captured: dict = {}

    class FakeResponse:
        def read(self) -> bytes:
            return json.dumps(
                {"message": {"content": "ok"}, "prompt_eval_count": 7, "eval_count": 3}
            ).encode()

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

    def fake_urlopen(request, timeout=None):
        captured.update(json.loads(request.data))
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return captured


def test_answers_do_not_think(sent_payload: dict) -> None:
    """Answering from supplied passages is extraction. Thinking only costs seconds
    and leaks a stray control token into the answer."""
    llm.generate("sys", "user", settings.ollama_model)

    assert sent_payload["think"] is False
    assert sent_payload["options"]["num_predict"] == settings.anthropic_max_tokens_answer


def test_a_judgement_call_thinks_and_takes_the_local_helper_budget(sent_payload: dict) -> None:
    """Multi-Pass's critique. Without thinking it answers COMPLETE to everything."""
    llm.generate("sys", "user", settings.ollama_model, helper=True, reason=True)

    assert sent_payload["think"] is True
    # Not the Anthropic helper cap: thinking tokens come out of this same budget,
    # and at 256 the reasoning ate the whole thing and returned an empty reply.
    assert sent_payload["options"]["num_predict"] == settings.ollama_helper_num_predict
    assert settings.ollama_helper_num_predict > settings.anthropic_max_tokens_helper


def test_a_router_call_is_cheap_without_thinking(sent_payload: dict) -> None:
    """Phase 7's premise: a cheap classifier in front of expensive workers.

    Thinking measured at 7-16s per call — more than the answer it routes. If this
    ever asserts True, Auto RAG's router costs more than the work it dispatches and
    the technique teaches the opposite of its lesson.
    """
    llm.generate("sys", "user", settings.ollama_model, helper=True, reason=False)

    assert sent_payload["think"] is False
    assert sent_payload["options"]["num_predict"] == settings.ollama_helper_num_predict


def test_context_window_is_set_on_every_call(sent_payload: dict) -> None:
    """Ollama's 2048 default silently truncates RAG prompts — the Phase 1 bug."""
    llm.generate("sys", "user", settings.ollama_model, helper=True, reason=True)

    assert sent_payload["options"]["num_ctx"] == settings.ollama_num_ctx
    assert sent_payload["options"]["num_ctx"] >= 8192
