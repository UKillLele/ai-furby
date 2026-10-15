from types import SimpleNamespace

from ai_furby.assistant import MAX_HISTORY_MESSAGES, PERSONA, Assistant
from ai_furby.config import DEFAULT_CLAUDE_MODEL, Settings

SETTINGS = Settings(furby_address=None, furby_connect_timeout=10.0, claude_model=DEFAULT_CLAUDE_MODEL)


class FakeMessages:
    def __init__(self, reply_text: str = "Hi there!") -> None:
        self.reply_text = reply_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply_text)])


class FakeClient:
    def __init__(self, reply_text: str = "Hi there!") -> None:
        self.messages = FakeMessages(reply_text)


def test_reply_returns_text_and_sends_persona_and_model():
    client = FakeClient("  Hello, friend!  ")
    assistant = Assistant(SETTINGS, client=client)

    assert assistant.reply("hi") == "Hello, friend!"

    call = client.messages.calls[0]
    assert call["model"] == "claude-haiku-5-5"
    assert call["system"] == PERSONA
    assert call["messages"] == [{"role": "user", "content": "hi"}]


def test_history_carries_into_next_request():
    client = FakeClient("ok")
    assistant = Assistant(SETTINGS, client=client)

    assistant.reply("first")
    assistant.reply("second")

    second_call = client.messages.calls[1]["messages"]
    assert [m["content"] for m in second_call] == ["first", "ok", "second"]


def test_history_is_trimmed_to_limit():
    client = FakeClient("ok")
    assistant = Assistant(SETTINGS, client=client)

    for i in range(MAX_HISTORY_MESSAGES):
        assistant.reply(f"msg {i}")

    assert len(client.messages.calls[-1]["messages"]) == MAX_HISTORY_MESSAGES


def test_reset_clears_history():
    client = FakeClient("ok")
    assistant = Assistant(SETTINGS, client=client)
    assistant.reply("first")
    assistant.reset()
    assistant.reply("again")

    assert client.messages.calls[-1]["messages"] == [{"role": "user", "content": "again"}]


def test_missing_api_key_gives_clear_error(monkeypatch):
    import pytest

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        Assistant(SETTINGS)
