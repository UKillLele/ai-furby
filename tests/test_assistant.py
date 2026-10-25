from types import SimpleNamespace

import pytest

from ai_furby.assistant import MAX_HISTORY_MESSAGES, PERSONA, Assistant, split_mood
from ai_furby.config import DEFAULT_CHAT_MODEL, Settings

SETTINGS = Settings(furby_address=None, furby_connect_timeout=10.0, mistral_api_key="test-key")


class FakeResponse:
    def __init__(self, text: str) -> None:
        self._text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"role": "assistant", "content": self._text}}]}


class FakeHttp:
    def __init__(self, reply_text: str = "Hi there!") -> None:
        self.reply_text = reply_text
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, **_kwargs):
        self.calls.append({"url": url, "headers": headers, **json})
        return FakeResponse(self.reply_text)


def test_reply_returns_text_and_sends_persona_and_model():
    client = FakeHttp("  Hello, friend!  ")
    assistant = Assistant(SETTINGS, client=client)

    assert assistant.reply("hi") == "Hello, friend!"

    call = client.calls[0]
    assert call["model"] == DEFAULT_CHAT_MODEL
    assert call["messages"][0] == {"role": "system", "content": PERSONA}
    assert call["messages"][-1] == {"role": "user", "content": "hi"}
    assert call["headers"] == {"Authorization": "Bearer test-key"}


def test_history_carries_into_next_request():
    client = FakeHttp("ok")
    assistant = Assistant(SETTINGS, client=client)

    assistant.reply("first")
    assistant.reply("second")

    contents = [m["content"] for m in client.calls[1]["messages"][1:]]
    assert contents == ["first", "ok", "second"]


def test_history_is_trimmed_to_limit():
    client = FakeHttp("ok")
    assistant = Assistant(SETTINGS, client=client)

    for i in range(MAX_HISTORY_MESSAGES):
        assistant.reply(f"msg {i}")

    # The system message is added on top of the trimmed history.
    assert len(client.calls[-1]["messages"]) == MAX_HISTORY_MESSAGES + 1


def test_reset_clears_history():
    client = FakeHttp("ok")
    assistant = Assistant(SETTINGS, client=client)
    assistant.reply("first")
    assistant.reset()
    assistant.reply("again")

    assert client.calls[-1]["messages"][1:] == [{"role": "user", "content": "again"}]


def test_missing_api_key_gives_clear_error(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY is not set"):
        Assistant(Settings(furby_address=None, mistral_api_key=""))


def test_split_mood_separates_tag_from_spoken_text():
    assert split_mood("Hello there!\nMOOD: excited") == ("Hello there!", "excited")
    assert split_mood("Hi.\nMOOD: Sleepy  ") == ("Hi.", "sleepy")


def test_split_mood_ignores_unknown_or_missing_mood():
    assert split_mood("Hi there.\nMOOD: grumpy") == ("Hi there.", None)
    assert split_mood("Just talking.") == ("Just talking.", None)


def test_reply_sets_last_mood():
    client = FakeHttp("Yay, hello!\nMOOD: happy")
    assistant = Assistant(SETTINGS, client=client)

    assert assistant.reply("hi") == "Yay, hello!"
    assert assistant.last_mood == "happy"
