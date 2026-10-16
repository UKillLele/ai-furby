from types import SimpleNamespace

from ai_furby.assistant import PERSONA, Assistant
from ai_furby.config import DEFAULT_CLAUDE_MODEL, Settings
from ai_furby.memory import FactStore
from ai_furby.voice_loop import remembered_fact

SETTINGS = Settings(furby_address=None, furby_connect_timeout=10.0, claude_model=DEFAULT_CLAUDE_MODEL)


class RecordingClient:
    def __init__(self) -> None:
        self.systems: list[str] = []
        self.messages = self

    def create(self, **kwargs):
        self.systems.append(kwargs["system"])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Okay.")])


def test_facts_persist_across_stores(tmp_path):
    path = tmp_path / "memory.db"
    FactStore(path).add("my dog is Max")
    assert FactStore(path).all() == ["my dog is Max"]


def test_facts_are_added_to_the_system_prompt(tmp_path):
    facts = FactStore(tmp_path / "memory.db")
    facts.add("my dog is Max")
    client = RecordingClient()
    Assistant(SETTINGS, client=client, facts=facts).reply("hi")

    assert "my dog is Max" in client.systems[0]
    assert client.systems[0].startswith(PERSONA)


def test_no_facts_leaves_prompt_unchanged(tmp_path):
    client = RecordingClient()
    Assistant(SETTINGS, client=client, facts=FactStore(tmp_path / "memory.db")).reply("hi")

    assert client.systems[0] == PERSONA


def test_remember_that_extracts_the_fact():
    assert remembered_fact("Remember that my dog is Max.") == "my dog is Max"
    assert remembered_fact("please remember that I like tea") == "I like tea"
    assert remembered_fact("What do you remember?") is None
