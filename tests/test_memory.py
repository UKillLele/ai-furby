from types import SimpleNamespace

from ai_furby.assistant import PERSONA, Assistant, split_fact, split_mood
from ai_furby.config import DEFAULT_CLAUDE_MODEL, Settings
from ai_furby.memory import FactStore
from ai_furby.voice_loop import memory_command, remembered_fact

SETTINGS = Settings(furby_address=None, furby_connect_timeout=10.0, claude_model=DEFAULT_CLAUDE_MODEL)


class RecordingClient:
    def __init__(self, text: str = "Okay.") -> None:
        self.systems: list[str] = []
        self.text = text
        self.messages = self

    def create(self, **kwargs):
        self.systems.append(kwargs["system"])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


def test_facts_persist_across_stores(tmp_path):
    path = tmp_path / "memory.db"
    FactStore(path).save("dog_name", "my dog is Max")
    assert FactStore(path).all() == ["my dog is Max"]


def test_same_key_replaces_the_old_fact(tmp_path):
    facts = FactStore(tmp_path / "memory.db")
    facts.save("dog_name", "my dog is Max")
    facts.save("dog_name", "my dog is Bo")
    assert facts.all() == ["my dog is Bo"]


def test_facts_are_added_to_the_system_prompt(tmp_path):
    facts = FactStore(tmp_path / "memory.db")
    facts.save("dog_name", "my dog is Max")
    client = RecordingClient()
    Assistant(SETTINGS, client=client, facts=facts).reply("hi")

    assert "my dog is Max" in client.systems[0]
    assert client.systems[0].startswith(PERSONA)


def test_fact_line_is_split_off_and_mood_still_works():
    text, fact = split_fact("Nice to meet Max!\nFACT: dog_name | Max\nMOOD: happy")
    assert fact == ("dog_name", "Max")
    assert split_mood(text) == ("Nice to meet Max!", "happy")
    assert "FACT" not in text


def test_reply_without_fact_line_has_no_fact():
    client = RecordingClient("Hi there!\nMOOD: happy")
    assistant = Assistant(SETTINGS, client=client)
    assistant.reply("hi")
    assert assistant.last_fact is None
    assert assistant.last_mood == "happy"


def test_assistant_reply_saves_fact_through_last_fact(tmp_path):
    client = RecordingClient("Max is a great name!\nFACT: dog_name | Max\nMOOD: happy")
    assistant = Assistant(SETTINGS, client=client, facts=FactStore(tmp_path / "memory.db"))
    assert assistant.reply("my dog is Max") == "Max is a great name!"
    assert assistant.last_fact == ("dog_name", "Max")


def test_forget_last_and_forget_matching(tmp_path):
    facts = FactStore(tmp_path / "memory.db")
    facts.save("dog_name", "my dog is Max")
    facts.save("favorite_food", "I like blueberry pie")
    assert facts.forget_last() == "I like blueberry pie"
    assert facts.forget_matching("dog") == 1
    assert facts.all() == []


def test_remember_that_extracts_the_fact():
    assert remembered_fact("Remember that my dog is Max.") == "my dog is Max"
    assert remembered_fact("please remember that I like tea") == "I like tea"
    assert remembered_fact("What do you remember?") is None


def test_memory_commands_are_recognized():
    assert memory_command("What do you remember?") == ("list", "")
    assert memory_command("Forget that.") == ("forget_last", "")
    assert memory_command("Forget about my dog.") == ("forget", "my dog")
    assert memory_command("Hello Furby") is None
