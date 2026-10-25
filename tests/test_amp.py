import subprocess

from ai_furby import audio


def test_amp_switch_does_nothing_without_a_pin(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: calls.append(a))
    audio.configure_amp(None)
    audio.set_amp(True)
    assert calls == []


def test_amp_switch_drives_the_pin_high_and_low(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda args, **k: calls.append(args))
    audio.configure_amp(4)
    audio.set_amp(True)
    audio.set_amp(False)
    audio.configure_amp(None)
    assert calls == [["pinctrl", "4", "op", "dh"], ["pinctrl", "4", "op", "dl"]]
