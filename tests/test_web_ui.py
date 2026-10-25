import asyncio

from ai_furby.web_ui import handle

TOKEN = "secret-token"


class FakeFurby:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def set_backlight(self, enabled):
        self.calls.append(("eyes", enabled))

    async def set_antenna_color(self, red, green, blue):
        self.calls.append(("antenna", red, green, blue))

    async def act_for(self, action, seconds, repeat_seconds=None):
        self.calls.append(("mood", action, seconds))


def run(coro):
    return asyncio.run(coro)


def post(path, furby, token=TOKEN, headers=None):
    all_headers = {"x-token": token} if token else {}
    all_headers.update(headers or {})
    return run(handle("POST", path, all_headers, furby, TOKEN))


def test_page_is_served_without_a_token():
    status, ctype, body = run(handle("GET", "/", {}, FakeFurby(), TOKEN))
    assert status == 200 and "text/html" in ctype and "Furby" in body


def test_changes_need_the_token():
    furby = FakeFurby()
    assert post("/api/eyes?on=1", furby, token="")[0] == 401
    assert post("/api/eyes?on=1", furby, token="wrong")[0] == 401
    assert furby.calls == []


def test_eyes_on_and_off():
    furby = FakeFurby()
    assert post("/api/eyes?on=1", furby)[0] == 200
    assert post("/api/eyes?on=0", furby)[0] == 200
    assert furby.calls == [("eyes", True), ("eyes", False)]


def test_sleep_turns_antenna_and_eyes_off():
    furby = FakeFurby()
    assert post("/api/sleep", furby)[0] == 200
    assert ("antenna", 0, 0, 0) in furby.calls
    assert ("eyes", False) in furby.calls


def test_antenna_color_is_checked_for_range():
    furby = FakeFurby()
    assert post("/api/antenna?r=255&g=160&b=0", furby)[0] == 200
    assert post("/api/antenna?r=300&g=0&b=0", furby)[0] == 400
    assert post("/api/antenna?r=x&g=0&b=0", furby)[0] == 400
    assert furby.calls == [("antenna", 255, 160, 0)]


def test_only_known_moods_run():
    furby = FakeFurby()
    assert post("/api/mood?name=happy", furby)[0] == 200
    assert post("/api/mood?name=../../x", furby)[0] == 400
    assert len(furby.calls) == 1 and furby.calls[0][0] == "mood"


def test_unknown_paths_are_not_found():
    assert post("/api/nothing", FakeFurby())[0] == 404
    assert run(handle("GET", "/api/eyes", {}, FakeFurby(), TOKEN))[0] == 404
