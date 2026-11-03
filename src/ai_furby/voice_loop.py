"""Listen, think, speak. The Furby is optional: without FURBY_ADDRESS the assistant runs audio-only."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
import time

import numpy as np

from ai_furby.antenna_touch import is_input
from ai_furby.assistant import Assistant
from ai_furby.config import Settings
from ai_furby.furby_remote import FurbyRemote
from ai_furby.memory import FactStore
from ai_furby.mood_actions import ACTION_SECONDS, MOOD_ACTIONS, pick_greeting, pick_mouth_action
from ai_furby.speech_to_text import SpeechToText
from ai_furby.text_to_speech import TextToSpeech

log = logging.getLogger(__name__)

STOP_PHRASES = ("stop listening",)
STOP_GREETINGS = ("goodbye", "good bye", "bye")
STOP_NAMES = ("furby", "kirby", "fury", "furry", "ferby", "firby", "furbie", "furbee", "turby", "burby")


def memory_command(heard: str) -> tuple[str, str] | None:
    """Recognizes "what do you remember", "forget that", and "forget <words>". Returns (command, words) or None."""
    text = heard.lower().strip(" .!?")
    if re.fullmatch(r"(what do you remember|what do you know about me|what have i told you)", text):
        return ("list", "")
    if text in ("forget that", "forget that last thing", "never mind that"):
        return ("forget_last", "")
    match = re.fullmatch(r"forget (?:about |that )?(.+)", text)
    if match:
        return ("forget", match.group(1))
    return None


def _run_memory_command(command: tuple[str, str], facts: FactStore, heard: str) -> str:
    kind, words = command
    if kind == "list":
        known = facts.all()
        if not known:
            return "I don't remember anything about you yet."
        return "Here's what I remember: " + "; ".join(known) + "."
    if kind == "forget_last":
        removed = facts.forget_last()
        return f"Okay, I forgot: {removed}." if removed else "There's nothing new for me to forget."
    removed_count = facts.forget_matching(words)
    if removed_count:
        return f"Okay, I forgot {removed_count} thing{'s' if removed_count != 1 else ''} about {words}."
    return f"I don't have anything about {words} to forget."


async def _say(text: str, tts, audio, settings: Settings) -> None:
    try:
        speech = await asyncio.to_thread(tts.synthesize, text)
        await asyncio.to_thread(audio.play, speech, tts.sample_rate, settings.speaker_device)
    except Exception:  # a speaker problem should not stop the conversation
        log.warning("Could not play the reply", exc_info=True)
    await asyncio.sleep(ECHO_SETTLE_SECONDS)


def remembered_fact(heard: str) -> str | None:
    """"Remember that my dog is Max" -> "my dog is Max"; anything else returns None."""
    match = re.match(r"\s*(?:please\s+)?remember\s+that\s+(.+)", heard, re.IGNORECASE)
    return match.group(1).strip(" .!?") if match else None


def is_stop_phrase(heard: str) -> bool:
    """Ignores case, punctuation, and extra spaces, so "Goodbye, Furby!" matches.

    The transcriber often hears the name as a similar word ("Kirby"), so those count too.
    A bare "goodbye" never stops the Furby, since it's easy to say by accident.
    """
    normalized = " ".join(re.sub(r"[^a-z ]", " ", heard.lower()).split())
    if normalized in STOP_PHRASES:
        return True
    return any(f"{greeting} {name}" == normalized for greeting in STOP_GREETINGS for name in STOP_NAMES)
LISTENING_COLOR = (0, 255, 0)
# Room echo of Furby's own voice fades after playback ends; wait so the mic does not hear and answer it.
ECHO_SETTLE_SECONDS = 1.0
ANTENNA_WINDOW_SECONDS = 1.5  # antenna readings this close together count as one touch
ANTENNA_TOUCHES_NEEDED = 2  # antenna readings needed within the window; one reading was too easy to trigger by accident
FOLLOW_UP_SECONDS = 20.0
BOOT_ANTENNA_SECONDS = 3.0  # antenna lights this long at startup to show the program is running
LINK_RETRY_SECONDS = 5.0  # while the Furby is out of contact, try to reconnect this often
LINK_RETRY_WINDOW_SECONDS = 30.0  # for this long after the prompt, before going quiet until the link is back
LINK_PROMPT = "Hello, I'm Furby. Please touch my antenna so I can connect."
READY_RATE = 16000  # sample rate of the ready chime
WAKE_FOLLOW_UP_SECONDS = 1.0  # after the wake word, how long to wait for something to say before greeting
# After the mic closes, Windows can still be switching the Bluetooth headset; give it a moment before playing.
MIC_RELEASE_SECONDS = 0.5
MOVEMENT_DELAY_SECONDS = 0.3  # how long audio leads the Furby's movement; tune by listening and watching
SPEAKING_COLOR = (0, 0, 255)
THINKING_COLOR = (255, 160, 0)  # amber while the assistant is working on a reply


async def run(settings: Settings) -> None:
    stt = SpeechToText(settings)
    tts = TextToSpeech(settings)
    facts = FactStore(settings.memory_path)
    assistant = Assistant(settings, facts=facts)

    async with contextlib.AsyncExitStack() as stack:
        furby = None
        if settings.furby_address:
            furby = await stack.enter_async_context(FurbyRemote(settings))

        # Imported only after the Furby is connected: on Windows, loading sounddevice first makes the
        # Bluetooth connection fail with "Thread is configured for Windows GUI".
        from ai_furby import audio

        # One microphone stays open for the whole run, so nothing said after the wake word is lost.
        mic = audio.MicFeed(settings.mic_device)
        mic.start()
        stack.callback(mic.stop)

        if furby is None:
            print("Furby: not connected (FURBY_ADDRESS is not set in this terminal), so it won't move.")
        else:
            print(f"Furby: connected to {settings.furby_address}")
        wake = None
        if settings.wake_word_enabled:
            from ai_furby.wake_word import WakeWordListener, default_model_path

            wake = WakeWordListener(default_model_path(), threshold=settings.wake_threshold)
            print("Say 'Hey Furby' to start talking. Say 'goodbye Furby' to stop.")
        else:
            print("Listening. Say 'goodbye Furby' or press Ctrl+C to stop.")
        conversing = False  # True while a conversation is ongoing and the wake word isn't required
        listen_until = 0.0  # the listening window ends at this time; the antenna is green for exactly that window
        mask_worn = asyncio.Event()  # set while the sleep mask is on; nothing wakes the Furby until it's removed
        if furby is not None:
            mask_task = asyncio.create_task(_watch_sleep_mask(furby, mask_worn))
            stack.callback(mask_task.cancel)
            link_task = asyncio.create_task(_watch_link(furby, tts, audio, settings))
            stack.callback(link_task.cancel)
        if furby is not None and settings.web_port and settings.web_token:
            from ai_furby.web_ui import serve as serve_phone_page

            page_task = asyncio.create_task(serve_phone_page(furby, settings.web_port, settings.web_token))
            stack.callback(page_task.cancel)
            print("Phone page: http://furby.local/ (on the Pi; use the computer's address elsewhere)")
        # The chime is the startup signal: it plays once the program is ready to listen, connected or not.
        print("Furby: ready. Say 'Hey Furby' to start talking.")
        try:
            await asyncio.to_thread(audio.play, _ready_chime(), READY_RATE, settings.speaker_device)
        except Exception:  # a speaker problem should not stop startup
            log.warning("Could not play the ready chime", exc_info=True)
        # Eyes and antenna light for a few seconds to show the program has started, then go dark until the wake word.
        await _wake_up(furby)
        await _set_antenna(furby, LISTENING_COLOR)
        await asyncio.sleep(BOOT_ANTENNA_SECONDS)
        await _set_antenna(furby, (0, 0, 0))
        movement: asyncio.Task | None = None  # the Furby's movement for the last reply, if it is still running
        while True:
            if mask_worn.is_set():
                # Asleep while the mask is on: ignore the wake word, antenna, and speech until it's removed.
                conversing = False
                await _go_to_sleep(furby)
                while mask_worn.is_set():
                    await asyncio.sleep(0.5)
                print("(sleep mask removed; say the wake word to talk)")
                continue
            woke_by = None  # "voice" or "antenna" right after a wake; None otherwise
            if wake is not None and not conversing:
                await _go_to_sleep(furby)
                woke_by = await _wait_for_wake_word_or_antenna(wake, mic, furby)
                print(f"(woken by {woke_by})")
                log.info("Furby is awake (woken by %s); eyes on", woke_by)
                await _wake_up(furby)
                if woke_by == "antenna":
                    await _greet(furby, tts, audio, settings)
                    conversing = True
                    listen_until = time.monotonic() + FOLLOW_UP_SECONDS
                    woke_by = None  # the greeting already answered the wake
            await _set_antenna(furby, LISTENING_COLOR)
            # During a conversation, wait up to FOLLOW_UP_SECONDS for the next thing you say; silence ends it.
            if woke_by == "voice":
                wait = WAKE_FOLLOW_UP_SECONDS
            else:
                wait = max(0.0, listen_until - time.monotonic()) if conversing else 30.0
            if woke_by is None:
                mic.drain()  # drop the Furby's own voice and movement noise; after a wake, keep the audio for the question
            utterance = await asyncio.to_thread(
                audio.record_utterance, mic, threshold=settings.mic_threshold, wait_seconds=wait
            )
            if mask_worn.is_set():
                continue  # the mask went on while we were listening; ignore whatever was heard
            if utterance.size == 0:
                if woke_by == "voice":
                    # The wake word came with nothing after it: greet, then keep the conversation open.
                    await _greet(furby, tts, audio, settings)
                    conversing = True
                    listen_until = time.monotonic() + FOLLOW_UP_SECONDS
                    continue
                if conversing and woke_by is None and time.monotonic() < listen_until:
                    # The listening window is still open, so keep listening.
                    await asyncio.sleep(0.2)
                    continue
                if conversing:
                    print("(quiet for a while; say the wake word to talk again)")
                    log.info("Furby is idle; eyes off until the wake word")
                conversing = False
                continue
            conversing = True
            await _set_antenna(furby, THINKING_COLOR)  # amber while Mistral transcribes and answers

            heard = await asyncio.to_thread(stt.transcribe, utterance)
            if not heard:
                # Background noise, such as the Furby's own movement, can pass the mic threshold without any words.
                # Keep listening until the quiet period since the last real speech runs out, then go back to sleep.
                if conversing and time.monotonic() < listen_until:
                    continue
                conversing = False
                continue
            print(f"You: {heard}")
            command = memory_command(heard)
            if command is not None:
                reply_text = _run_memory_command(command, facts, heard)
                print(f"Furby: {reply_text}")
                await _say(reply_text, tts, audio, settings)
                continue

            fact = remembered_fact(heard)
            if fact:
                facts.save(fact, fact)
                print(f"(remembered: {fact})")

            if is_stop_phrase(heard):
                print("Furby: Bye for now!")
                await _say_goodbye(furby, tts, audio, settings)
                conversing = False
                if settings.wake_word_enabled:
                    continue  # back to waiting for the wake word
                break

            try:
                reply = await asyncio.to_thread(assistant.reply, heard)
                if assistant.last_fact is not None:
                    key, text = assistant.last_fact
                    facts.save(key, text)
                    print(f"(learned: {text})")
            except Exception as error:  # a failed API call should not end the conversation
                log.exception("Assistant request failed")
                print(f"Furby: (could not think of a reply: {type(error).__name__})")
                await _set_antenna(furby, (0, 0, 0))
                continue
            print(f"Furby: {reply}")
            log.info("Mood: %s", assistant.last_mood)

            # Whatever happens to the speech, the listening window opens afterwards, so the mic and green light come back.
            try:
                speech = await asyncio.to_thread(tts.synthesize, reply)  # amber stays on until the speech is ready
                seconds = len(speech) / tts.sample_rate
                await asyncio.sleep(MIC_RELEASE_SECONDS)
                await _set_antenna(furby, SPEAKING_COLOR)  # blue while the response plays
                if movement is not None and not movement.done():
                    movement.cancel()  # a new reply replaces the previous movement
                movement = await _speak_with_movement(
                    furby, assistant.last_mood, audio, speech, tts.sample_rate, seconds, settings.speaker_device
                )
            except Exception as error:  # a speech or speaker problem should not end the conversation
                log.exception("Reply playback failed")
                print(f"(could not play speech: {type(error).__name__}; check the speech service and speakers)")
            listen_until = time.monotonic() + FOLLOW_UP_SECONDS  # the green listening window opens after the reply
            await asyncio.sleep(ECHO_SETTLE_SECONDS)


async def _speak_with_movement(
    furby: FurbyRemote | None,
    mood: str | None,
    audio,
    speech,
    sample_rate: int,
    seconds: float,
    speaker_device: str | None,
) -> asyncio.Task | None:
    """Plays the reply while the Furby moves. Returns when the speech ends; the movement continues in the background
    and is returned so the caller can cancel or await it."""
    speaking = asyncio.create_task(asyncio.to_thread(audio.play, speech, sample_rate, speaker_device))
    if furby is None or mood is None or mood not in MOOD_ACTIONS:
        await speaking
        return None
    # Audio takes a moment to open its stream, and Bluetooth commands take a moment to reach the Furby.
    # Let the audio start first, then start the movement, so the two line up.
    log.info("Movement starting: %s for %.1f s", mood, seconds - MOVEMENT_DELAY_SECONDS)
    await asyncio.sleep(MOVEMENT_DELAY_SECONDS)
    movement = asyncio.create_task(_move_with_beak(furby, MOOD_ACTIONS[mood], seconds - MOVEMENT_DELAY_SECONDS))
    await speaking
    # The mic opens as soon as the speech ends, so the movement finishes while the Furby listens.
    return movement


async def _move_with_beak(furby: FurbyRemote, action: tuple[int, int, int, int], seconds: float) -> None:
    """One full mood movement at the start of the reply, then beak movement for the rest of it."""
    try:
        first = min(ACTION_SECONDS.get(action, 2.5), seconds)
        log.info("Mood action %s for %.1f s", action, first)
        await furby.act_for(action, first)
        if seconds > first:
            await furby.act_for(pick_mouth_action(), seconds - first)
    except Exception:  # an action failure should not stop the reply
        log.warning("Could not play the movement", exc_info=True)


async def _say_goodbye(furby, tts, audio, settings: Settings) -> None:
    """Furby says goodbye out loud, then the speech settles before the mic opens again."""
    try:
        speech = await asyncio.to_thread(tts.synthesize, "Bye for now!")
        seconds = len(speech) / tts.sample_rate
        movement = await _speak_with_movement(furby, "sleepy", audio, speech, tts.sample_rate, seconds, settings.speaker_device)
        if movement is not None:
            await movement  # the goodbye waits for its movement, since the program may exit next
    except Exception:  # a speaker problem should not stop the goodbye
        log.warning("Could not play the goodbye", exc_info=True)
    await asyncio.sleep(ECHO_SETTLE_SECONDS)


async def _go_to_sleep(furby: FurbyRemote | None) -> None:
    """Sleep mode: the eyes' light and antenna go dark and the Furby's own behavior is turned off until the wake word."""
    if furby is None:
        return
    await _set_antenna(furby, (0, 0, 0))
    try:
        await furby.set_backlight(False)
    except Exception:
        log.warning("Could not put the Furby to sleep", exc_info=True)


async def _wait_for_wake_word_or_antenna(wake, mic, furby: FurbyRemote | None) -> str:
    """Returns "voice" when the wake word is heard or "antenna" when the antenna is touched, whichever comes first."""
    stop = threading.Event()
    voice = asyncio.create_task(asyncio.to_thread(wake.wait_for_wake, mic, None, stop))
    touch = asyncio.create_task(_antenna_touched(furby))
    try:
        done, _pending = await asyncio.wait({voice, touch}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        stop.set()
        touch.cancel()
        await asyncio.gather(voice, touch, return_exceptions=True)
    return "voice" if voice in done else "antenna"


async def _greet(furby, tts, audio, settings: Settings) -> None:
    """Furby asks what it can do, out loud."""
    phrase = pick_greeting()
    print(f"Furby: {phrase}")
    try:
        speech = await asyncio.to_thread(tts.synthesize, phrase)
        seconds = len(speech) / tts.sample_rate
        speaking = asyncio.create_task(asyncio.to_thread(audio.play, speech, tts.sample_rate, settings.speaker_device))
        if furby is not None:
            # The greeting always moves the mouth, starting once the audio is going.
            await asyncio.sleep(MOVEMENT_DELAY_SECONDS)
            await asyncio.gather(_mouth_for(furby, seconds - MOVEMENT_DELAY_SECONDS), speaking)
        else:
            await speaking
    except Exception:  # a speaker problem should not stop the conversation
        log.warning("Could not play the greeting", exc_info=True)
    await asyncio.sleep(ECHO_SETTLE_SECONDS)


async def _mouth_for(furby: FurbyRemote, seconds: float) -> None:
    """Moves the mouth with one of the approved mouth actions for about `seconds`."""
    try:
        await furby.act_for(pick_mouth_action(), max(seconds, 0.5))
    except Exception:  # a movement failure should not stop the greeting
        log.warning("Could not move the mouth", exc_info=True)


def _ready_chime() -> np.ndarray:
    """Two short rising tones. Needs no network or Bluetooth, so it shows startup has finished either way."""
    pieces = []
    for frequency in (880.0, 1320.0):
        t = np.arange(int(READY_RATE * 0.15)) / READY_RATE
        pieces.append(0.3 * np.sin(2 * np.pi * frequency * t).astype(np.float32))
        pieces.append(np.zeros(int(READY_RATE * 0.05), dtype=np.float32))
    return np.concatenate(pieces)


async def _watch_link(furby: FurbyRemote, tts, audio, settings: Settings) -> None:
    """At startup, and whenever the Furby drops out of reach: asks for an antenna touch out loud and retries
    the connection every few seconds for a while."""
    while not furby.link_reported:
        await asyncio.sleep(0.5)  # wait for the first connection report from the Bluetooth child
    while True:
        await asyncio.sleep(1.0)
        if furby.link_up:
            continue
        print("(Furby is not connected)")
        await _say(LINK_PROMPT, tts, audio, settings)
        deadline = time.monotonic() + LINK_RETRY_WINDOW_SECONDS
        while not furby.link_up and time.monotonic() < deadline:
            furby.reconnect()
            await asyncio.sleep(LINK_RETRY_SECONDS)
        # After the retry window, stay quiet until the link is back, so the prompt isn't repeated every second.
        while not furby.link_up:
            await asyncio.sleep(1.0)
        print("(Furby reconnected)")


async def _watch_sleep_mask(furby: FurbyRemote, mask_worn: asyncio.Event) -> None:
    """Tracks the sleep mask: on sets `mask_worn`, off clears it."""
    while True:
        await asyncio.sleep(0.2)
        for event in furby.drain_mask_events():
            if event == "mask:on":
                print("(sleep mask on)")
                mask_worn.set()
            elif event == "mask:off":
                mask_worn.clear()


async def _antenna_touched(furby: FurbyRemote | None) -> None:
    """Returns after the antenna is moved twice within ANTENNA_WINDOW_SECONDS. Never returns without furby."""
    if furby is None:
        await asyncio.Event().wait()
        return
    recent: list[float] = []
    while True:
        await asyncio.sleep(0.1)
        now = time.monotonic()
        for raw in furby.drain_sensor_readings():
            if is_input(raw):
                recent.append(now)
        recent = [t for t in recent if now - t <= ANTENNA_WINDOW_SECONDS]
        if len(recent) >= ANTENNA_TOUCHES_NEEDED:
            print("(antenna touched)")
            return


async def _wake_up(furby: FurbyRemote | None) -> None:
    if furby is None:
        return
    try:
        await furby.set_backlight(True)
    except Exception:
        log.warning("Could not turn the Furby's eyes on", exc_info=True)


_antenna_tasks: set[asyncio.Task] = set()  # keeps the color commands alive until they finish


async def _set_antenna(furby: FurbyRemote | None, color: tuple[int, int, int]) -> None:
    """Sends the color without waiting for it. A dropped Bluetooth link can take about 20 seconds to time out,
    and the conversation (including the listening window) must not stall on it."""
    if furby is None:
        return
    task = asyncio.create_task(_send_antenna(furby, color))
    _antenna_tasks.add(task)
    task.add_done_callback(_antenna_tasks.discard)


async def _send_antenna(furby: FurbyRemote, color: tuple[int, int, int]) -> None:
    try:
        await furby.set_antenna_color(*color)
    except Exception:  # the Furby is a nice-to-have; a dropped BLE link should not stop the conversation
        log.warning("Could not set Furby antenna color", exc_info=True)
