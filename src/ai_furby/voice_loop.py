"""Listen, think, speak. The Furby is optional: without FURBY_ADDRESS the assistant runs audio-only."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
import time

from ai_furby.antenna_touch import is_input
from ai_furby.assistant import Assistant
from ai_furby.config import Settings
from ai_furby.furby_remote import FurbyRemote
from ai_furby.mood_actions import ACTION_SECONDS, MOOD_ACTIONS, pick_greeting, pick_mouth_action
from ai_furby.speech_to_text import SpeechToText
from ai_furby.text_to_speech import TextToSpeech

log = logging.getLogger(__name__)

STOP_PHRASES = ("goodbye furby", "bye furby", "stop listening")


def is_stop_phrase(heard: str) -> bool:
    """Ignores case, punctuation, and extra spaces, so "Goodbye, Furby!" matches."""
    words = re.sub(r"[^a-z ]", " ", heard.lower())
    return " ".join(words.split()) in STOP_PHRASES
LISTENING_COLOR = (0, 255, 0)
# Room echo of Furby's own voice fades after playback ends; wait so the mic does not hear and answer it.
ECHO_SETTLE_SECONDS = 1.0
ANTENNA_WINDOW_SECONDS = 1.5  # antenna readings this close together count as one touch
ANTENNA_TOUCHES_NEEDED = 2  # antenna readings needed within the window; one reading was too easy to trigger by accident
FOLLOW_UP_SECONDS = 10.0
WAKE_FOLLOW_UP_SECONDS = 6.0  # after the wake word, how long to wait for something to say before greeting  # after a reply, the next thing you say can come within this window without the wake word
# After the mic closes, Windows can still be switching the Bluetooth headset; give it a moment before playing.
MIC_RELEASE_SECONDS = 0.5
MOVEMENT_DELAY_SECONDS = 0.3  # how long audio leads the Furby's movement; tune by listening and watching
SPEAKING_COLOR = (0, 0, 255)
THINKING_COLOR = (255, 160, 0)  # amber while Claude is working on a reply


async def run(settings: Settings) -> None:
    stt = SpeechToText(settings)
    tts = TextToSpeech(settings)
    assistant = Assistant(settings)

    async with contextlib.AsyncExitStack() as stack:
        furby = None
        if settings.furby_address:
            furby = await stack.enter_async_context(FurbyRemote(settings))

        # Imported only after the Furby is connected: on Windows, loading sounddevice first makes the
        # Bluetooth connection fail with "Thread is configured for Windows GUI".
        from ai_furby import audio

        if furby is None:
            print("Furby: not connected (FURBY_ADDRESS is not set in this terminal), so it won't move.")
        else:
            print(f"Furby: connected to {settings.furby_address}")
        wake = None
        if settings.wake_word_enabled:
            from ai_furby.wake_word import WakeWordListener, default_model_path

            wake = WakeWordListener(default_model_path(), threshold=settings.wake_threshold, device=settings.mic_device)
            print("Say 'Hey Jarvis' (stand-in for 'Hey Furby') to start talking. Say 'goodbye Furby' to stop.")
        else:
            print("Listening. Say 'goodbye Furby' or press Ctrl+C to stop.")
        conversing = False  # True while a conversation is ongoing and the wake word isn't required
        while True:
            woke_by = None  # "voice" or "antenna" right after a wake; None otherwise
            if wake is not None and not conversing:
                await _go_to_sleep(furby)
                woke_by = await _wait_for_wake_word_or_antenna(wake, furby)
                print(f"(woken by {woke_by})")
                await _wake_up(furby)
                if woke_by == "antenna":
                    await _greet(furby, tts, audio, settings)
                    conversing = True
                    woke_by = None  # the greeting already answered the wake
            await _set_antenna(furby, LISTENING_COLOR)
            # During a conversation, wait up to FOLLOW_UP_SECONDS for the next thing you say; silence ends it.
            if woke_by == "voice":
                wait = WAKE_FOLLOW_UP_SECONDS
            else:
                wait = FOLLOW_UP_SECONDS if conversing else 30.0
            utterance = await asyncio.to_thread(
                audio.record_utterance, threshold=settings.mic_threshold, device=settings.mic_device, wait_seconds=wait
            )
            if utterance.size == 0:
                if woke_by == "voice":
                    # The wake word came with nothing after it: greet, then keep the conversation open.
                    await _greet(furby, tts, audio, settings)
                    conversing = True
                    continue
                if conversing:
                    print("(quiet for a while; say the wake word to talk again)")
                conversing = False
                continue
            conversing = True

            heard = await asyncio.to_thread(stt.transcribe, utterance)
            if not heard:
                # Background noise can pass the mic threshold without any words. That must not keep the
                # conversation open, or the Furby never goes back to sleep.
                conversing = False
                continue
            print(f"You: {heard}")
            if is_stop_phrase(heard):
                print("Furby: Bye for now!")
                await _say_goodbye(furby, tts, audio, settings)
                conversing = False
                if settings.wake_word_enabled:
                    continue  # back to waiting for the wake word
                break

            await _set_antenna(furby, THINKING_COLOR)  # lit only while the reply is being worked out
            try:
                reply = await asyncio.to_thread(assistant.reply, heard)
            except Exception as error:  # a failed API call should not end the conversation
                log.exception("Claude request failed")
                print(f"Furby: (could not think of a reply: {type(error).__name__})")
                await _set_antenna(furby, (0, 0, 0))
                continue
            print(f"Furby: {reply}")
            log.info("Mood: %s", assistant.last_mood)

            await asyncio.sleep(MIC_RELEASE_SECONDS)
            await _set_antenna(furby, SPEAKING_COLOR)
            speech = await asyncio.to_thread(tts.synthesize, reply)
            seconds = len(speech) / tts.sample_rate
            try:
                await _speak_with_movement(
                    furby, assistant.last_mood, audio, speech, tts.sample_rate, seconds, settings.speaker_device
                )
            except Exception as error:  # a speaker problem should not end the conversation
                log.exception("Playback failed")
                print(f"(could not play speech: {type(error).__name__}; check the speakers)")
            await asyncio.sleep(ECHO_SETTLE_SECONDS)


async def _speak_with_movement(
    furby: FurbyRemote | None,
    mood: str | None,
    audio,
    speech,
    sample_rate: int,
    seconds: float,
    speaker_device: str | None,
) -> None:
    """Plays the reply while the Furby moves for the same length of time."""
    speaking = asyncio.create_task(asyncio.to_thread(audio.play, speech, sample_rate, speaker_device))
    if furby is None or mood is None or mood not in MOOD_ACTIONS:
        await speaking
        return
    # Audio takes a moment to open its stream, and Bluetooth commands take a moment to reach the Furby.
    # Let the audio start first, then start the movement, so the two line up.
    log.info("Movement starting: %s for %.1f s", mood, seconds - MOVEMENT_DELAY_SECONDS)
    await asyncio.sleep(MOVEMENT_DELAY_SECONDS)
    await asyncio.gather(_move_with_beak(furby, MOOD_ACTIONS[mood], seconds - MOVEMENT_DELAY_SECONDS), speaking)


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
        await asyncio.to_thread(audio.play, speech, tts.sample_rate, settings.speaker_device)
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


async def _wait_for_wake_word_or_antenna(wake, furby: FurbyRemote | None) -> str:
    """Returns "voice" when the wake word is heard or "antenna" when the antenna is touched, whichever comes first."""
    stop = threading.Event()
    voice = asyncio.create_task(asyncio.to_thread(wake.wait_for_wake, None, stop))
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
        await asyncio.to_thread(audio.play, speech, tts.sample_rate, settings.speaker_device)
    except Exception:  # a speaker problem should not stop the conversation
        log.warning("Could not play the greeting", exc_info=True)
    await asyncio.sleep(ECHO_SETTLE_SECONDS)


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


async def _set_antenna(furby: FurbyRemote | None, color: tuple[int, int, int]) -> None:
    if furby is None:
        return
    try:
        await furby.set_antenna_color(*color)
    except Exception:  # the Furby is a nice-to-have; a dropped BLE link should not stop the conversation
        log.warning("Could not set Furby antenna color", exc_info=True)
