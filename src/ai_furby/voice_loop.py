"""Listen, think, speak. The Furby is optional: without FURBY_ADDRESS the assistant runs audio-only."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re

from ai_furby.assistant import Assistant
from ai_furby.config import Settings
from ai_furby.furby_remote import FurbyRemote
from ai_furby.mood_actions import ACTION_SECONDS, BEAK_ACTION, MOOD_ACTIONS
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
MOVEMENT_DELAY_SECONDS = 0.3  # how long audio leads the Furby's movement; tune by listening and watching
SPEAKING_COLOR = (0, 0, 255)


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

        print("Listening. Say 'goodbye Furby' or press Ctrl+C to stop.")
        while True:
            await _set_antenna(furby, LISTENING_COLOR)
            utterance = await asyncio.to_thread(
                audio.record_utterance, threshold=settings.mic_threshold, device=settings.mic_device
            )
            if utterance.size == 0:
                continue

            heard = await asyncio.to_thread(stt.transcribe, utterance)
            if not heard:
                continue
            print(f"You: {heard}")
            if is_stop_phrase(heard):
                print("Furby: Bye for now!")
                break

            try:
                reply = await asyncio.to_thread(assistant.reply, heard)
            except Exception as error:  # a failed API call should not end the conversation
                log.exception("Claude request failed")
                print(f"Furby: (could not think of a reply: {type(error).__name__})")
                continue
            print(f"Furby: {reply}")
            log.info("Mood: %s", assistant.last_mood)

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
            await furby.act_for(BEAK_ACTION, seconds - first)
    except Exception:  # an action failure should not stop the reply
        log.warning("Could not play the movement", exc_info=True)


async def _set_antenna(furby: FurbyRemote | None, color: tuple[int, int, int]) -> None:
    if furby is None:
        return
    try:
        await furby.set_antenna_color(*color)
    except Exception:  # the Furby is a nice-to-have; a dropped BLE link should not stop the conversation
        log.warning("Could not set Furby antenna color", exc_info=True)
