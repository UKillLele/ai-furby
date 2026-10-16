"""Listen, think, speak. The Furby is optional: without FURBY_ADDRESS the assistant runs audio-only."""

from __future__ import annotations

import asyncio
import contextlib
import logging

from ai_furby.assistant import Assistant
from ai_furby.config import Settings
from ai_furby.furby_controller import FurbyController
from ai_furby.speech_to_text import SpeechToText
from ai_furby.text_to_speech import TextToSpeech

log = logging.getLogger(__name__)

STOP_PHRASES = ("goodbye furby", "bye furby", "stop listening")
LISTENING_COLOR = (0, 255, 0)
# Room echo of Furby's own voice fades after playback ends; wait so the mic does not hear and answer it.
ECHO_SETTLE_SECONDS = 1.0
SPEAKING_COLOR = (0, 0, 255)


async def run(settings: Settings) -> None:
    stt = SpeechToText(settings)
    tts = TextToSpeech(settings)
    assistant = Assistant(settings)

    async with contextlib.AsyncExitStack() as stack:
        furby = None
        if settings.furby_address:
            furby = await stack.enter_async_context(FurbyController(settings))
            # Stop the Furby's own idle behavior so it only acts on this assistant's commands.
            await furby.set_group_active(False)
            stack.push_async_callback(_restore_autonomy, furby)

        # Imported only after the Furby is connected: on Windows, loading sounddevice first makes the
        # Bluetooth connection fail with "Thread is configured for Windows GUI", and it also breaks playback.
        from ai_furby import audio


        print("Listening. Say 'goodbye Furby' or press Ctrl+C to stop.")
        while True:
            await _keep_furby_quiet(furby)
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
            if heard.lower().strip(" .!?") in STOP_PHRASES:
                print("Furby: Bye for now!")
                break

            try:
                reply = await asyncio.to_thread(assistant.reply, heard)
            except Exception as error:  # a failed API call should not end the conversation
                log.exception("Claude request failed")
                print(f"Furby: (could not think of a reply: {type(error).__name__})")
                continue
            print(f"Furby: {reply}")

            await _set_antenna(furby, SPEAKING_COLOR)
            try:
                speech = await asyncio.to_thread(tts.synthesize, reply)
                await _play_with_retry(audio.play, speech, tts.sample_rate, settings.speaker_device)
            except Exception as error:  # a speaker hiccup should not end the conversation; the reply is still printed
                log.exception("Playback failed")
                print(f"(could not play speech: {type(error).__name__}; check the headset is connected)")
            await asyncio.sleep(ECHO_SETTLE_SECONDS)


async def _keep_furby_quiet(furby: FurbyController | None) -> None:
    """The Furby seems to resume its own behavior over time, so re-assert the group-off command each turn."""
    if furby is None:
        return
    try:
        await furby.set_group_active(False)
    except Exception:
        log.warning("Could not keep the Furby quiet", exc_info=True)


async def _play_with_retry(play, speech, sample_rate: int, speaker: str) -> None:
    """Headsets can briefly refuse playback right after the mic has been in use, so try once more."""
    try:
        await asyncio.to_thread(play, speech, sample_rate, speaker)
    except Exception:
        log.warning("Playback failed once; retrying", exc_info=True)
        await asyncio.sleep(1.5)
        await asyncio.to_thread(play, speech, sample_rate, speaker)


async def _restore_autonomy(furby: FurbyController) -> None:
    try:
        await furby.set_group_active(True)
    except Exception:  # best effort on the way out; the Furby also recovers when it is power cycled
        log.warning("Could not restore the Furby's autonomous behavior", exc_info=True)


async def _set_antenna(furby: FurbyController | None, color: tuple[int, int, int]) -> None:
    if furby is None:
        return
    try:
        await furby.set_antenna_color(*color)
    except Exception:  # the Furby is a nice-to-have; a dropped BLE link should not stop the conversation
        log.warning("Could not set Furby antenna color", exc_info=True)
