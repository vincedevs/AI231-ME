from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from .audio import AudioFramer
from .config import AlfredConfig
from .contracts import ApplicationState, Decision, PolicyOutcome
from .policy import apply_policy

LOGGER = logging.getLogger(__name__)
EASTER_EGG_EVENTS = tuple(f"easter_egg_{number}" for number in range(1, 6))


class AlfredApplication:
    """Single-threaded orchestration; audio callbacks only populate the input queue."""

    def __init__(
        self,
        *,
        config: AlfredConfig,
        wake_detector: Any,
        command_capture: Any,
        vcm: Any,
        feedback: Any,
        actions: Any,
        clarification_capture: Any | None = None,
        slot_clarifier: Any | None = None,
        speech_recognizer: Any | None = None,
        message_follow_up: Any | None = None,
        weather_asr_enabled: bool | None = None,
        benchmark_logger: Any | None = None,
        benchmark_mode: bool = False,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.wake_detector = wake_detector
        self.command_capture = command_capture
        self.clarification_capture = clarification_capture
        self.slot_clarifier = slot_clarifier
        self.speech_recognizer = speech_recognizer
        self.message_follow_up = message_follow_up
        self.weather_asr_enabled = (
            speech_recognizer is not None if weather_asr_enabled is None else weather_asr_enabled
        )
        self.vcm = vcm
        self.feedback = feedback
        self.actions = actions
        self.benchmark_logger = benchmark_logger
        self.benchmark_mode = benchmark_mode
        self.sleep = sleep
        self.easter_egg_index = 0
        self.state = ApplicationState.STARTING
        self.transitions: list[ApplicationState] = [self.state]
        self.wake_framer = AudioFramer(int(config.document["wake_word"]["chunk_samples"]))

    def transition(self, state: ApplicationState) -> None:
        LOGGER.info("state=%s previous=%s", state, self.state)
        self.state = state
        self.transitions.append(state)

    def _cooldown(self, microphone: Any) -> None:
        self.transition(ApplicationState.COOLDOWN)
        self.sleep(float(self.config.document["runtime"]["post_feedback_cooldown_seconds"]))
        microphone.clear()
        self.wake_framer.clear()
        self.wake_detector.reset()
        self.transition(ApplicationState.IDLE)

    def _finish_interaction(self, microphone: Any) -> None:
        """Restore media only after all command feedback has finished playing."""
        self.actions.resume_media_after_interaction()
        self._cooldown(microphone)

    def _finish_benchmark_interaction(self, microphone: Any) -> None:
        """Reset immediately without feedback, media control, or action behavior."""
        microphone.clear()
        self.wake_framer.clear()
        self.wake_detector.reset()
        self.transition(ApplicationState.IDLE)

    def _handle_easter_egg(self, microphone: Any) -> None:
        self.transition(ApplicationState.EASTER_EGG)
        event = EASTER_EGG_EVENTS[self.easter_egg_index]
        self.easter_egg_index = (self.easter_egg_index + 1) % len(EASTER_EGG_EVENTS)
        self.feedback.play(event)
        self._finish_interaction(microphone)

    def _handle_policy_outcome(self, outcome: PolicyOutcome, microphone: Any) -> None:
        if outcome.decision is Decision.UNSUPPORTED:
            LOGGER.info("command_rejected decision=unsupported reason=%s", outcome.reason)
            self.transition(ApplicationState.PLAYING_FEEDBACK)
            self.feedback.play("unsupported")
            self._finish_interaction(microphone)
            return
        if outcome.decision is Decision.LOW_CONFIDENCE:
            LOGGER.info("command_rejected decision=low_confidence reason=%s", outcome.reason)
            self.transition(ApplicationState.PLAYING_FEEDBACK)
            self.feedback.play("low_confidence")
            self._finish_interaction(microphone)
            return

        request = outcome.action_request
        if request is None:
            raise RuntimeError("Execute policy outcome did not contain an action request")
        self.transition(ApplicationState.ANNOUNCING_ACTION)
        try:
            self.feedback.announce(request.intent, request.slots)
        except Exception:
            LOGGER.exception("action_announcement_failed intent=%s", request.intent)
            self.transition(ApplicationState.PLAYING_FEEDBACK)
            self.feedback.play("action_failure")
            self._finish_interaction(microphone)
            return
        microphone_released = False
        if request.intent == "CALL":
            self.transition(ApplicationState.IN_CALL)
            try:
                microphone.stop()
            except Exception:
                LOGGER.exception("microphone_release_failed_before_call")
                self.transition(ApplicationState.PLAYING_FEEDBACK)
                self.feedback.play("action_failure")
                self._finish_interaction(microphone)
                return
            else:
                microphone_released = True
        else:
            self.transition(ApplicationState.EXECUTING_ACTION)
        try:
            result = self.actions.execute(request)
        except Exception:
            LOGGER.exception("action_failed intent=%s handler=%s", request.intent, request.handler)
            self.transition(ApplicationState.PLAYING_FEEDBACK)
            self.feedback.play("action_failure")
        else:
            LOGGER.info(
                "action_complete intent=%s handler=%s duration_ms=%.3f message=%s",
                request.intent,
                request.handler,
                result.duration_ms,
                result.message,
            )
            self.transition(ApplicationState.PLAYING_FEEDBACK)
            response_spoken = False
            try:
                self.feedback.respond(result.message)
            except Exception:
                LOGGER.exception("action_response_failed intent=%s", request.intent)
            else:
                response_spoken = bool(getattr(self.feedback, "spoken_responses", False))
            if not response_spoken:
                feedback_name = "action_success" if result.status == "success" else "action_failure"
                self.feedback.play(feedback_name)
        finally:
            if microphone_released:
                try:
                    microphone.start()
                except Exception:
                    LOGGER.exception("microphone_restart_failed_after_call")
        self._finish_interaction(microphone)

    def _clarify_slot(self, result: Any, microphone: Any) -> Any | None:
        if self.slot_clarifier is None or self.clarification_capture is None:
            return result
        slot_name = self.slot_clarifier.required_slot(result)
        if slot_name is None:
            return result

        LOGGER.info("clarification_required intent=%s slot=%s", result.intent, slot_name)
        self.transition(ApplicationState.REQUESTING_SLOT)
        try:
            self.feedback.respond(self.slot_clarifier.prompt(result.intent))
        except Exception:
            LOGGER.exception("clarification_prompt_failed slot=%s", slot_name)
            return None
        self.transition(ApplicationState.PLAYING_CLARIFICATION_CUE)
        self.feedback.play("listening_start")
        microphone.clear()
        self.transition(ApplicationState.WAITING_FOR_SLOT)
        capture = self.clarification_capture.capture(
            lambda: microphone.read(timeout=1.0),
            on_speech_start=lambda: self.transition(ApplicationState.CAPTURING_SLOT),
        )
        if capture.waveform is None:
            LOGGER.info("clarification_capture_failed reason=%s", capture.reason)
            self.feedback.play("listening_end")
            return None
        self.feedback.play("listening_end")
        microphone.clear()
        self.transition(ApplicationState.TRANSCRIBING_SLOT)
        try:
            transcript = self.slot_clarifier.transcribe(capture.waveform)
        except Exception:
            LOGGER.exception("clarification_transcription_failed slot=%s", slot_name)
            return None
        LOGGER.info("clarification_transcript slot=%s characters=%d", slot_name, len(transcript))
        self.transition(ApplicationState.VALIDATING_SLOT)
        clarified = self.slot_clarifier.apply(result, transcript)
        if clarified is None:
            LOGGER.info("clarification_rejected slot=%s", slot_name)
        return clarified

    def _handle_clarification_failure(self, microphone: Any) -> None:
        self.transition(ApplicationState.PLAYING_FEEDBACK)
        response_spoken = False
        try:
            self.feedback.respond("I couldn't understand that value, so I won't execute it.")
        except Exception:
            LOGGER.exception("clarification_failure_response_failed")
        else:
            response_spoken = bool(getattr(self.feedback, "spoken_responses", False))
        if not response_spoken:
            self.feedback.play("low_confidence")
        self._finish_interaction(microphone)

    def _collect_weather_location(self, microphone: Any) -> str | None:
        if self.speech_recognizer is None or self.clarification_capture is None:
            LOGGER.warning("weather_location_follow_up_unavailable")
            return None
        weather_config = self.config.document["actions"]["weather"]
        self.transition(ApplicationState.REQUESTING_WEATHER_LOCATION)
        try:
            self.feedback.respond(str(weather_config["location_prompt"]))
        except Exception:
            LOGGER.exception("weather_location_prompt_failed")
            return None
        self.feedback.play("listening_start")
        microphone.clear()
        self.transition(ApplicationState.WAITING_FOR_WEATHER_LOCATION)
        capture = self.clarification_capture.capture(
            lambda: microphone.read(timeout=1.0),
            on_speech_start=lambda: self.transition(ApplicationState.CAPTURING_WEATHER_LOCATION),
        )
        if capture.waveform is None:
            LOGGER.info("weather_location_capture_failed reason=%s", capture.reason)
            self.feedback.play("listening_end")
            return None
        self.feedback.play("listening_end")
        microphone.clear()
        self.transition(ApplicationState.TRANSCRIBING_WEATHER_LOCATION)
        try:
            transcript = self.speech_recognizer.transcribe(capture.waveform)
        except Exception:
            LOGGER.exception("weather_location_transcription_failed")
            return None
        location = " ".join(transcript.strip().split())
        maximum = int(weather_config["maximum_location_characters"])
        if len(location) < 2 or len(location) > maximum:
            LOGGER.info("weather_location_rejected characters=%d", len(location))
            return None
        LOGGER.info("weather_location_transcribed characters=%d", len(location))
        return location

    def _capture_message_field(
        self,
        *,
        prompt: str,
        microphone: Any,
        requesting_state: ApplicationState,
        waiting_state: ApplicationState,
        capturing_state: ApplicationState,
        transcribing_state: ApplicationState,
    ) -> str | None:
        self.transition(requesting_state)
        try:
            self.feedback.respond(prompt)
        except Exception:
            LOGGER.exception("message_follow_up_prompt_failed")
            return None
        self.feedback.play("listening_start")
        microphone.clear()
        self.transition(waiting_state)
        capture = self.clarification_capture.capture(
            lambda: microphone.read(timeout=1.0),
            on_speech_start=lambda: self.transition(capturing_state),
        )
        if capture.waveform is None:
            LOGGER.info("message_follow_up_capture_failed reason=%s", capture.reason)
            self.feedback.play("listening_end")
            return None
        self.feedback.play("listening_end")
        microphone.clear()
        self.transition(transcribing_state)
        try:
            transcript = self.speech_recognizer.transcribe(capture.waveform)
        except Exception:
            LOGGER.exception("message_follow_up_transcription_failed")
            return None
        LOGGER.info("message_follow_up_transcript characters=%d", len(transcript))
        return transcript

    def _collect_message_payload(self, microphone: Any) -> dict[str, str] | None:
        if (
            self.message_follow_up is None
            or self.speech_recognizer is None
            or self.clarification_capture is None
        ):
            LOGGER.error("message_follow_up_unavailable")
            return None
        recipient_transcript = self._capture_message_field(
            prompt=self.message_follow_up.recipient_prompt,
            microphone=microphone,
            requesting_state=ApplicationState.REQUESTING_MESSAGE_RECIPIENT,
            waiting_state=ApplicationState.WAITING_FOR_MESSAGE_RECIPIENT,
            capturing_state=ApplicationState.CAPTURING_MESSAGE_RECIPIENT,
            transcribing_state=ApplicationState.TRANSCRIBING_MESSAGE_RECIPIENT,
        )
        if recipient_transcript is None:
            return None
        recipient = self.message_follow_up.recipient(recipient_transcript)
        if recipient is None:
            LOGGER.info("message_recipient_rejected")
            return None
        message_transcript = self._capture_message_field(
            prompt=self.message_follow_up.message_prompt,
            microphone=microphone,
            requesting_state=ApplicationState.REQUESTING_MESSAGE_CONTENT,
            waiting_state=ApplicationState.WAITING_FOR_MESSAGE_CONTENT,
            capturing_state=ApplicationState.CAPTURING_MESSAGE_CONTENT,
            transcribing_state=ApplicationState.TRANSCRIBING_MESSAGE_CONTENT,
        )
        if message_transcript is None:
            return None
        self.transition(ApplicationState.VALIDATING_MESSAGE)
        payload = self.message_follow_up.compose(recipient, message_transcript)
        if payload is None:
            LOGGER.info("message_content_rejected")
            return None
        return payload.as_mapping()

    def _handle_message_failure(self, microphone: Any) -> None:
        self.transition(ApplicationState.PLAYING_FEEDBACK)
        response_spoken = False
        try:
            self.feedback.respond("I couldn't prepare your message, so nothing was sent.")
        except Exception:
            LOGGER.exception("message_follow_up_failure_response_failed")
        else:
            response_spoken = bool(getattr(self.feedback, "spoken_responses", False))
        if not response_spoken:
            self.feedback.play("action_failure")
        self._finish_interaction(microphone)

    def _handle_primary_wake(self, microphone: Any) -> None:
        if self.benchmark_logger is not None:
            try:
                self.benchmark_logger.wake()
            except Exception:
                LOGGER.exception("benchmark_wake_log_failed")
                if self.benchmark_mode:
                    raise
        self.transition(ApplicationState.WAKE_DETECTED)
        if not self.benchmark_mode:
            self.transition(ApplicationState.PLAYING_LISTENING_CUE)
            self.feedback.play("listening_start")
        microphone.clear()

        self.transition(ApplicationState.WAITING_FOR_SPEECH)
        capture = self.command_capture.capture(
            lambda: microphone.read(timeout=1.0),
            on_speech_start=lambda: self.transition(ApplicationState.CAPTURING_COMMAND),
        )
        if capture.waveform is None:
            LOGGER.info("command_capture_failed reason=%s", capture.reason)
            if self.benchmark_mode:
                self._finish_benchmark_interaction(microphone)
                return
            self.transition(ApplicationState.PLAYING_FEEDBACK)
            self.feedback.play("listening_end")
            self._finish_interaction(microphone)
            return

        if not self.benchmark_mode:
            self.feedback.play("listening_end")
        microphone.clear()
        self.transition(ApplicationState.INFERRING)
        inference_started = time.perf_counter()
        result = self.vcm.predict(capture.waveform)
        inference_ms = (time.perf_counter() - inference_started) * 1_000.0
        if self.benchmark_logger is not None:
            try:
                sample_rate = int(self.config.document["audio"]["sample_rate"])
                samples = int(getattr(capture.waveform, "size", len(capture.waveform)))
                self.benchmark_logger.command(
                    result,
                    infer_ms=inference_ms,
                    audio_ms=(samples / sample_rate) * 1_000.0,
                )
            except Exception:
                LOGGER.exception("benchmark_command_log_failed")
                if self.benchmark_mode:
                    raise
        LOGGER.info(
            "vcm_result decision=%s intent=%s confidence=%.4f in_scope=%s",
            result.decision,
            result.intent,
            result.intent_confidence,
            result.in_scope_score,
        )
        if self.benchmark_mode:
            self._finish_benchmark_interaction(microphone)
            return
        result = self._clarify_slot(result, microphone)
        if result is None:
            self._handle_clarification_failure(microphone)
            return
        action_payload: dict[str, str] | None = None
        if result.decision is Decision.EXECUTE and result.intent == "WEATHER":
            if (
                not self.weather_asr_enabled
                or self.speech_recognizer is None
                or self.clarification_capture is None
            ):
                LOGGER.info("weather_location_follow_up_skipped reason=weather_asr_disabled")
                action_payload = {}
            else:
                location = self._collect_weather_location(microphone)
                action_payload = {} if location is None else {"location_query": location}
        if result.decision is Decision.EXECUTE and result.intent == "MESSAGE":
            if (
                self.message_follow_up is None
                or self.speech_recognizer is None
                or self.clarification_capture is None
            ):
                LOGGER.info("message_follow_up_skipped reason=asr_disabled")
                action_payload = {}
            else:
                action_payload = self._collect_message_payload(microphone)
                if action_payload is None:
                    self._handle_message_failure(microphone)
                    return
        self.transition(ApplicationState.CHECKING_POLICY)
        outcome = apply_policy(result, self.config)
        if action_payload is not None and outcome.action_request is not None:
            outcome = replace(
                outcome,
                action_request=replace(outcome.action_request, payload=action_payload),
            )
        self._handle_policy_outcome(outcome, microphone)

    def process_wake_event(self, event: str, microphone: Any) -> bool:
        if event not in {"im_batman", "hey_alfred"}:
            raise ValueError(f"Unknown wake event: {event}")
        if self.benchmark_mode and event == "im_batman":
            self._finish_benchmark_interaction(microphone)
            return False
        # Both wake words must be audible over music in normal mode. Benchmark
        # mode must not issue Spotify or any other action-layer command.
        if not self.benchmark_mode:
            self.actions.pause_media_for_interaction()
        try:
            if event == "im_batman":
                self._handle_easter_egg(microphone)
                return False
            if event == "hey_alfred":
                self._handle_primary_wake(microphone)
                return True
        except Exception:
            # Normal paths restore or intentionally cancel playback in
            # _finish_interaction. This recovery path prevents an unexpected
            # audio or feedback failure from leaving Spotify paused forever.
            if not self.benchmark_mode:
                self.actions.resume_media_after_interaction()
            raise

    def run(self, microphone: Any, once: bool = False) -> None:
        self.transition(ApplicationState.IDLE)
        while True:
            block = microphone.read(timeout=1.0)
            for wake_frame in self.wake_framer.push(block):
                event = self.wake_detector.process(wake_frame)
                if event is None:
                    continue
                handled_command = self.process_wake_event(event, microphone)
                if once and handled_command:
                    self.transition(ApplicationState.STOPPING)
                    return
                break
