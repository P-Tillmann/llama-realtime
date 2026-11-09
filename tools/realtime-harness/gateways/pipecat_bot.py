#
# Spike: Pipecat as the WebRTC front of llama-server's /v1/realtime.
# Browser --WebRTC (SmallWebRTC, aiortc)--> this bot --WebSocket (OpenAI Realtime protocol)--> llama-server
# Adapted from pipecat examples/realtime/realtime-openai.py (BSD 2-Clause, Daily).
#

import os

from loguru import logger

from pipecat.frames.frames import LLMRunFrame
from pipecat.observers.loggers.transcription_log_observer import TranscriptionLogObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker, ProcessorUnusablePolicy
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    AssistantTurnStoppedMessage,
    LLMContextAggregatorPair,
    UserTurnMessageAddedMessage,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.openai.realtime.events import (
    AudioConfiguration,
    AudioInput,
    InputAudioTranscription,
    SemanticTurnDetection,
    TurnDetection,
    SessionProperties,
)
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.workers.runner import WorkerRunner

transport_params = {
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}

GREET = os.environ.get("GREET", "1") == "1"


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
    llm = OpenAIRealtimeLLMService(
        api_key=os.environ.get("LLAMA_API_KEY", "none"),
        base_url=os.environ.get("LLAMA_REALTIME_URL", "ws://127.0.0.1:8080/v1/realtime"),
        settings=OpenAIRealtimeLLMService.Settings(
            model="local",
            system_instruction="You are a friendly voice assistant. Answer in one or two short sentences.",
            session_properties=SessionProperties(
                audio=AudioConfiguration(
                    input=AudioInput(
                        transcription=InputAudioTranscription() if os.environ.get('TRANSCRIBE', '1') == '1' else None,
                        turn_detection=SemanticTurnDetection() if os.environ.get('TURN', 'semantic') == 'semantic' else TurnDetection(type='server_vad', silence_duration_ms=500),
                    )
                ),
            ),
        ),
    )
    context = LLMContext([{"role": "developer", "content": "Say hello!"}] if GREET else [])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
    pipeline = Pipeline([transport.input(), user_aggregator, llm, transport.output(), assistant_aggregator])
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[TranscriptionLogObserver()],
        processor_unusable_policy=ProcessorUnusablePolicy.END,
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("SPIKE client connected")
        if GREET:
            await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("SPIKE client disconnected")
        await runner.cancel()

    @user_aggregator.event_handler("on_user_turn_message_added")
    async def on_user(aggregator, message: UserTurnMessageAddedMessage):
        logger.info(f"SPIKE user: {message.content}")

    @assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant(aggregator, message: AssistantTurnStoppedMessage):
        logger.info(f"SPIKE assistant: {message.content}")

    await runner.run()


async def bot(runner_args: RunnerArguments):
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
