from typing import List, Tuple
import time

from vllm import EngineArgs, LLMEngine, RequestOutput, SamplingParams, EngineArgs
from vllm.utils import FlexibleArgumentParser


def process_requests(engine: LLMEngine,
                     test_prompts: List[Tuple[str, SamplingParams]]):
    """Continuously process a list of prompts and handle the outputs."""
    timeLast = None
    while test_prompts or engine.has_unfinished_requests():
      if test_prompts and (timeLast is None or time.time() - timeLast > 5):
        prompt, request_id, is_first_chunk, action = test_prompts.pop(0)
        if action == "prefill":
          if is_first_chunk:
            engine.run_first_add_chunk(request_id, prompt)
          else:
            engine.run_add_chunk(request_id, prompt)
        elif action == "decode":
          engine.run_decode(request_id)
        elif action == "vanilla":
          engine.add_request(request_id, prompt, SamplingParams(max_tokens=50, temperature=0, seed=42))
        timeLast = time.time()
    

      request_outputs: List[RequestOutput] = engine.step()

      for request_output in request_outputs:
        if request_output.finished:
          print("Generated:", request_output.outputs[0].text)


def initialize_engine() -> LLMEngine:
    """Initialize the LLMEngine from the command line arguments."""
    # engine_args = EngineArgs.from_cli_args(args)
    engine_args = EngineArgs(model="meta-llama/Llama-3.1-8B-Instruct")
    return LLMEngine.from_engine_args(engine_args)


def main():
    """Main function that sets up and runs the prompt processing."""
    engine = initialize_engine()
    test_prompts = [
      ("To be or not to be, ", "1", True, "prefill"),
      ("that is the ", "1", False, "prefill"),
      (None, "1", False, "decode"),
      # ("To be or not to be, that is the ", "1", True, "vanilla")
    ]
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    main()