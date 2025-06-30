# import vllm
# from collections import deque
# from threading import Thread
# from vllm.sampling_params import SamplingParams
# import time

# class dummy_worker:
#   def __init__(self):
#     self.q = deque()
#     engine_args = vllm.EngineArgs(model="meta-llama/Llama-3.1-8B-Instruct")
#     self.engine = vllm.LLMEngine.from_engine_args(engine_args)
#     self.output_file = open("output2.txt", "w")
    
#     def write_output(text):
#         self.output_file.write(text)
#         self.output_file.flush()

#     def start_engine():
#       write_output("Starting engine\n")
#       while True:
#         if len(self.q) > 0:
#           command = self.q.popleft()
#           self.process_command(command)
        
#         outputs = self.engine.step()
#         for output in outputs:
#           if output.finished:
#             for out in output.outputs:
#               write_output("GENERATED: " + out.text + "\n")
#         # write_output("step\n")
    
#     Thread(target=start_engine, daemon=True).start()
  
#   def add_command(self, command):
#     self.q.append(command)

#   def process_command(self, command):
#     prompt, request_id, is_first_chunk, action = command
#     if action == "prefill":
#       if is_first_chunk:
#         self.engine.run_first_add_chunk(request_id, prompt)
#       else:
#         self.engine.run_add_chunk(request_id, prompt)
#     elif action == "decode":
#       self.engine.run_decode(request_id)
#     elif action == "vanilla":
#       self.engine.add_request(request_id, prompt, params=SamplingParams(max_tokens=4096))
#     # elif action == "export":
#     #   self.engine.run_export(request_id)



# if __name__ == "__main__":
#   worker = dummy_worker()
#   worker.add_command(("Hello, ", "1", True, "vanilla"))
#   # worker.add_command(("Hello, ", "1", True, "prefill"))
#   # time.sleep(5)
#   # # worker.add_command(("my name is ", "1", False, "prefill"))
#   # # time.sleep(5)
#   # # worker.add_command(("and I like to ", "1", False, "prefill"))
#   # # worker.add_command(("Bob and I liXke to dance salsa. ", "1", False, "prefill"))
#   # # time.sleep(5)
#   # # print("running queue", worker.engine.scheduler[0].running)
#   # worker.add_command((None, "1", False, "decode"))
#   # print("FINISHED")

import argparse
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
    ]
    process_requests(engine, test_prompts)


if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using the LLMEngine class directly')
    parser = EngineArgs.add_cli_args(parser)
    main()