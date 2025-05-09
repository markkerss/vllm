import vllm
from collections import deque
from threading import Thread
import time
class dummy_worker:
  def __init__(self):
    self.q = deque()
    engine_args = vllm.EngineArgs(model="meta-llama/Llama-3.1-8B-Instruct")
    self.engine = vllm.LLMEngine.from_engine_args(engine_args)
    self.output_file = open("output2.txt", "w")
    
    def write_output(text):
        self.output_file.write(text)
        self.output_file.flush()

    def start_engine():
      while True:
        if len(self.q) > 0:
          command = self.q.popleft()
          self.process_command(command)
        
        outputs = self.engine.step()
        for output in outputs:
          if output.finished:
            for out in output.outputs:
              write_output("GENERATED: " + out.text + "\n")
        write_output("step\n")
    
    Thread(target=start_engine, daemon=True).start()
  
  def add_command(self, command):
    self.q.append(command)

  def process_command(self, command):
    prompt, request_id, is_first_chunk, action = command
    if action == "prefill":
      if is_first_chunk:
        self.engine.run_first_add_chunk(request_id, prompt)
      else:
        self.engine.run_add_chunk(request_id, prompt)
    elif action == "decode":
      self.engine.run_decode(request_id)
    # elif action == "export":
    #   self.engine.run_export(request_id)



if __name__ == "__main__":
  worker = dummy_worker()
  worker.add_command(("Hello, ", "1", True, "prefill"))
  time.sleep(5)
  worker.add_command(("my name is ", "1", False, "prefill"))
  # time.sleep(5)
  # worker.add_command((None, "1", False, "decode"))
