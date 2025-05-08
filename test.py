import vllm
from collections import deque
from threading import Thread

class dummy_worker:
  def __init__(self):
    self.q = deque()
    self.engine = vllm.LLMEngine(model="meta-llama/Meta-Llama-3-8B-Instruct")
    vllm.LLMEngine()

    def start_engine():
      while True:
        if not self.q.empty():
          command = self.q.popleft()
          self.process_command(command)
        
        outputs = self.engine.step()
        for output in outputs:
          if output.finished:
            for out in output.outputs:
              print(out.text)
    
    Thread(target=start_engine).start()
  
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
  worker.add_command(("world!", "1", False, "prefill"))
  worker.add_command((None, "1", False, "decode"))
