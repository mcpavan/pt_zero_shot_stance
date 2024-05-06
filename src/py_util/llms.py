from typing import Any
from llama_cpp import Llama

class LlamaCpp_Model:
    def __init__(self, params, num_labels=2):
        self.model = Llama(model_path=params["model_path"])
        self.model_type = "llama_cpp"
        self.num_labels = num_labels
        self.output_dim = 1
    
    def __call__(self, *args: Any, **kwds: Any):
        return self.forward(*args, **kwds)
    
    def forward(self, prompt, params):
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            suffix: A suffix to append to the generated text. If None, no suffix is appended.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature to use for sampling.
            top_p: The top-p value to use for sampling.
            logprobs: The number of logprobs to return. If None, no logprobs are returned.
            echo: Whether to echo the prompt.
            stop: A list of strings to stop generation when encountered.
            repeat_penalty: The penalty to apply to repeated tokens.
            top_k: The top-k value to use for sampling.
            stream: Whether to stream the results.
        """

        output = self.model(prompt, **params)
        return output["choices"][0]["text"]
