import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from shark.shark_runner import SharkInference, SharkBenchmark
import time

torch.manual_seed(0)
tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")

class MiniLMSequenceClassification(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/MiniLM-L12-H384-uncased",  # The pretrained model.
            num_labels=2,  # The number of output labels--2 for binary classification.
            output_attentions=False,  # Whether the model returns attentions weights.
            output_hidden_states=False,  # Whether the model returns all hidden-states.
            torchscript=True,
        )

    def forward(self, tokens):
        return self.model.forward(tokens)[0]


test_input = torch.randint(2, (1,128))

shark_module = SharkBenchmark(
    MiniLMSequenceClassification(), (test_input,), jit_trace=True
)

shark_module.benchmark_all((test_input,))

# model = MiniLMSequenceClassification()
# for i in range(10):
#     begin = time.time()
#     out = model.forward((test_input,))
#     end = time.time()
#     if i == 10 - 1:
#         break
# print(f"Torch benchmark:{10/(end-begin)} iter/second, Total Iterations:{10}")


