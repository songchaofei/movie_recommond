import torch
import torch.nn as nn
from RNN_MODEL import RNN


input_size = 768
hidden_size = 128
output_categories = 2

input1 = torch.rand(1, input_size)
hidden1 = torch.rand(1, hidden_size)


rnn = RNN(input_size, hidden_size, output_categories)
output, hidden = rnn(input1, hidden1)

print(output)
print(output.shape)