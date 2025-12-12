# chapter5-3-LSTM/model.py

import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gates = nn.Linear(input_size + hidden_size, hidden_size * 4)
        nn.init.constant_(self.gates.bias[0 : self.hidden_size], 1.0)

        if output_size is not None:
            self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, memory):
        input_hidden = torch.cat((input, hidden), dim=1)

        f, i, c, o = self.gates(input_hidden).chunk(4, dim=1)
        f_gate = torch.sigmoid(f)
        i_gate = torch.sigmoid(i)
        c_gate = torch.tanh(c)
        o_gate = torch.sigmoid(o)

        memory = f_gate * memory + i_gate * c_gate

        hidden = o_gate * torch.tanh(memory)

        if self.output_size is not None:
            output = self.fc(hidden)
        else:
            output = None

        return hidden, memory, output

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # gate layers
        layers = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            layer_output_size = output_size if layer == num_layers - 1 else None
            layers.append(LSTMCell(layer_input_size, hidden_size, layer_output_size))
        self.layers = nn.ModuleList(layers)

    # 单词排队前向传播
    # 单词前向传播时会在每个cell会留下memory和hidden，供下个单词经过该cell时参考
    # 每个cell的x除了第一个之外都是上一个cell的hidden
    # 个人理解：之所以用fc(hidden)作为输出，是因为在lstm结构中暗含着hidden中含有下一个单词的信息(以hidden作为下一个cell的输入，而下一个cell的输入就是下一个单词)
    def forward(self, input):
        batch_size, seq_length, _ = input.size()
        hidden_states = [torch.zeros(batch_size, self.hidden_size, device=input.device) for _ in range(self.num_layers)]
        memory_states = [torch.zeros(batch_size, self.hidden_size, device=input.device) for _ in range(self.num_layers)]

        outputs = []
        for t in range(seq_length):
            x = input[:, t, :]
            for layer in range(self.num_layers):
                hidden_states[layer], memory_states[layer], output = self.layers[layer](x, hidden_states[layer], memory_states[layer])
                x = hidden_states[layer]
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        return outputs[:, -1, :]  # Return the last output


if __name__ == "__main__":
    # Test the LSTM model
    input_size = 10
    hidden_size = 20
    output_size = 1
    seq_length = 5
    batch_size = 3

    model = LSTM(input_size, hidden_size, output_size)
    test_input = torch.randn(batch_size, seq_length, input_size)
    test_output = model(test_input)
    print("Test output shape:", test_output.shape)  # Expected: (batch_size, output_size)