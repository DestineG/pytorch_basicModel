# chapter5.3-GRU/model.py

import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gates_rz = nn.Linear(input_size + hidden_size, hidden_size * 2)
        self.gates_n = nn.Linear(input_size + hidden_size, hidden_size)

        if output_size is not None:
            self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input_hidden = torch.cat((input, hidden), dim=1)

        r, z = self.gates_rz(input_hidden).chunk(2, dim=1)
        r_gate = torch.sigmoid(r)
        z_gate = torch.sigmoid(z)

        input_hidden_n = torch.cat((input, r_gate * hidden), dim=1)
        n = self.gates_n(input_hidden_n)
        n_gate = torch.tanh(n)

        hidden = (1 - z_gate) * n_gate + z_gate * hidden

        if self.output_size is not None:
            output = self.fc(hidden)
        else:
            output = None

        return hidden, output

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # gate layers
        layers = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            layer_output_size = output_size if layer == num_layers - 1 else None
            layers.append(GRUCell(layer_input_size, hidden_size, layer_output_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, input):
        batch_size, seq_length, _ = input.size()
        hidden_states = [torch.zeros(batch_size, self.hidden_size).to(input.device) for _ in range(self.num_layers)]
        outputs = []

        for t in range(seq_length):
            x = input[:, t, :]
            for layer in range(self.num_layers):
                hidden_states[layer], out = self.layers[layer](x, hidden_states[layer])
                x = hidden_states[layer]
            outputs.append(out.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs[:, -1, :]  # 只返回最后一个时间步的输出


if __name__ == "__main__":    # 简单测试GRU模型
    batch_size = 4
    seq_length = 10
    input_size = 3
    hidden_size = 5
    output_size = 1
    num_layers = 2

    model = GRU(input_size, hidden_size, output_size, num_layers)
    test_input = torch.randn(batch_size, seq_length, input_size)
    output = model(test_input)
    print("输出形状:", output.shape)  # 应该是 (batch_size, output_size)