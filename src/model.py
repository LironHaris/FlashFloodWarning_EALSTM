import torch
import torch.nn as nn
import config

class EALSTMCell(nn.Module):
    """
    Entity-Aware LSTM Cell.
    This is the mathematical core. Unlike a standard LSTM, it takes two inputs:
    1. Dynamic input (x): Rainfall, Temperature (Changes over time)
    2. Static input (a): Basin attributes (Constant over time)

    The Static input controls the Input Gate (i) and Forget Gate (f),
    effectively "modulating" the cell based on the basin type.
    """

    def __init__(self, input_dim_dyn, input_dim_stat, hidden_dim):
        super(EALSTMCell, self).__init__()

        self.input_dim_dyn = input_dim_dyn
        self.input_dim_stat = input_dim_stat
        self.hidden_dim = hidden_dim

        self.weight_i = nn.Linear(input_dim_stat, hidden_dim)

        self.weight_f = nn.Linear(input_dim_stat, hidden_dim)

        self.weight_g = nn.Linear(input_dim_dyn + hidden_dim, hidden_dim)

        self.weight_o = nn.Linear(input_dim_dyn + hidden_dim, hidden_dim)

        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_dynamic, x_static, h_prev, c_prev):
        """
        One step of the LSTM.
        x_dynamic: [batch_size, input_dim_dyn] (Rain today)
        x_static:  [batch_size, input_dim_stat] (Basin traits)
        h_prev:    Previous hidden state
        c_prev:    Previous cell state
        """

        i = self.sigmoid(self.weight_i(x_static))
        f = self.sigmoid(self.weight_f(x_static))

        combined = torch.cat((x_dynamic, h_prev), dim=1)

        g = self.activation(self.weight_g(combined))  # Candidate memory
        o = self.sigmoid(self.weight_o(combined))  # Output gate

        c_next = f * c_prev + i * g

        h_next = o * self.activation(c_next)

        return h_next, c_next


class EALSTM(nn.Module):
    """
    The Full Model Wrapper.
    Iterates over the time sequence using the Cell defined above.
    """

    def __init__(self, input_dim_dyn, input_dim_stat, hidden_dim=256, dropout_rate=0.4):
        super(EALSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm_cell = EALSTMCell(input_dim_dyn, input_dim_stat, hidden_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_dynamic, x_static):
        """
        x_dynamic: [batch_size, seq_length, dyn_features]
        x_static:  [batch_size, stat_features]
        """
        batch_size = x_dynamic.size(0)
        seq_length = x_dynamic.size(1)

        h = torch.zeros(batch_size, self.hidden_dim).to(x_dynamic.device)
        c = torch.zeros(batch_size, self.hidden_dim).to(x_dynamic.device)

        for t in range(seq_length):
            x_t = x_dynamic[:, t, :]
            h, c = self.lstm_cell(x_t, x_static, h, c)

        prediction = self.head(self.dropout(h))
        return prediction


if __name__ == "__main__":
    print("--- Initializing EA-LSTM Model ---")

    dyn_features = len(config.DYNAMIC_FEATURES) + 2

    stat_features = 213

    hidden_size = 256

    model = EALSTM(input_dim_dyn=dyn_features,
                   input_dim_stat=stat_features,
                   hidden_dim=hidden_size)

    print(model)

    batch_size = 16
    seq_len = 270

    dummy_dyn = torch.randn(batch_size, seq_len, dyn_features)
    dummy_stat = torch.randn(batch_size, stat_features)

    print(f"\nForward Pass Check:")
    print(f"Input Dynamic: {dummy_dyn.shape}")
    print(f"Input Static:  {dummy_stat.shape}")

    output = model(dummy_dyn, dummy_stat)

    print(f"Output Shape:  {output.shape}")

    if output.shape == (batch_size, 1):
        print("\nSUCCESS! Model output shape is correct [Batch, 1].")
    else:
        print("\nERROR! Output shape is wrong.")