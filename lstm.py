import torch
import torch.nn as nn
class LSTMModel(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs, sequence_len, n_lstm_layers=1, n_deep_layers=10, device="cpu",
                 dropout=0.2):
        '''
       n_features: number of input features (1 for univariate forecasting)
       n_hidden: number of neurons in each hidden layer
       n_outputs: number of outputs to predict for each training example
       n_deep_layers: number of hidden dense layers after the lstm layer
       sequence_len: number of steps to look back at for prediction
       dropout: float (0 < dropout < 1) dropout ratio between dense layers
       '''
        super().__init__()

        self.n_lstm_layers = n_lstm_layers
        self.nhid = n_hidden
        self.device = device  # set option for device selection

        # LSTM Layer
        self.lstm = nn.LSTM(n_features,
                            n_hidden,
                            num_layers=n_lstm_layers,
                            batch_first=True)  # As we have transformed our data in this way

        # first dense after lstm
        self.fc1 = nn.Linear(n_hidden * sequence_len, n_hidden)
        # Dropout layer
        self.dropout = nn.Dropout(p=dropout)

        # Create fully connected layers (n_hidden x n_deep_layers)
        dnn_layers = []
        for i in range(n_deep_layers):
            # Last layer (n_hidden x n_outputs)
            if i == n_deep_layers - 1:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(nhid, n_outputs))
            # All other layers (n_hidden x n_hidden) with dropout option
            else:
                dnn_layers.append(nn.ReLU())
                dnn_layers.append(nn.Linear(nhid, nhid))
                if dropout:
                    dnn_layers.append(nn.Dropout(p=dropout))
        # compile DNN layers
        self.dnn = nn.Sequential(*dnn_layers)

    def forward(self, x):

        # Initialize hidden state
        hidden_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)
        cell_state = torch.zeros(self.n_lstm_layers, x.shape[0], self.nhid)

        # move hidden state to device

        hidden_state = hidden_state.to(self.device)
        cell_state = cell_state.to(self.device)

        self.hidden = (hidden_state, cell_state)

        # Forward Pass
        x, h = self.lstm(x, self.hidden)  # LSTM
        x = self.dropout(x.contiguous().view(x.shape[0], -1))  # Flatten lstm out
        x = self.fc1(x)  # First Dense
        return self.dnn(x)  # Pass forward through fully connected DNN.