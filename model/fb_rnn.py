"""
Implementation of synchronous Forward Backward Models
"""

import numpy as np
import torch
import torch.nn as nn
from two_out_lstm_v2 import TwoOutLSTM_v2

torch.manual_seed(1)
np.random.seed(5)


class FBRNN():

    def __init__(self, molecule_size=7, encoding_dim=55, lr=.01, hidden_units=256):

        self._molecule_size = molecule_size
        self._input_dim = 2 * encoding_dim
        self._layer = 2
        self._hidden_units = hidden_units

        # Learning rate
        self._lr = lr

        # Build new model
        self._lstm = TwoOutLSTM_v2(self._input_dim, self._hidden_units, self._layer)

        # Check availability of GPUs
        self._gpu = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self._lstm = self._lstm.cuda()

        # Adam optimizer
        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=self._lr, betas=(0.9, 0.999))

        # Cross entropy loss
        self._loss = nn.CrossEntropyLoss(reduction='mean')

    def build(self, name=None):
        """Build new model or load model by name"""

        if (name is None):
            self._lstm = TwoOutLSTM_v2(self._input_dim, self._hidden_units, self._layer)

        else:
            self._lstm = torch.load(name + '.dat', map_location=self._device)

        if torch.cuda.is_available():
            self._lstm = self._lstm.cuda()

        self._optimizer = torch.optim.Adam(self._lstm.parameters(), lr=self._lr, betas=(0.9, 0.999))

    def print_model(self):
        '''Print name and shape of all tensors'''
        for name, p in self._lstm.state_dict().items():
            print(name)
            print(p.shape)

    def train(self, data, label, epochs=1, batch_size=1):
        '''Train the model
        :param  data:   data array (n_samples, molecule_length, encoding_length)
        :param label:   label array (n_samples, molecule_length)
        :param  epochs: number of epochs for the training
        :param batch_size:  batch size for training
        :return statistic:  array storing computed losses (epochs, batch)
        '''

        # Compute tensor of labels
        label = torch.from_numpy(label).to(self._device)

        # Number of samples
        n_samples = data.shape[0]

        # Calculate number of batches per epoch
        if (n_samples % batch_size) is 0:
            n_iter = n_samples // batch_size
        else:
            n_iter = n_samples // batch_size + 1

        # Store losses
        statistic = np.zeros((epochs, n_iter))

        # Prepare model
        self._lstm.train()

        # Iteration over epochs
        for i in range(epochs):

            # Iteration over batches
            for n in range(n_iter):

                # Compute indices used as batch
                batch_start = n * batch_size
                batch_end = min((n + 1) * batch_size, n_samples)

                # Prepare data with two tokens as input
                data_batch = self.prepare_data(data[batch_start:batch_end])

                # Change axes from (n_samples, molecule_size//2+1, 2*encoding_dim)
                # to (molecule_size//2+1, n_samples, 2*encoding_dim)
                data_batch = np.swapaxes(data_batch, 0, 1)
                data_batch = torch.from_numpy(data_batch).to(self._device)

                # Initialize loss for molecule
                molecule_loss = torch.zeros(1).to(self._device)

                # Reset model with correct batch size
                self._lstm.new_sequence(batch_end - batch_start, self._device)

                # Iteration over molecules
                for j in range(self._molecule_size // 2):
                    # Prepare input tensor with dimension (1,batch_size, 2*molecule_size)
                    input = data_batch[j].view(1, batch_end - batch_start, -1)

                    # Probabilities for forward and backward token
                    forward, back = self._lstm(input)

                    # Mean cross-entropy loss forward prediction
                    loss_forward = self._loss(forward.view(batch_end - batch_start, -1),
                                              label[batch_start:batch_end, self._molecule_size // 2 + 1 + j])

                    # Mean cross-entropy loss backward prediction
                    loss_back = self._loss(back.view(batch_end - batch_start, -1),
                                           label[batch_start:batch_end, self._molecule_size // 2 - 1 - j])

                    # Add losses from both sides
                    loss_tot = torch.add(loss_forward, loss_back)

                    # print('Total loss:', loss_tot)
                    # Add to molecule loss
                    molecule_loss = torch.add(molecule_loss, loss_tot)

                # Compute backpropagation
                self._optimizer.zero_grad()
                molecule_loss.backward(retain_graph=True)

                # Store statistics: loss per token (middle token not included)
                statistic[i, n] = molecule_loss.cpu().detach().numpy()[0] / (self._molecule_size - 1)

                # Perform optimization step and reset gradients
                self._optimizer.step()

        return statistic

    def validate(self, data, label, batch_size=128):
        ''' Validation of model and compute error
        :param data:    test data (n_samples, molecule_size, encoding_size)
        :param label:   label data (n_samples, molecule_size)
        :param batch_size:  batch size for validation
        :return:        mean loss over test data
        '''

        # Use train mode to get loss consistent with training
        self._lstm.train()

        # Gradient is not compute to reduce memory requirements
        with torch.no_grad():
            # Compute tensor of labels
            label = torch.from_numpy(label).to(self._device)

            # Number of samples
            n_samples = data.shape[0]

            # Initialize loss for molecule
            tot_loss = 0

            # Calculate number of batches per epoch
            if (n_samples % batch_size) is 0:
                n_iter = n_samples // batch_size
            else:
                n_iter = n_samples // batch_size + 1

            for n in range(n_iter):

                # Compute indices used as batch
                batch_start = n * batch_size
                batch_end = min((n + 1) * batch_size, n_samples)

                # Prepare data with two tokens as input
                data_batch = self.prepare_data(data[batch_start:batch_end])

                # Change axes from (n_samples, molecule_size//2+1, 2*encoding_dim)
                # to (molecule_size//2+1, n_samples, 2*encoding_dim)
                data_batch = np.swapaxes(data_batch, 0, 1)
                data_batch = torch.from_numpy(data_batch).to(self._device)

                # Initialize loss for molecule at correct device
                molecule_loss = torch.zeros(1).to(self._device)

                # Reset model with correct batch size and device
                self._lstm.new_sequence(batch_end - batch_start, self._device)

                for j in range(self._molecule_size // 2):
                    # Prepare input tensor with dimension (1,n_samples, 2*molecule_size)
                    input = data_batch[j].view(1, batch_end - batch_start, -1)

                    # Forward and backward output
                    forward, back = self._lstm(input)

                    # Mean cross-entropy loss forward prediction
                    loss_forward = self._loss(forward.view(batch_end - batch_start, -1),
                                              label[batch_start:batch_end, self._molecule_size // 2 + 1 + j])

                    # Mean Cross-entropy loss backward prediction
                    loss_back = self._loss(back.view(batch_end - batch_start, -1),
                                           label[batch_start:batch_end, self._molecule_size // 2 - 1 - j])

                    # Add losses from both sides
                    loss_tot = torch.add(loss_forward, loss_back)

                    # Add to molecule loss
                    molecule_loss = torch.add(molecule_loss, loss_tot)

                tot_loss += molecule_loss.cpu().detach().numpy()[0] / (self._molecule_size - 1)
        return tot_loss / n_iter

    def sample(self, middle_token='G', T=1):
        '''Generate new molecule
        :param middle_token:    starting token for the generation
        :param T:               sampling temperature
        :return molecule:       newly generated molecule (molecule_size, encoding_length)
        '''

        # Prepare model
        self._lstm.eval()

        # Gradient is not compute to reduce memory requirements
        with torch.no_grad():
            # Output array with merged forward and backward directions
            output = np.zeros((self._molecule_size // 2 + 1, self._input_dim))

            # Store molecule
            molecule = np.zeros((1, self._molecule_size, self._input_dim // 2))

            # Set middle token as first output
            output[0, :self._input_dim // 2] = middle_token[:]
            output[0, self._input_dim // 2:] = middle_token[:]

            # Set middle token for molecule
            molecule[0, self._molecule_size // 2] = middle_token[:]

            # Prepare input as tensor at correct device
            input = torch.from_numpy(np.array(output[0, :]).astype(np.float32)).view(1, 1, -1).to(self._device)

            # Prepare model
            self._lstm.new_sequence(device=self._device)

            # Sample from model
            for j in range(self._molecule_size // 2):
                # Compute prediction
                forward, back = self._lstm(input)

                # Conversion to numpy and creation of new token by sampling from the obtained probability distribution
                token_forward = self.sample_token(np.squeeze(forward.cpu().detach().numpy()), T)
                token_back = self.sample_token(np.squeeze(back.cpu().detach().numpy()), T)

                # Set selected tokens
                molecule[0, self._molecule_size // 2 + 1 + j, token_forward] = 1.0
                molecule[0, self._molecule_size // 2 - 1 - j, token_back] = 1.0

                # Prepare input of next step
                output[j + 1, token_forward] = 1.0
                output[j + 1, self._input_dim // 2 + token_back] = 1.0
                input = torch.from_numpy(output[j + 1, :].astype(np.float32)).view(1, 1, -1).to(self._device)

        return molecule

    def prepare_data(self, data):
        '''Reshape data to get two tokens as single input
        :params data:           data array (n_samples, molecule_length, encoding_length)
        :return cominde_input:  reshape data (n_samples, molecule_size//2 +1, 2*encoding_length)
        '''

        # Number of samples
        n_samples = data.shape[0]

        # Reshaped data array
        combined_input = np.zeros((n_samples, self._molecule_size // 2 + 1, self._input_dim)).astype(np.float32)

        for i in range(n_samples):
            # First Input is two times the token in the middle
            combined_input[i, 0, :self._input_dim // 2] = data[i, self._molecule_size // 2, :]
            combined_input[i, 0, self._input_dim // 2:] = data[i, self._molecule_size // 2, :]

            # Merge two tokens to a single input
            for j in range(self._molecule_size // 2):
                combined_input[i, j + 1, :self._input_dim // 2] = data[i, self._molecule_size // 2 + 1 + j, :]
                combined_input[i, j + 1, self._input_dim // 2:] = data[i, self._molecule_size // 2 - 1 - j, :]

        return combined_input

    def sample_token(self, out, T=1.0):
        ''' Sample token
        :param out: output values from model
        :param T:   sampling temperature
        :return:    index of predicted token
        '''

        # Explicit conversion to float64 avoiding truncation errors
        out = out.astype('float64')

        # Compute probabilities with specific temperature
        out_T = out / T - max(out / T)
        p = np.exp(out_T) / np.sum(np.exp(out_T))

        # Generate new token at random
        char = np.random.multinomial(1, p, size=1)
        return np.argmax(char)

    def save(self, name='test_model'):
        torch.save(self._lstm, name + '.dat')
