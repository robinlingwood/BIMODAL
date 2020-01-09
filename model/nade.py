"""
Implementation of Neural Autoregressive Distribution Estimator (NADE)
"""

import numpy as np
import torch
import torch.nn as nn
from one_out_lstm import OneOutLSTM

torch.manual_seed(1)
np.random.seed(5)


class NADE():

    def __init__(self, molecule_size=7, encoding_dim=55, lr=.01, hidden_units=256, generation='random',
                 missing_token=np.zeros((55))):

        self._molecule_size = molecule_size
        self._input_dim = encoding_dim
        self._output_dim = encoding_dim
        self._layer = 2
        self._hidden_units = hidden_units
        self._generation = generation
        self._missing = missing_token

        # Learning rate
        self._lr = lr

        # Build new model
        self._lstm_fordir = OneOutLSTM(self._input_dim, self._hidden_units, self._layer)
        self._lstm_backdir = OneOutLSTM(self._input_dim, self._hidden_units, self._layer)

        # Check availability of GPUs
        self._gpu = torch.cuda.is_available()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self._lstm_fordir = self._lstm_fordir.cuda()
            self._lstm_backdir = self._lstm_backdir.cuda()

        # Adam optimizer
        self._optimizer = torch.optim.Adam(list(self._lstm_fordir.parameters()) + list(self._lstm_backdir.parameters()),
                                           lr=self._lr, betas=(0.9, 0.999))
        # Cross entropy loss
        self._loss = nn.CrossEntropyLoss(reduction='mean')

    def build(self, name=None):
        """Build new model or load model by name"""

        if (name is None):
            self._lstm_fordir = OneOutLSTM(self._input_dim, self._hidden_units, self._layer)
            self._lstm_backdir = OneOutLSTM(self._input_dim, self._hidden_units, self._layer)

        else:
            self._lstm_fordir = torch.load(name + '_fordir.dat', map_location=self._device)
            self._lstm_backdir = torch.load(name + '_backdir.dat', map_location=self._device)

        if torch.cuda.is_available():
            self._lstm_fordir = self._lstm_fordir.cuda()
            self._lstm_backdir = self._lstm_backdir.cuda()

        self._optimizer = torch.optim.Adam(list(self._lstm_fordir.parameters()) + list(self._lstm_backdir.parameters()),
                                           lr=self._lr, betas=(0.9, 0.999))

    def train(self, data, label, epochs=1, batch_size=1):
        '''Train the model
        :param  data:   data array (n_samples, molecule_length, encoding_length)
                label:  label array (n_samples, molecule_length)
                epochs: number of epochs for the training
        :return statistic:  array storing computed losses (epochs, batchs)
        '''

        # Number of samples
        n_samples = data.shape[0]

        # Change axes from (n_samples, molecule_size, encoding_dim) to (molecule_size, n_samples, encoding_dim)
        data = np.swapaxes(data, 0, 1)

        # Create tensor for labels
        label = torch.from_numpy(label).to(self._device)

        # Calculate number of batches per epoch
        if (n_samples % batch_size) is 0:
            n_iter = n_samples // batch_size
        else:
            n_iter = n_samples // batch_size + 1

        # To store losses
        statistic = np.zeros((epochs, n_iter))

        # Prepare model for training
        self._lstm_fordir.train()
        self._lstm_backdir.train()

        # Iteration over epochs
        for i in range(epochs):

            # Iteration over batches
            for n in range(n_iter):

                # Reset gradient for each epoch
                self._optimizer.zero_grad()

                # Compute indices used as batch
                batch_start = n * batch_size
                batch_end = min((n + 1) * batch_size, n_samples)

                # Initialize loss for molecule
                molecule_loss = 0

                # Compute data for this batch
                batch_data = torch.from_numpy(data[:, batch_start:batch_end, :].astype('float32')).to(self._device)

                # Different cases for training
                if self._generation == 'random':

                    # Initialize loss for molecule
                    tot_loss = torch.zeros(1).to(self._device)

                    # Reset model with correct batch size
                    self._lstm_fordir.new_sequence(batch_end - batch_start, self._device)
                    self._lstm_backdir.new_sequence(batch_end - batch_start, self._device)

                    # Output for each position
                    position_out = torch.zeros(self._molecule_size, batch_end - batch_start, self._input_dim).to(
                        self._device)

                    # Forward iteration over molecules (Token at position n-2 and n-1 not read since no prediction for next tokens)
                    for j in range(self._molecule_size - 2):
                        # Prepare input tensor with dimension (1,batch_size, molecule_size)
                        input = batch_data[j].view(1, batch_end - batch_start, -1)

                        # Probabilities for forward and backward token
                        position_out[j + 1] = torch.add(position_out[j + 1], self._lstm_fordir(input))

                    # Backward iteration over molecules (Token at position 0 and 1 not read since no prediction for next tokens)
                    for j in range(self._molecule_size - 1, 1, -1):
                        # Prepare input tensor with dimension (1,batch_size, molecule_size)
                        input = batch_data[j].view(1, batch_end - batch_start, -1)

                        # Probabilities for forward and backward token
                        position_out[j - 1] = torch.add(position_out[j - 1], self._lstm_backdir(input))

                        # Compute loss for token from 1 to n-2 (loss not computed for first (0) and last token (n-1))
                    for j in range(1, self._molecule_size - 1):
                        # Cross-entropy loss
                        loss = self._loss(position_out[j], label[batch_start:batch_end, j])

                        # Sum loss over molecule
                        molecule_loss += loss.item()

                        # Add loss tensor
                        tot_loss = torch.add(tot_loss, loss)

                    # Compute gradients
                    tot_loss.backward()

                    # Store statistics: loss per token (middle token not included)
                    statistic[i, n] = molecule_loss / (self._molecule_size - 2)

                    # Perform optimization step
                    self._optimizer.step()

                elif self._generation == 'fixed':
                    # Prepare missing data for this batch
                    missing_data = np.repeat(self._missing, batch_end - batch_start, axis=0)
                    missing_data = np.swapaxes(missing_data, 0, 1)
                    missing_data = torch.from_numpy(missing_data.astype('float32')).to(self._device)

                    # The losses for position p and position molecule_size-p-1 are computed within a single loop iteration
                    for p in range(1, int(np.ceil(self._molecule_size / 2))):

                        # Initialize new sequence
                        self._lstm_fordir.new_sequence(batch_end - batch_start, self._device)
                        self._lstm_backdir.new_sequence(batch_end - batch_start, self._device)

                        # Iteration forward direction
                        # Read tokens until position p
                        for j in range(p):
                            input = batch_data[j].view(1, batch_end - batch_start, -1)
                            out = self._lstm_fordir(input)
                        pred_1 = out

                        # Read token at position p, since this token is predicted before the token at position molecule_size-1-p
                        input = batch_data[p].view(1, batch_end - batch_start, -1)
                        self._lstm_fordir(input)

                        # Read missing value until position molecule_size-1-p
                        for j in range(p + 1, self._molecule_size - 1 - p):
                            out = self._lstm_fordir(missing_data)
                        pred_2 = out

                        # Iteration backward direction
                        # Read backwards until position molecule_size-1-p
                        for j in range(self._molecule_size - 1, self._molecule_size - p - 1, -1):
                            input = batch_data[j].view(1, batch_end - batch_start, -1)
                            out = self._lstm_backdir(input)
                        pred_2 = torch.add(pred_2, out)

                        # Read missing values backwards until position p
                        for j in range(self._molecule_size - p - 1, p, -1):
                            out = self._lstm_backdir(missing_data)
                        pred_1 = torch.add(pred_1, out)

                        # Cross-entropy loss for position p
                        loss_1 = self._loss(pred_1[0], label[batch_start:batch_end, p])
                        loss_1.backward(retain_graph=True)  # Accumulate gradients
                        molecule_loss += loss_1.item()

                        # Compute loss for position molecule_size-1-p if it is not equal to position p. They are equal in the case of an odd SMILES length for the middle token.
                        if p != self._molecule_size - 1 - p:
                            loss_2 = self._loss(pred_2[0], label[batch_start:batch_end, self._molecule_size - p - 1])
                            loss_2.backward()  # Accumulate gradients
                            molecule_loss += loss_2.item()
                            del loss_2, pred_2  # Delete to reduce memory usage

                        del loss_1, pred_1  # Delete to reduce memory usage

                    # Store statistics: loss per token (middle token not included)
                    statistic[i, n] = molecule_loss / (self._molecule_size - 2)

                    # Perform optimization step
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
        self._lstm_fordir.train()
        self._lstm_backdir.train()

        # Gradient is not compute to reduce memory requirements
        with torch.no_grad():
            # Compute tensor of labels
            label = torch.from_numpy(label).to(self._device)

            # Number of samples
            n_samples = data.shape[0]

            # Change axes from (n_samples, molecule_size, encoding_dim) to (molecule_size , n_samples, encoding_dim)
            data = np.swapaxes(data, 0, 1)

            # Initialize loss for complete validation set
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

                # Data used in this batch
                batch_data = torch.from_numpy(data[:, batch_start:batch_end, :].astype('float32')).to(self._device)

                # Output for each position
                position_out = torch.zeros(self._molecule_size, batch_end - batch_start, self._input_dim).to(
                    self._device)

                # Initialize loss for molecule
                molecule_loss = 0

                # Different cases for validation
                if self._generation == 'random':

                    # Reset model with correct batch size and device
                    self._lstm_fordir.new_sequence(batch_end - batch_start, self._device)
                    self._lstm_backdir.new_sequence(batch_end - batch_start, self._device)

                    # Forward iteration over molecules (Token at position n-2 and n-1 not read since no prediction for next tokens)
                    for j in range(self._molecule_size - 2):
                        # Prepare input tensor with dimension (1,batch_size, molecule_size)
                        input = batch_data[j].view(1, batch_end - batch_start, -1)

                        # Probabilities for forward and backward token
                        position_out[j + 1] = torch.add(position_out[j + 1], self._lstm_fordir(input))

                    # Backward iteration over molecules (Token at position 0 and 1 not read since no prediction for next tokens)
                    for j in range(self._molecule_size - 1, 1, -1):
                        # Prepare input tensor with dimension (1,batch_size, molecule_size)
                        input = batch_data[j].view(1, batch_end - batch_start, -1)

                        # Probabilities for forward and backward token
                        position_out[j - 1] = torch.add(position_out[j - 1], self._lstm_backdir(input))

                    # Compute loss for token from 1 ro n-2 (loss not computed for first (0) and last token (n-1))
                    for j in range(1, self._molecule_size - 1):
                        # Cross-entropy loss
                        loss = self._loss(position_out[j], label[batch_start:batch_end, j])

                        # Sum loss over molecule
                        molecule_loss += loss.item()

                    # Add loss per token to total loss (start token and end token not counted)
                    tot_loss += molecule_loss / (self._molecule_size - 2)

                elif self._generation == 'fixed':

                    # Prepare missing data for this batch
                    missing_data = np.repeat(self._missing, batch_end - batch_start, axis=0)
                    missing_data = np.swapaxes(missing_data, 0, 1)
                    missing_data = torch.from_numpy(missing_data.astype('float32')).to(self._device)

                    # The losses for position p and position molecule_size-p-1 are computed within a single loop iteration
                    for p in range(1, int(np.ceil(self._molecule_size / 2))):

                        # Reset model with correct batch size and device
                        self._lstm_fordir.new_sequence(batch_end - batch_start, self._device)
                        self._lstm_backdir.new_sequence(batch_end - batch_start, self._device)

                        # Iteration forward direction
                        # Read until position p
                        for j in range(p):
                            input = batch_data[j].view(1, batch_end - batch_start, -1)
                            out = self._lstm_fordir(input)
                        pred_1 = out

                        # Read token at position p, since this token is predicted before the token at position molecule_size-1-p
                        input = batch_data[p].view(1, batch_end - batch_start, -1)
                        self._lstm_fordir(input)

                        # Read missing value until position molecule_size-1-p
                        for j in range(p + 1, self._molecule_size - 1 - p):
                            out = self._lstm_fordir(missing_data)
                        pred_2 = out

                        # Iteration backward direction
                        # Read backwards until position molecule_size-1-p
                        for j in range(self._molecule_size - 1, self._molecule_size - p - 1, -1):
                            input = batch_data[j].view(1, batch_end - batch_start, -1)
                            out = self._lstm_backdir(input)
                        pred_2 = torch.add(pred_2, out)

                        # Read backwards until position p
                        for j in range(self._molecule_size - p - 1, p, -1):
                            out = self._lstm_backdir(missing_data)
                        pred_1 = torch.add(pred_1, out)

                        # Cross-entropy loss for position p
                        loss_1 = self._loss(pred_1[0], label[batch_start:batch_end, p])
                        molecule_loss += loss_1.item()

                        # Compute loss for position molecule_size-1-p if it is not equal to position p. They are equal in the case of an odd SMILES length for the middle token.
                        if p != self._molecule_size - 1 - p:
                            loss_2 = self._loss(pred_2[0], label[batch_start:batch_end, self._molecule_size - p - 1])
                            molecule_loss += loss_2.item()
                            del loss_2, pred_2

                        del loss_1, pred_1

                    # Add loss per token to total loss (start token and end token not counted)
                    tot_loss += molecule_loss / (self._molecule_size - 2)

        # Return loss per token
        return tot_loss / n_iter

    def sample(self, seq, T=1):
        '''Generate new molecule
        :param seq: starting sequence
        :param T:   sampling temperature
        :return     newly generated molecule (1, molecule_length, encoding_length)
        '''

        # Prepare model
        self._lstm_fordir.eval()
        self._lstm_backdir.eval()

        # Gradient is not compute to reduce memory requirements
        with torch.no_grad():
            # Output array with merged forward and backward directions

            # Change axes from (1, molecule_size, encoding_dim) to (molecule_size , 1, encoding_dim)
            seq = np.swapaxes(seq, 0, 1).astype('float32')

            # Create tensor for data and select correct device
            seq = torch.from_numpy(seq).to(self._device)

            # Construct specific order for the generation
            if self._generation == 'random':
                order = np.random.choice(np.arange(self._molecule_size - 2) + 1, self._molecule_size - 2, replace=False)
            elif self._generation == 'fixed':
                order = np.zeros(self._molecule_size - 2).astype(int)
                order[0::2] = np.arange(1, len(order[0::2]) + 1)
                order[1::2] = np.arange(self._molecule_size - 2, len(order[0::2]), -1)

            # Construct molecule in a predefined order
            for r in order:
                # Reset model with correct batch size and device
                self._lstm_fordir.new_sequence(1, self._device)
                self._lstm_backdir.new_sequence(1, self._device)

                # Forward iteration over molecule up to token r
                for j in range(r):
                    # Prepare input tensor with dimension (molecule_size, 1, encoding_dim)
                    input = seq[j].view(1, 1, -1)

                    # Probabilities for forward and backward token (Overwriting until r is reached)
                    output_for = self._lstm_fordir(input)

                # Backward iteration over molecule up to token r
                for j in range(self._molecule_size - 1, r, -1):
                    # Prepare input tensor with dimension (1,batch_size, molecule_size)
                    input = seq[j].view(1, 1, -1)

                    # Probabilities for forward and backward token (Overwriting until r is reached)
                    output_back = self._lstm_backdir(input)

                # Add output from forward and backward iterations
                out = torch.add(output_for, output_back)

                # Compute new token
                token = self.sample_token(np.squeeze(out.cpu().detach().numpy()), T)

                # Exchange token in sequence
                seq[r, 0, :] = torch.zeros(self._input_dim)
                seq[r, 0, token] = 1.0

        return np.swapaxes(seq.cpu().numpy(), 0, 1)

    def sample_token(self, out, T=1.0):
        ''' Sample token
        :param out: output values from model
        :param T:   sampling temperature
        :return:    index of predicted token
        '''
        # Explicit conversion to float64 avoiding truncation errors
        out = out.astype('float64')

        # Compute probabilities with specific temperature
        out_T = out / T
        p = np.exp(out_T) / np.sum(np.exp(out_T))

        # Generate new token at random
        char = np.random.multinomial(1, p, size=1)
        return np.argmax(char)

    def save(self, name='test_model'):
        torch.save(self._lstm_fordir, name + '_fordir.dat')
        torch.save(self._lstm_backdir, name + '_backdir.dat')
