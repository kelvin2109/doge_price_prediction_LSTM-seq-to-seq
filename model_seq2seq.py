import numpy as np
import torch.nn as nn
import torch 
from tqdm import trange
import random

class LSTM_encoder(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):

        super(LSTM_encoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size,hidden_size,num_layers = num_layers,batch_first=True)

    
    def forward(self,x):
        lstm_out, self.hidden = self.lstm(x.view(x.size(0),x.size(1),x.size(2)))
        return lstm_out,self.hidden

    def init_hidden(self,batch_size):
        #zeroed hidden-state and cell-state
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
    
    
    


class LSTM_decoder(nn.Module):
    def __init__(self,input_size,hidden_size, num_layers): 
     
        super(LSTM_decoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, encoder_hidden_states):
        lstm_out, self.hidden = self.lstm(x.unsqueeze(1),encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(1))
        
        return output,self.hidden

class lstm_seq2seq(nn.Module):
    #train lstm encoder-decoder
    def __init__(self,input_size,hidden_size,num_layers):
        
        super(lstm_seq2seq, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.encoder = LSTM_encoder(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)
        self.decoder = LSTM_decoder(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)
        
    def train(self, input_tensor, target_tensor, epochs, target_len, batch_size, training_prediction, teacher_forcing_ratio=0.5,learning_rate = 0.01, is_dynamic_tf = True):
        
        losses = np.full(epochs, np.nan)
        
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        criterion = nn.MSELoss()
        
        n_batches = int(input_tensor.size(0)/batch_size)
        
        #trange used to create a bar-percentage like download-bar                                  
        with trange(epochs) as total_epochs:
            for e in total_epochs:
                
                batch_loss = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0
                
                for i in range(n_batches):
                    #input_batch = [batch_size,seq_len,input_size]
                    input_batch = input_tensor[i*batch_size:batch_size+(batch_size*i), :, :]
                    target_batch = target_tensor[i*batch_size:batch_size+(batch_size*i), :, :]
                    
                    outputs = torch.zeros(batch_size, target_len, input_batch.shape[2])
                    
                    encoder_hidden = self.encoder.init_hidden(batch_size)
                    
                    optimizer.zero_grad()
                    
                    encoder_output, encoder_hidden = self.encoder(input_batch)
                    
                    decoder_input = input_batch[:,-1,:]
                    
                    decoder_hidden = encoder_hidden
                    
                    
                    if training_prediction == 'recursive':
                        
                        for t in range(target_len):
                        #decoder_output = (batch_size, sequence_length, proj_size)
                        #decoder_hidden = (num_layers, batch_size, proj_size)
                            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                            outputs[:,t,:] = decoder_output
                            decoder_input = decoder_output
                             
                    
                    if training_prediction == 'teacher_forcing':
                        
                        if random.random() < teacher_forcing_ratio:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[:,t,:] = decoder_output
                                decoder_input = target_batch[:, t, :]
                                 
                        # predict recursively 
                        else:
                            for t in range(target_len): 
                                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                                outputs[:,t,:] = decoder_output
                                decoder_input = decoder_output
                                 

                    
                    # compute loss
                    loss = criterion(outputs, target_batch)
                    batch_loss += loss.item()
                    
                    #backpropagation
                    loss.backward()
                    optimizer.step()
                    
                batch_loss /= n_batches
                losses[e] = batch_loss
                total_epochs.set_postfix(loss="{0:.3f}".format(batch_loss))

                #the objective of using dyanmic_tf is to 
                if is_dynamic_tf and teacher_forcing_ratio > 0:
                    teacher_forcing_ratio = teacher_forcing_ratio - 0.02
                
                
        return losses

    def predict(self, input_tensor, target_len):
        
        input_tensor = input_tensor.unsqueeze(0)
        encoder_output, encoder_hidden = self.encoder(input_tensor)

        outputs = torch.zeros(target_len, input_tensor.shape[2])

        #decode input
        decoder_input = input_tensor[:,-1,:]
        decoder_hidden = encoder_hidden

        for t in range(target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[t,:] = decoder_output.squeeze(0)
            decoder_input = decoder_output
        #the reason why we call detach before we transform the tensor back to numpy
        #is because tensor has a additional layer which is storing the computional graph.
        outputs_arr = outputs.detach().numpy()

        return outputs_arr