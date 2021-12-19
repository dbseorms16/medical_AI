import torch
import torch.nn as nn

def make_model(opt):
    return CnnLstm(opt)

class CnnLstm(nn.Module):
    def __init__(self, opt, input_dim=256, hidden_dim=100, num_layers=7, output_dim=2):
        super(CnnLstm, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        #self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm3 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm4 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.lstm5 = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim*5, output_dim)
        
        
    def forward(self, x):
        # Set initial hidden states (and cell states for LSTM)
        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda:0') 
        c1 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda:0') 

        h2 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda:0') 
        c2 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda:0') 

        h3 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda:0') 
        c3 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda:0') 

        h4 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda:0') 
        c4 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda:0') 

        h5 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda:0') 
        c5 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to('cuda:0') 
        # print("init value h0:",h0.size())
        # Forward propagate RNN
        #out, _ = self.rnn(x, h0)  
        # or:
        out1, _ = self.lstm1(x[:,0,:,:], (h1,c1))  
        out2, _ = self.lstm2(x[:,1,:,:], (h2,c2))  
        out3, _ = self.lstm3(x[:,2,:,:], (h3,c3))  
        out4, _ = self.lstm4(x[:,3,:,:], (h4,c4))  
        out5, _ = self.lstm5(x[:,4,:,:], (h5,c5))  
        #print("value ltsm exit:",out.size())
        
        # Decode the hidden state of the last time step
        out1 = out1[:, -1, :]
        out2 = out2[:, -1, :]
        out3 = out3[:, -1, :]
        out4 = out4[:, -1, :]
        out5 = out5[:, -1, :]

        out = torch.cat((out1,out2,out3,out4,out5), dim=1)

         
        out = self.fc(out)
        
        #print("out for linear:",out.shape)
        
        # out: (n, 7) parameters
        return out