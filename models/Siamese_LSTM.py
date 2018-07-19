import torch.nn as nn
import torch
import torch.nn.functional as F

class Siamese_LSTM(torch.nn.Module) :
    def __init__(self, vocab_size, device, word_matrix=None, embedding_dim=300, hidden_dim=300):
        super(Siamese_LSTM, self).__init__()
        self.device         = device
        self.hidden_dim     = hidden_dim
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)

        if word_matrix is not None:
            word_matrix = torch.tensor(word_matrix).to(self.device)
            self.word_embedding.weight.data.copy_(word_matrix)
            self.word_embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.linear1 = nn.Linear(3, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.dist    = nn.PairwiseDistance(2)
        
        
    def forward(self, batched_data):
        text1 = torch.tensor(batched_data[1]).to(self.device)
        text2 = torch.tensor(batched_data[2]).to(self.device)

        text1_word_embedding = self.word_embedding(text1)
        text2_word_embedding = self.word_embedding(text2)
        text1_seq_embedding = self.lstm_embedding(self.lstm, text1_word_embedding)
        text2_seq_embedding = self.lstm_embedding(self.lstm, text2_word_embedding)
        
        
        # Interactions
        cosine_sim = F.cosine_similarity(text1_seq_embedding, text2_seq_embedding).view(-1, 1)
        dot_value  = torch.bmm(
                            text1_seq_embedding.view(text1_seq_embedding.size()[0], 1, text1_seq_embedding.size()[1]), 
                            text2_seq_embedding.view(text1_seq_embedding.size()[0], text1_seq_embedding.size()[1], 1)
                            ).view(text1_seq_embedding.size()[0], 1)
        dist_value = self.dist(text1_seq_embedding, text2_seq_embedding).view(text1_seq_embedding.size()[0], 1)


        # Dense layers
        result = torch.cat((cosine_sim, dot_value, dist_value), dim=1)
        
        result = self.linear1(result)
        result = self.dropout(result)
        result = F.relu(result)

        result = self.linear2(result)
        result = self.dropout(result)
        result = F.relu(result)

        result = self.linear3(result)

        return merged
    
    def lstm_embedding(self, lstm, word_embedding):
        lstm_out,(lstm_h, lstm_c) = lstm(word_embedding)
        seq_embedding = torch.cat((lstm_h[0], lstm_h[1]), dim=1)
        return seq_embedding
