import torch.nn as nn
import torch
import torch.nn.functional as F

class Siamese_CNN(torch.nn.Module):
    def __init__(self, vocab_size, device, word_matrix=None, embedding_dim=300, kernel_num=100, window_size=3):
        super(Siamese_CNN, self).__init__()
        self.device         = device
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        channel_in          = 1
        channel_out         = kernel_num
        self.conv           = nn.Conv2d(channel_in, channel_out, (window_size, embedding_dim))

        if word_matrix is not None:
            word_matrix = torch.tensor(word_matrix).to(self.device)
            self.word_embedding.weight.data.copy_(word_matrix)
            self.word_embedding.weight.requires_grad = False
        
        self.linear1 = nn.Linear(3, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(p=0.1)
        self.dist    = nn.PairwiseDistance(2)
        
    def forward(self, batched_data) :
        text1 = torch.tensor(batched_data[1]).to(self.device)
        text2 = torch.tensor(batched_data[2]).to(self.device)
        
        # Representations
        text1_word_embedding = self.word_embedding(text1)
        text1_word_embedding = text1_word_embedding.unsqueeze(1)    
        text1_word_embedding = F.tanh(self.conv(text1_word_embedding))
        text1_word_embedding = text1_word_embedding.squeeze(3)
        text1_word_embedding = F.max_pool1d(text1_word_embedding, text1_word_embedding.size(2)).squeeze(2)

        text2_word_embedding = self.word_embedding(text2)
        text2_word_embedding = text2_word_embedding.unsqueeze(1)    
        text2_word_embedding = F.tanh(self.conv(text2_word_embedding))
        text2_word_embedding = text2_word_embedding.squeeze(3)
        text2_word_embedding = F.max_pool1d(text2_word_embedding, text2_word_embedding.size(2)).squeeze(2)
        
        # Interactions
        cosine_sim = F.cosine_similarity(text1_word_embedding, text2_word_embedding).view(-1, 1)
        dot_value  = torch.bmm(
                            text1_word_embedding.view(text1_word_embedding.size()[0], 1, text1_word_embedding.size()[1]), 
                            text2_word_embedding.view(text1_word_embedding.size()[0], text1_word_embedding.size()[1], 1)
                            ).view(text1_word_embedding.size()[0], 1)
        dist_value = self.dist(text1_word_embedding, text2_word_embedding).view(text1_word_embedding.size()[0], 1)


        # Dense layers
        result = torch.cat((cosine_sim, dot_value, dist_value), dim=1)
        
        result = self.linear1(result)
        result = self.dropout(result)
        result = F.relu(result)

        result = self.linear2(result)
        result = self.dropout(result)
        result = F.relu(result)

        result = self.linear3(result)
        return result
    
    