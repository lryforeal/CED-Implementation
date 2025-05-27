import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from parameter import *
if not train_mode:
    from test_parameter import *
# a pointer network layer for policy output
class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim):
        super(SingleHeadAttention, self).__init__()
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = embedding_dim
        self.key_dim = self.value_dim
        self.tanh_clipping = 10
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.input_dim, self.key_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k, mask=None):

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)

        k_flat = k.reshape(-1, n_dim)
        q_flat = q.reshape(-1, n_dim)

        shape_k = (n_batch, n_key, -1)
        shape_q = (n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)
        K = torch.matmul(k_flat, self.w_key).view(shape_k)

        U = self.norm_factor * torch.matmul(Q, K.transpose(1, 2))
        U = self.tanh_clipping * torch.tanh(U)

        if mask is not None:
            U = U.masked_fill(mask == 1, -1e8)
        attention = torch.log_softmax(U, dim=-1)  # n_batch*n_query*n_key

        return attention


# standard multi head attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.input_dim = embedding_dim
        self.embedding_dim = embedding_dim
        self.value_dim = self.embedding_dim // self.n_heads
        self.key_dim = self.value_dim
        self.norm_factor = 1 / math.sqrt(self.key_dim)

        self.w_query = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_key = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.key_dim))
        self.w_value = nn.Parameter(torch.Tensor(self.n_heads, self.input_dim, self.value_dim))
        self.w_out = nn.Parameter(torch.Tensor(self.n_heads, self.value_dim, self.embedding_dim))

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, key_padding_mask=None, attn_mask=None):
        if k is None:
            k = q
        if v is None:
            v = q

        n_batch, n_key, n_dim = k.size()
        n_query = q.size(1)
        n_value = v.size(1)

        k_flat = k.contiguous().view(-1, n_dim)
        v_flat = v.contiguous().view(-1, n_dim)
        q_flat = q.contiguous().view(-1, n_dim)
        shape_v = (self.n_heads, n_batch, n_value, -1)
        shape_k = (self.n_heads, n_batch, n_key, -1)
        shape_q = (self.n_heads, n_batch, n_query, -1)

        Q = torch.matmul(q_flat, self.w_query).view(shape_q)  # n_heads*batch_size*n_query*key_dim
        K = torch.matmul(k_flat, self.w_key).view(shape_k)  # n_heads*batch_size*targets_size*key_dim
        V = torch.matmul(v_flat, self.w_value).view(shape_v)  # n_heads*batch_size*targets_size*value_dim
        
        U = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))  # n_heads*batch_size*n_query*targets_size
        
        if attn_mask is not None:
            attn_mask = attn_mask.view(1, n_batch, n_query, n_key).expand_as(U)

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.repeat(1, n_query, 1)
            key_padding_mask = key_padding_mask.view(1, n_batch, n_query, n_key).expand_as(U)  # copy for n_heads times

        if attn_mask is not None and key_padding_mask is not None:
            mask = (attn_mask + key_padding_mask)
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        if mask is not None:
            U = U.masked_fill(mask > 0, -1e8)

        attention = torch.softmax(U, dim=-1)  # n_heads*batch_size*n_query*targets_size

        heads = torch.matmul(attention, V)  # n_heads*batch_size*n_query*value_dim

        # out = heads.permute(1, 2, 0, 3).reshape(n_batch, n_query, n_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).reshape(-1, self.n_heads * self.value_dim),
            # batch_size*n_query*n_heads*value_dim
            self.w_out.view(-1, self.embedding_dim)
            # n_heads*value_dim*embedding_dim
        ).view(-1, n_query, self.embedding_dim)

        return out, attention  # batch_size*n_query*embedding_dim


class Normalization(nn.Module):
    def __init__(self, embedding_dim):
        super(Normalization, self).__init__()
        self.normalizer = nn.LayerNorm(embedding_dim)

    def forward(self, input):
        return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(EncoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512), nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        h0 = src
        h = self.normalization1(src)
        h, _ = self.multiHeadAttention(q=h, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_head):
        super(DecoderLayer, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(embedding_dim, n_head)
        self.normalization1 = Normalization(embedding_dim)
        self.feedForward = nn.Sequential(nn.Linear(embedding_dim, 512),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(512, embedding_dim))
        self.normalization2 = Normalization(embedding_dim)

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        h0 = tgt
        tgt = self.normalization1(tgt)
        memory = self.normalization1(memory)
        h, w = self.multiHeadAttention(q=tgt, k=memory, v=memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        h = h + h0
        h1 = h
        h = self.normalization2(h)
        h = self.feedForward(h)
        h2 = h + h1
        return h2, w


class Encoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(EncoderLayer(embedding_dim, n_head) for i in range(n_layer))

    def forward(self, src, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            src = layer(src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_dim=128, n_head=8, n_layer=1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(embedding_dim, n_head) for i in range(n_layer)])

    def forward(self, tgt, memory, key_padding_mask=None, attn_mask=None):
        for layer in self.layers:
            tgt, w = layer(tgt, memory, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        return tgt, w


class PolicyNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(PolicyNet, self).__init__()
        self.initial_embedding = nn.Linear(input_dim, embedding_dim) # layer for non-end position
        self.current_embedding = nn.Linear(embedding_dim * 2, embedding_dim)

        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        self.pointer = SingleHeadAttention(embedding_dim)

    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        node_feature = self.initial_embedding(node_inputs)
        enhanced_node_feature = self.encoder(src=node_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)

        return enhanced_node_feature

    def output_policy(self, enhanced_node_feature, edge_inputs, enemy_inputs,current_index, edge_padding_mask, node_padding_mask, enemy_padding_mask, greedy=False):
        current_edge = edge_inputs.permute(0, 2, 1)
        enemy_positions = enemy_inputs.permute(0, 2, 1)
        embedding_dim = enhanced_node_feature.size()[2]
        
        neigboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim))   # [batch_size, n_moves, embedding_dim]        
        enemy_features = torch.gather(enhanced_node_feature, 1, enemy_positions.repeat(1, 1, embedding_dim))  # [batch_size, n_enemies, embedding_dim]
        all_action_features = torch.cat([neigboring_feature, enemy_features], dim=1)  # [batch_size, n_moves + n_enemies, embedding_dim]

        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim))

        if edge_padding_mask is not None and enemy_padding_mask is not None:
            current_mask = torch.cat((edge_padding_mask, enemy_padding_mask), dim=2)
        else:
            current_mask = None

        enhanced_current_node_feature, _ = self.decoder(current_node_feature, enhanced_node_feature, node_padding_mask)
        enhanced_current_node_feature = self.current_embedding(torch.cat((enhanced_current_node_feature, current_node_feature), dim=-1))

        logp = self.pointer(enhanced_current_node_feature, all_action_features, current_mask)
        logp = logp.squeeze(1)
        return logp

    def forward(self, node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask=None, edge_padding_mask=None, edge_mask=None, enemy_padding_mask=None, greedy=False):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        logp = self.output_policy(enhanced_node_feature, edge_inputs, enemy_inputs, current_index, edge_padding_mask, node_padding_mask, enemy_padding_mask)
        return logp


class QNet(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(QNet, self).__init__()
        self.initial_embedding = nn.Linear(input_dim, embedding_dim) # layer for non-end position
        self.action_embedding = nn.Linear(embedding_dim*3, (K_SIZE+N_ENEMY)*N_SELF)
        
        self.encoder = Encoder(embedding_dim=embedding_dim, n_head=8, n_layer=6)
        self.decoder = Decoder(embedding_dim=embedding_dim, n_head=8, n_layer=1)

        self.q_values_layer = nn.Linear(embedding_dim, 1)
        self.linear_layer = nn.Linear(in_features=((K_SIZE+N_ENEMY)*N_SELF)**2, out_features=(K_SIZE + N_ENEMY)*N_SELF)
        
    def encode_graph(self, node_inputs, node_padding_mask, edge_mask):
        embedding_feature = self.initial_embedding(node_inputs)
        embedding_feature = self.encoder(src=embedding_feature, key_padding_mask=node_padding_mask, attn_mask=edge_mask)

        return embedding_feature

    def output_q_values(self, enhanced_node_feature, edge_inputs, enemy_inputs, current_index, edge_padding_mask, node_padding_mask):
        current_edge = edge_inputs.permute(0, 2, 1)
        enemy_positions = enemy_inputs.permute(0, 2, 1)
        current_index = current_index.permute(0, 2, 1)
        batch_size = edge_inputs.size()[0]
        embedding_dim = enhanced_node_feature.size()[2] 
        
        neigboring_feature = torch.gather(enhanced_node_feature, 1, current_edge.repeat(1, 1, embedding_dim)) # [batch_size, n_self*k_size, embedding_dim]
        enemy_features = torch.gather(enhanced_node_feature, 1, enemy_positions.repeat(1, 1, embedding_dim))  # [batch_size, n_enemy*n_self, embedding_dim]
        current_node_feature = torch.gather(enhanced_node_feature, 1, current_index.repeat(1, 1, embedding_dim)) # [batch_size, n_self, embedding_dim]
        
        neigboring_feature_reshaped = neigboring_feature.reshape(batch_size, N_SELF, K_SIZE, embedding_dim)
        enemy_features_reshaped = enemy_features.reshape(batch_size, N_SELF, N_ENEMY, embedding_dim)
        all_features = []
        for i in range(N_SELF):
            robot_neighbors = neigboring_feature_reshaped[:, i]
            robot_enemies = enemy_features_reshaped[:, i]
            robot_features = torch.cat([robot_neighbors, robot_enemies], dim=1)
            all_features.append(robot_features)

        all_action_features = torch.cat(all_features, dim=1)
        enhanced_current_node_feature, attention_weights = self.decoder(current_node_feature, enhanced_node_feature, node_padding_mask)
    
        action_features = torch.cat((enhanced_current_node_feature.repeat_interleave(K_SIZE+N_ENEMY, dim=1), current_node_feature.repeat_interleave(K_SIZE+N_ENEMY, dim=1), all_action_features), dim=-1)
        
        q_features = self.action_embedding(action_features) # [batch_size, (k_size+n_enemy)*3, (k_size+n_enemy)*3]
        q_features = q_features.flatten(start_dim=1)  # [batch_size, (k_size+n_enemy)*3 * (k_size+n_enemy)*3]
        q_features = self.linear_layer(q_features)
        
        tensor = torch.split(q_features, K_SIZE + N_ENEMY, dim=1)
        tensor_0 = tensor[0].unsqueeze(2).unsqueeze(3)  # (batch_size, K_SIZE, 1, 1)
        tensor_1 = tensor[1].unsqueeze(1).unsqueeze(3)  # (batch_size, 1, K_SIZE, 1)
        tensor_2 = tensor[2].unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, K_SIZE)
        result = tensor_0 + tensor_1 + tensor_2

        q_values = result.reshape(batch_size, (K_SIZE + N_ENEMY)**N_SELF)        
        q_values = q_values.unsqueeze(2)

        return q_values, attention_weights

    def forward(self, node_inputs, edge_inputs, enemy_inputs, current_index, node_padding_mask=None, edge_padding_mask=None,
                edge_mask=None):
        enhanced_node_feature = self.encode_graph(node_inputs, node_padding_mask, edge_mask)
        q_values, attention_weights = self.output_q_values(enhanced_node_feature, edge_inputs, enemy_inputs, current_index, edge_padding_mask, node_padding_mask)
        return q_values, attention_weights
