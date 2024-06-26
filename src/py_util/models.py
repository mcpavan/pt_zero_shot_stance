import torch
import torch.nn as nn
import model_layers as ml

class BiCondLSTMModel(torch.nn.Module):
    '''
    Bidirectional Coniditional Encoding LSTM (Augenstein et al, 2016, EMNLP)
    Single layer bidirectional LSTM where initial states are from the topic encoding.
    Topic is also with a bidirectional LSTM. Prediction done with a single layer FFNN with
    tanh then softmax, to use cross-entropy loss.
    '''

    def __init__(self, hidden_dim, text_input_dim, topic_input_dim, num_layers=1, drop_prob=0, num_labels=3, use_cuda=False):
        super(BiCondLSTMModel, self).__init__()
        
        self.use_cuda = use_cuda
        self.num_labels = num_labels
        self.output_dim = 1 if self.num_labels == 2 else self.num_labels

        self.bilstm = ml.BiCondLSTMLayer(
            hidden_dim = hidden_dim,
            text_input_dim = text_input_dim,
            topic_input_dim = topic_input_dim,
            num_layers = num_layers,
            use_cuda = use_cuda
        )

        self.dropout = nn.Dropout(p=drop_prob) #dropout on last layer
        self.pred_layer = ml.PredictionLayer(
            input_dim = 2 * num_layers * hidden_dim,
            output_dim = self.output_dim,
            use_cuda=use_cuda
        )

    def forward(self, text_embeddings, topic_embeddings, text_length, topic_length):
        text_embeddings = text_embeddings.transpose(0, 1) # (T, B, E)
        topic_embeddings = topic_embeddings.transpose(0, 1) # (C, B, E)

        _, combo_fb_hm, _, _ = self.bilstm(text_embeddings, topic_embeddings, text_length, topic_length) # (dir*Hidden*N_layers, B)

        #dropout
        combo_fb_hm = self.dropout(combo_fb_hm) # (B, dir*Hidden*N_layers)
        y_pred = self.pred_layer(combo_fb_hm) # (B, 2)
        return y_pred

class BiLSTMJointAttentionModel(torch.nn.Module):
    '''
    Text    -> Embedding    -> Bidirectional LSTM   - A
    Topic   -> Embedding    -> Bidirectional LSTM   - A
    A = Multihead Attention Mechanism -> Dense -> Softmax
    '''

    def __init__(self, lstm_text_input_dim=768, lstm_topic_input_dim=768, lstm_hidden_dim=20, lstm_num_layers=1, lstm_drop_prob=0,
                 attention_density=None, attention_heads=4, attention_drop_prob=0, drop_prob=0, num_labels=3, use_cuda=False):
        super(BiLSTMJointAttentionModel, self).__init__()
        
        self.use_cuda = use_cuda
        self.num_labels = num_labels
        self.output_dim = 1 if self.num_labels == 2 else self.num_labels

        self.bilstm = ml.BiLSTMJointAttentionLayer(
            lstm_topic_input_dim=lstm_topic_input_dim,
            lstm_text_input_dim=lstm_text_input_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_drop_prob,
            attention_density=attention_density,
            attention_heads=attention_heads,
            attention_dropout=attention_drop_prob,
            use_cuda=use_cuda,
        )

        self.dropout = nn.Dropout(p=drop_prob) #dropout on last layer
        self.pred_layer = ml.PredictionLayer(
            input_dim = None,
            output_dim = self.output_dim,
            use_cuda=use_cuda
        )

    def forward(self, text_embeddings, topic_embeddings, text_length, topic_length):
        text_embeddings = text_embeddings.transpose(0, 1) # (T, B, E)
        topic_embeddings = topic_embeddings.transpose(0, 1) # (C, B, E)

        bilstm_return_dict = self.bilstm(text_embeddings, topic_embeddings, text_length, topic_length)

        #dropout
        attention_dropout = self.dropout(bilstm_return_dict["attention_output"]) # (B, text_len*attn_den)

        y_pred = self.pred_layer(attention_dropout) # (B, 2)

        return y_pred

class BiLSTMAttentionModel(torch.nn.Module):
    '''
    Text -> Embedding -> Bidirectional LSTM -> Multihead Self-Attention Mechanism -> Dense -> Softmax
    '''

    def __init__(self, lstm_text_input_dim=768, lstm_hidden_dim=20, lstm_num_layers=1, lstm_drop_prob=0,
                 attention_density=16, attention_heads=4, attention_drop_prob=0, drop_prob=0, num_labels=3, use_cuda=False):
        super(BiLSTMAttentionModel, self).__init__()
        
        self.use_cuda = use_cuda
        self.num_labels = num_labels
        self.output_dim = 1 if self.num_labels == 2 else self.num_labels

        self.bilstm = ml.BiLSTMAttentionLayer(
            lstm_text_input_dim=lstm_text_input_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_num_layers=lstm_num_layers,
            lstm_dropout=lstm_drop_prob,
            attention_density=attention_density,
            attention_heads=attention_heads,
            attention_dropout=attention_drop_prob,
            use_cuda=use_cuda,
        )

        self.dropout = nn.Dropout(p=drop_prob) #dropout on last layer
        self.pred_layer = ml.PredictionLayer(
            input_dim = None,
            output_dim = self.output_dim,
            use_cuda=use_cuda
        )

    def forward(self, text_embeddings, text_length):
        text_embeddings = text_embeddings.transpose(0, 1) # (T, B, E)
        
        bilstm_return_dict = self.bilstm(text_embeddings, text_length)

        #dropout
        attention_dropout = self.dropout(bilstm_return_dict["attention_output"]) # (B, text_len*attn_den)

        y_pred = self.pred_layer(attention_dropout) # (B, 2)

        return y_pred

class CrossNet(torch.nn.Module):
    '''
    Cross Net (Xu et al. 2018)
    Cross-Target Stance Classification with Self-Attention Networks
    BiCond + Aspect Attention Layer
    '''
    def __init__(self, hidden_dim, attn_dim, text_input_dim, topic_input_dim, num_layers=1, drop_prob=0, num_labels=3, use_cuda=False):
        super(CrossNet, self).__init__()

        self.use_cuda = use_cuda
        self.num_labels = num_labels
        self.output_dim = 1 if self.num_labels == 2 else self.num_labels
        self.text_input_dim = text_input_dim
        self.topic_input_dim = topic_input_dim
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        self.crossNet_layer = ml.CrossNetLayer(
            hidden_dim=self.hidden_dim,
            attn_dim=self.attn_dim,
            text_input_dim=self.text_input_dim,
            topic_input_dim=self.topic_input_dim,
            num_layers=num_layers,
            dropout_prob=drop_prob,
            use_cuda=self.use_cuda
        )

        self.dropout = nn.Dropout(p=drop_prob) #dropout on last layer
        self.pred_layer = ml.PredictionLayer(
            input_dim = 2 * self.hidden_dim,#2 * num_layers * self.hidden_dim,
            output_dim = self.output_dim,
            use_cuda=use_cuda
        )

    def forward(self, text_embeddings, topic_embeddings, text_length, topic_length):
        text_embeddings = text_embeddings.transpose(0, 1) # (T, B, E)
        topic_embeddings = topic_embeddings.transpose(0, 1) # (C, B, E)

        _, att_vec, _ = self.crossNet_layer(text_embeddings, topic_embeddings, text_length, topic_length)

        #dropout
        att_vec_drop = self.dropout(att_vec) # (B, H*N, dir * N_layers)

        y_pred = self.pred_layer(att_vec_drop) # (B, 2)

        return y_pred

class AAD(torch.nn.Module):
    def __init__(self, src_encoder, tgt_encoder, text_input_dim, discriminator_dim,
                 num_labels=3, drop_prob=0, use_cuda=False):
        super(AAD, self).__init__()

        self.text_input_dim = text_input_dim
        self.discriminator_dim = discriminator_dim
        self.num_labels = num_labels
        self.output_dim = 1 if self.num_labels == 2 else self.num_labels
        self.use_cuda = use_cuda

        self.src_encoder = src_encoder
        self.tgt_encoder = tgt_encoder

        self.classifier = ml.AADClassifier(
            input_dim=self.text_input_dim,
            output_dim=self.output_dim,
            use_cuda=self.use_cuda
        )

        self.discriminator = ml.AADDiscriminator(
            intermediate_size=self.discriminator_dim,
            use_cuda=self.use_cuda
        )

    def forward(self, text_embeddings, **kwargs):
        # text: (B, T, E)
        y_pred = self.classifier(text_embeddings)
        return y_pred
