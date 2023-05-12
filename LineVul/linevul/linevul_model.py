import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MultiMarginLoss
from transformers import RobertaForSequenceClassification


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.window = 100
        self.hidden_size = config.hidden_size
        self.input = nn.Sequential(nn.Linear(self.hidden_size, 1000),
                                   nn.ReLU(),
                                   nn.Linear(1000, 500), nn.ReLU())
        self.num_layers = 2
        self.rnn = nn.LSTM(500, 500, num_layers=self.num_layers, batch_first=True,
                           dropout=0.2, bidirectional=True)
        self.output = nn.Sequential(nn.Linear(1000, 500),
                                    nn.ReLU(),
                                    nn.Linear(500, 2))

    def forward(self, features, **kwargs):
        self.window = features.size()[0]
        x = features[:, 0, :]
        x = self.input(x)
        x = x.unsqueeze(1).repeat(1, self.window, 1)
        x, _ = self.rnn(x)
        x = x[:, -1]
        return self.output(x)

    def init_hidden(self):
        if torch.cuda.is_available():
            return torch.zeros((2 * self.num_layers, self.window, self.hidden_size)).cuda()
        else:
            return torch.zeros((2 * self.num_layers, self.window, self.hidden_size))


class LinearSVM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]
        return self.linear(x)


class Model(RobertaForSequenceClassification):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        # self.classifier = Net(config)
        self.classifier = LinearSVM(config)
        # self.classifier = RobertaClassificationHead(config)
        self.args = args

    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = MultiMarginLoss()
                # loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = MultiMarginLoss()
                # loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                return prob