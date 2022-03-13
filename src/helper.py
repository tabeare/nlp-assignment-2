import numpy as np
from torch import nn

def pad_and_truncate(sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
    x = (np.ones(maxlen) * value).astype(dtype)
    if truncating == 'pre':
        trunc = sequence[-maxlen:]
    else:
        trunc = sequence[:maxlen]
    trunc = np.asarray(trunc, dtype=dtype)
    if padding == 'post':
        x[:len(trunc)] = trunc
    else:
        x[-len(trunc):] = trunc
    return x


#### UNUSED CLASSES
class BERT_SPC_PROTO(nn.Module):
    def __init__(self, bert, dropout, bert_dim, polarities_dim):
        super(BERT_SPC_PROTO, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(bert_dim, bert_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(bert_dim, polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, return_dict=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.dropout2(self.dense(pooled_output))
        logits = self.dense2(logits)
        return logits


class BERT_SPC(nn.Module):
    def __init__(self, bert, dropout, bert_dim, polarities_dim):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(bert_dim, polarities_dim)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids = inputs[0], inputs[1]
        _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids, return_dict=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits



