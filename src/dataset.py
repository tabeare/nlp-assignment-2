import sklearn 
import numpy as np 

from torch.utils.data import Dataset
from helper import pad_and_truncate

class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='UTF-8')
        lines = [line.strip().split("\t") for line in fin if line.strip()]
        fin.close()

        categories = [element[1] for element in lines]
        cat_encoder = sklearn.preprocessing.LabelEncoder()
        cat_encoder.fit(categories)
        le_cat = cat_encoder.transform(categories)

        all_data = []
        for i,line in enumerate(lines):

          # Get Polarity, category and aspect
          if line[0] == "positive":
            polarity = 2
          elif line[0] == "negative":
            polarity = 0
          elif line[0] == "neutral" :
            polarity = 1
          else :
            raise("Polarity problem")

          category = le_cat[i]
          aspect = line[2]

          # Get right and left text
          index_colon = line[3].index(':')
          start_num = int(line[3][:index_colon])
          end_num = int(line[3][index_colon+1:len(line[3])])
          text_left = line[4][:start_num].strip()
          text_right = line[4][end_num:].strip()

          text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
          context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
          left_indices = tokenizer.text_to_sequence(text_left)
          left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
          right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
          right_with_aspect_indices = tokenizer.text_to_sequence(aspect + " " + text_right, reverse=True)
          aspect_indices = tokenizer.text_to_sequence(aspect)
          left_len = np.sum(left_indices != 0)
          aspect_len = np.sum(aspect_indices != 0)
          aspect_boundary = np.asarray([left_len, left_len + aspect_len - 1], dtype=np.int64)

          text_len = np.sum(text_indices != 0)
          concat_bert_indices = tokenizer.text_to_sequence('[CLS] ' + text_left + " " + aspect + " " + text_right + ' [SEP] ' + aspect + " [SEP]")
          concat_segments_indices = [0] * (text_len + 2) + [1] * (aspect_len + 1)
          concat_segments_indices = pad_and_truncate(concat_segments_indices, tokenizer.max_seq_len)

          text_bert_indices = tokenizer.text_to_sequence("[CLS] " + text_left + " " + aspect + " " + text_right + " [SEP]")
          aspect_bert_indices = tokenizer.text_to_sequence("[CLS] " + aspect + " [SEP]")


          data = {
              'concat_bert_indices': concat_bert_indices,
              'concat_segments_indices': concat_segments_indices,
              'text_bert_indices': text_bert_indices,
              'aspect_bert_indices': aspect_bert_indices,
              'text_indices': text_indices,
              'context_indices': context_indices,
              'left_indices': left_indices,
              'left_with_aspect_indices': left_with_aspect_indices,
              'right_indices': right_indices,
              'right_with_aspect_indices': right_with_aspect_indices,
              'aspect_indices': aspect_indices,
              'aspect_boundary': aspect_boundary,
              'polarity': polarity,
              'category': category
          }

          all_data.append(data)
        self.data = all_data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
