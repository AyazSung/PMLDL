import numpy as np
import torch
from transformers import BertForSequenceClassification
from transformers import BertTokenizer


class ClassificationModel:
    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # load saved model
        self.model_my = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.model_my.load_state_dict(torch.load('../../models/bert_model.pt'))

        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model_my.to(self.device)

    def predict_toxicity(self, text):
        encoded_dict = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=64,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids = encoded_dict['input_ids']
        attention_mask = encoded_dict['attention_mask']
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        outputs = self.model_my(input_ids, token_type_ids=None, attention_mask=attention_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        pred_flat = np.argmax(logits, axis=1).flatten()
        return pred_flat[0]
