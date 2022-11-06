from transformers.models.bert.modeling_bert import BertForPreTraining
from transformers.models.bert.modeling_bert import BertLMHeadModel
from transformers.models.bert.modeling_bert import BertForMaskedLM
from transformers.models.bert.modeling_bert import BertForNextSentencePrediction

from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
from transformers.models.bert.modeling_bert import BertLMPredictionHead
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from transformers.models.bert.modeling_bert import BertPreTrainingHeads

from transformers.models.albert.modeling_albert import AlbertForPreTraining
from transformers.models.albert.modeling_albert import AlbertMLMHead
from transformers.models.albert.modeling_albert import AlbertSOPHead
from transformers.models.albert.modeling_albert import AlbertModel

import gc
import os 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import numpy as np 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device 



class NewBertForPreTraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)

        self.bert = NewBertModel(config)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


class NewBertLMHeadModel(BertLMHeadModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if not config.is_decoder:
            logger.warning("If you want to use `BertLMHeadModel` as a standalone, add `is_decoder=True.`")

        self.bert = NewBertModel(config, add_pooling_layer=False)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)


class NewBertForMaskedLM(BertForMaskedLM):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = NewBertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        if self.config.pad_token_id is None:
            raise ValueError("The PAD token should be defined for generation")

        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}



class NewAlbertSOPHead(AlbertSOPHead):
    def __init__(self, config: NewConfig):
        super().__init__()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled_output: torch.Tensor) -> torch.Tensor:
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


class NewAlbertForPreTraining(AlbertPreTrainedModel):
    def __init__(self, config: NewConfig):
        super().__init__(config)

        self.albert = NewBertModel(config)
        self.predictions = NewBertForMaskedLM(config)
        self.sop_classifier = NewAlbertSOPHead(config)

        # Initialize weights and apply final processing
        self.post_init()

'''
optimizers = {
        'adam': torch.optim.Adam,  # default lr=0.001
    } 

opt.optimizer = optimizers[opt.optimizer]

_params = filter(lambda p: p.requires_grad, self.model.parameters())

optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

loss.backward()

optimizer.step()

'''

model = NewBertForMaskedLM_().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 100
loss_all = []
for epoch in range(num_epochs):
    train_loss = 0
    train_num = 0
    for step,(x,y) in enumerate(train_loader):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        z_hat = model.forward(x)
        loss = criterion(z_hat,y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() *len(y)
        train_num+=len(y)
    loss_all.append(train_loss/train_num)
    print(f"Epoch:{epoch+1} Loss:{loss_all[-1]:0.8f}")
    del x,y,loss,train_loss,train_num
    gc.collect()
    torch.cuda.empty_cache()


#可视化 
for i, label in enumerate(vocab):
    W, WT = model.parameters()
    x,y = float(W[i][0]), float(W[i][1])
    plt.scatter(x, y)
    plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
plt.show()
