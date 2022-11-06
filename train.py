from source.model import NewBertModel
from source.model import NewBertForMaskedLM
from source.data import MultiSourceDataset
from source.config import NewConfig

from torch.utils.data import DataLoader

from transformers import BertTokenizer
from torch.optim import AdamW
import torch 
from copy import deepcopy

if __name__ == '__main__':
    
    data = MultiSourceDataset(
        {'args': ('wikitext', 'wikitext-103-v1',), 'kwargs': {'split': 'train'}}, 
        {'args': ('imdb',), 'kwargs': {'split': 'train'}},
        {'args': ('samsum',), 'kwargs': {'split': 'train'}},
        {'args': ('yelp_review_full',), 'kwargs': {'split': 'train'}},
        {'args': ('multi_news',), 'kwargs': {'split': 'train'}},
        {'args': ('ag_news',), 'kwargs': {'split': 'train'}},
    )

    loader = DataLoader(data, batch_size=1)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # config = BertConfig.from_pretrained('bert-base-uncased')
    model = NewBertForMaskedLM(NewConfig(num_sources=2))
    opt = AdamW(model.parameters(), lr=1e-3)
    # Note: tokenizer.mask_token_id = 103
    # criterion = MLMLoss(103)
    
    for idb, batch in enumerate(loader):
        
        opt.zero_grad()
        
        text, source_id = batch
        
        text_id = tokenizer(text, add_special_tokens=True,
                            max_length=512,
                            truncation=True,
                            padding='max_length',
                            return_tensors='pt')
        
        labels = deepcopy(text_id['input_ids'])
        labels[labels == 103] = -100
        
        model_output = model.forward(**text_id,
                                     source_ids=source_id,
                                     labels=labels,
                                     return_dict=True)
        
        loss = model_output['loss']
        
        loss.backward()
        opt.step()

        print(idb, loss.item())