
from lib2to3.pgen2 import token
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset

from datasets import load_dataset
import numpy as np

"""
1, MLM utils, find in transformers Huggingface
2, Dataset: (x) -> (x, s)
"""

class MultiSourceDataset(Dataset):
    
    def __init__(self, *sources):
        
        super().__init__()
        
        self.mapper, self.datasets = {}, {}
        
        for idt, d in enumerate(sources):
            
            args = d['args']
            name = args[0]
            kwargs = d['kwargs']
            
            self.mapper.update({name: idt})
            self.datasets.update({name: load_dataset(*args, **kwargs)})

    def __getitem__(self, index):
        if index < self.__len__():
            # is_sop = int(np.random.uniform() > 0.5)
            name1 = np.random.choice([k for k in self.datasets.keys()])
            id1 = np.random.randint(len(self.datasets[name1]))
            return self.datasets[name1][id1]['text'], self.mapper[name1]
        else:
            raise IndexError("Index out of bound.")
    
    def __len__(self):
        return sum([len(d) for d in self.datasets.values()])
