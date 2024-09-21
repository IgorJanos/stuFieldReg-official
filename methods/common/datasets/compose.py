from torch.utils.data import Dataset


class ComposeDataset(Dataset):
    def __init__(
        self,
        datasets,
        weights=None
    ):
        self.inner_datasets = datasets

        if (weights is None):
            weights = [ 1 ] *len(datasets)
        self.weights = weights

        self.count = 0
        for d,w in zip(datasets, weights):
            self.count += len(d)*w


    def __len__(self):
        return self.count
    

    def __getitem__(self, idx):

        id = 0        
        while (id < len(self.inner_datasets)):
            cur_len = len(self.inner_datasets[id])
            cur_w = self.weights[id]
            
            if (idx < cur_len*cur_w):
                idx = idx % cur_len
                break
            
            idx -= cur_len*cur_w
            id += 1

        # Delegate
        idx = idx % len(self.inner_datasets[id])
        return self.inner_datasets[id][idx]
