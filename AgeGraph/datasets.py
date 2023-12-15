import os
import os.path as osp
import shutil
from typing import Callable, List, Optional

import torch

from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip

class AgeDataset(InMemoryDataset):
    url = 'https://vanderbilt.box.com/shared/static'
    filename = 'lzzks4472czy9f9vc8aikp7pdbknmtfe.zip'

    def __init__(self, root: str, transform: Optional[Callable] = None, pre_transform: Optional[Callable] = None, pre_filter: Optional[Callable] = None):
        self.name = 'HCPAge'
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.num_nodes = 1000

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, self.name, 'raw')

    @property
    def raw_file_names(self) -> str:
        return 'data.pt'

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, self.name, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = f'{self.url}/{self.filename}'
        path = download_url(url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)
        os.rename(
            osp.join(self.raw_dir, self.name, 'processed', f'{self.name}.pt'),
            osp.join(self.raw_dir, 'data.pt'))
        shutil.rmtree(osp.join(self.raw_dir, self.name))

    def process(self):
        data, slices = torch.load(self.raw_paths[0])

        num_samples = slices['x'].size(0) - 1
        data_list : List[Data] = []

        for i in range(num_samples):
            x = data.x[slices['x'][i]:slices['x'][i + 1]],
            edge_index = data.edge_index[:, slices['edge_index'][i]:slices['edge_index'][i + 1]]
            sample = Data(x=x, edge_index=edge_index, y=data.y[i])

            if self.pre_filter is not None and not self.pre_filter(sample):
                continue

            if self.pre_transform is not None:
                sample = self.pre_transform(sample)

            data_list.append(sample)
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    