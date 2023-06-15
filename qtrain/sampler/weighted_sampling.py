import random
import torch.utils.data

class WeightedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self.dataset = dataset
        self.mask_indices = []
        self.no_mask_indices = []
        for i in range(len(dataset)):
            if i in dataset.positive_scans:
                self.mask_indices.append(i)
            else:
                self.no_mask_indices.append(i)
        self.num_mask_samples = len(self.mask_indices)
        self.num_no_mask_samples = len(self.no_mask_indices)
        self.total_samples = self.num_mask_samples + self.num_no_mask_samples
        
    def __iter__(self):
        self.mask_weight = 1.0 / self.num_mask_samples if self.num_mask_samples > 0 else 0
        self.no_mask_weight = 1.0 / self.num_no_mask_samples if self.num_no_mask_samples > 0 else 0
        mask_samples = random.choices(self.mask_indices, weights=[self.mask_weight] * self.num_mask_samples, k=self.num_no_mask_samples)
        no_mask_samples = random.choices(self.no_mask_indices, weights=[self.no_mask_weight] * self.num_no_mask_samples, k=self.num_mask_samples)
        samples = mask_samples + no_mask_samples
        random.shuffle(samples)
        return iter(samples)
    
    def __len__(self):
        return self.total_samples