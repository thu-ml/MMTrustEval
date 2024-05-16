from mmte.datasets import MockDataset, ConfAIde
from mmte.datasets.base import collate_fn
from mmte.methods import UnrelatedNoiseImage, UnrelatedNatureImage, UnrelatedColorImage
from torch.utils.data import DataLoader
if __name__ == '__main__':

    # method = UnrelatedNoiseImage(method_id='unrelated-image-noise', img_dir='tmp_dir', img_size=(30, 40), lazy_mode=True)
    # method = UnrelatedNatureImage(method_id='unrelated-image-nature', img_dir='tmp_dir', img_size=(300, 400), lazy_mode=True)
    method = UnrelatedColorImage(method_id='unrelated-image-color', img_dir='tmp_dir', img_size=(300, 400), lazy_mode=True)
    
    # dataset = MockDataset(datasize=10, dataset_id="mock_dataset", method_hook=method)
    dataset = ConfAIde(dataset_id='confaide-text', method_hook=method)
    dataloader = DataLoader(dataset=dataset, batch_size=1, collate_fn=collate_fn)

    for data in dataloader:
        print(data)