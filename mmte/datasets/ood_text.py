import json
import os
import random
from typing import Optional, Sequence

import yaml
from torch.utils.data import DataLoader

from mmte import ImageTxtSample, TxtSample, _OutputType
from mmte.datasets.base import BaseDataset, collate_fn
from mmte.methods.base import BaseMethod
from mmte.utils.registry import registry


@registry.register_dataset()
class OODText(BaseDataset):
    dataset_ids: Sequence[str] = ["dt-text", "dt-related-image", "dt-unrelated-image-color", "dt-unrelated-image-nature", "dt-unrelated-image-noise"]
    dataset_config: Optional[str] = "mmte/configs/datasets/ood-text.yaml"
    categories: Sequence[str] = ['augment', 'bible_p0.6', 'bible_p0', 'romantic_p0.6', 'romantic_p0', 'shake_p0.6', 'shake_p0', 'shake_w', 'tweet_p0.6', 'tweet_p0']
    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)
        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        self.generate_image = kwargs['generate_image']
        
        self.related_image_dir = self.config.get('related_image_dir', '')
        self.unrelated_image_dir = self.config.get('unrelated_image_dir', '')
        self.annotation_file = self.config.get('annotation_file', '')
        assert os.path.exists(self.related_image_dir) and os.path.exists(self.unrelated_image_dir) and os.path.exists(self.annotation_file)

        self.prompt = "Please label the sentiment of the following text as positive or negative. The sentence is: {} The answer should be exactly 'positive' or 'negative'."
            
        with open(self.annotation_file, 'r') as f:
            test_data_dict = json.load(f)

        if self.dataset_id in ["dt-unrelated-image-color", "dt-unrelated-image-nature", "dt-unrelated-image-noise"]:
            from mmte.datasets import UnrelatedImageDataset
            unrelated_id = self.dataset_id.split('dt-')[1]
            unrelated_dataset = UnrelatedImageDataset(dataset_id=unrelated_id)
        if self.dataset_id == "dt-related-image" and self.generate_image:
            from mmte.methods import RelatedGeneratedImage
            related_generator = RelatedGeneratedImage(method_id="related-image-generated", img_dir=self.related_image_dir)

        dataset = []
        for category in self.categories:
            for tid, each_text in enumerate(test_data_dict[category]):
                sentence = each_text['sentence']
                label = each_text['label']
                text_id = each_text['id']
                text=self.prompt.format(sentence)

                if self.dataset_id == "dt-text":
                    dataset.append(TxtSample(text=text, target=label))
                elif self.dataset_id == "dt-related-image":
                    if self.generate_image:
                        base_sentence = test_data_dict['base'][tid]['sentence']
                        image_path = os.path.join(self.related_image_dir, f'{text_id}.png')
                        related_generator.generate_image(text=base_sentence, size=(500,500), output_path=image_path)
                        dataset.append(ImageTxtSample(image_path=image_path, text=text, target=label))
                    else:
                        image_path = os.path.join(self.related_image_dir, f'{text_id}.png')
                        dataset.append(ImageTxtSample(image_path=image_path, text=text, target=label))
                else:
                    unrelated_sample: ImageTxtSample = random.sample(unrelated_dataset.dataset, k=1)[0]
                    dataset.append(ImageTxtSample(image_path=unrelated_sample.image_path, text=text, target=label))
            
        self.dataset = dataset

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
    

if __name__ == '__main__':
    dataset = OODText(dataset_id="dt-text")
    dataloader = DataLoader(dataset=dataset, batch_size=2, collate_fn=collate_fn)
    for data in dataloader:
        print(data)
        