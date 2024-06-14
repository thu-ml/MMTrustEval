
from typing import Optional, Sequence, Dict
from mmte.datasets.base import BaseDataset
from mmte.methods.base import BaseMethod
from mmte.utils.registry import registry
from mmte import ImageTxtSample, TxtSample, _OutputType
from mmte.datasets import UnrelatedImageDataset 
from pathlib import Path
import csv
import random
import yaml
import os


@registry.register_dataset()
class CrossModalJailbreakDataset(BaseDataset):
    # use cm as the abbreviation of cross-modal, and pos means positive, neg means negative
    dataset_ids: Sequence[str] = ["cm-jailbreak-text", "cm-jailbreak-unrelated", 
                                  "cm-jailbreak-pos", "cm-jailbreak-neg"]
    dataset_config: Optional[str] = "/mmte/configs/datasets/cm-jailbreak.yaml"

    def __init__(self, dataset_id: str, method_hook: Optional[BaseMethod] = None, **kwargs) -> None:
        super().__init__(dataset_id=dataset_id, method_hook=method_hook)

        with open(self.dataset_config) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        
        self.input_list = Path(self.config.get('prompt_list'))    
        self.pos_image_dir = Path(self.config.get('pos_image_dir'))
        self.neg_image_dir = Path(self.config.get('neg_image_dir'))
        assert os.path.exists(self.input_list)

        # load toxicity prompts, load the json file
        with open(self.input_list, "r") as f:
            reader = csv.DictReader(f)
            print(reader.fieldnames)

            dataset = []
            for row in reader:
                if row["id"] == "id":
                    continue
                
                question = row["question"]
                jailbreak_prompt = row["jailbreak"] if "jailbreak" in row else None

                if self.dataset_id == 'cm-jailbreak-text':
                    dataset.append(TxtSample(text=jailbreak_prompt, target=question))

                elif self.dataset_id == 'cm-jailbreak-unrelated':
                    unrelated_ids = ['color', 'nature', 'noise']
                    for unrelated_id in unrelated_ids:
                        unrelated_id = 'unrelated-image-' + unrelated_id
                        unrelated_dataset = UnrelatedImageDataset(dataset_id=unrelated_id)
                        unrelated_samples: ImageTxtSample = random.sample(unrelated_dataset.dataset, k=3)

                        for unrelated_sample in unrelated_samples:
                            dataset.append(ImageTxtSample(image_path=unrelated_sample.image_path, 
                                                          text=jailbreak_prompt, target=question))
                
                elif self.dataset_id == 'cm-jailbreak-pos':
                    # get image list under self.pos_image_dir
                    pos_image_lst = os.listdir(self.pos_image_dir)
                    for image_path in pos_image_lst:
                        image_path = os.path.join(self.pos_image_dir, image_path)
                        dataset.append(ImageTxtSample(image_path=image_path, text=jailbreak_prompt, target=question))

                elif self.dataset_id == 'cm-jailbreak-neg':
                    # get image list under self.neg_image_dir
                    neg_image_lst = os.listdir(self.neg_image_dir)
                    for image_path in neg_image_lst:
                        image_path = os.path.join(self.neg_image_dir, image_path)
                        dataset.append(ImageTxtSample(image_path=image_path, text=jailbreak_prompt, target=question))

                else:
                    raise ValueError(f"Unknown dataset_id: {self.dataset_id}")
            
        print(f"{len(dataset)} data loaded")
        self.dataset = dataset

    def __getitem__(self, index: int) -> _OutputType:
        if self.method_hook:
            return self.method_hook.run(self.dataset[index])
        return self.dataset[index]
    
    def __len__(self) -> int:
        return len(self.dataset)
