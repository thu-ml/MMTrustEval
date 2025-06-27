from mmte.utils.registry import registry
from mmte.models import BaseChat
from pprint import pp
import traceback
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-ids', type=str, nargs='+', default=[])
parser.add_argument('--use-local', action='store_true')
args = parser.parse_args()


if args.use_local:
    model_ids = registry.list_chatmodels()
    pp(model_ids)
    import pdb; pdb.set_trace()
else:
    model_ids = args.model_ids

for model_id in model_ids:
    print("*" * 10 + model_id + "*" * 10)

    model: BaseChat = registry.get_chatmodel_class(model_id)(model_id)


    try:
        print(
            model.chat(
                [
                    {
                        "role": "user",
                        "content": {
                            "text": "Describe the image.",
                            "image_path": "example.jpg",
                        },
                    }
                ],
                do_sample=False,
                max_new_tokens=111,
            )
        )
    except Exception as e:
        traceback.print_exc(e)
        continue

    try:
        print(
            model.chat(
                [{"role": "user", "content": "Tell me a joke"}],
                do_sample=False,
                max_new_tokens=111,
            )
        )
    except Exception as e:
        traceback.print_exc(e)
        continue
    
    print("*" * 20 + "\n\n\n\n\n")

    del model
    torch.cuda.empty_cache()