import mmte.models # ignore
from mmte.utils.registry import registry
import argparse
model_ids = registry.list_chatmodels()
print("models: ", model_ids)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-id')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    generation_kwargs = {}
    generation_kwargs['max_new_tokens'] = 20
    generation_kwargs['do_sample'] = False
    
    args = parse_args()
    model_id = args.model_id
    print(model_id)
    model_cls = registry.get_chatmodel_class(model_id)
    model = model_cls(model_id)


    message = [
        {
            "role": "user",
            "content": "Tell me a joke."
        }
    ]
    response = model.chat(messages=message, **generation_kwargs)
    print(response.content)