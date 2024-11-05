import os
import yaml
import argparse
import warnings
from pprint import pprint
warnings.filterwarnings("ignore")
from mmte.tasks.base import BaseTask
from mmte.utils.registry import registry
from mmte.evaluators.metrics import _supported_metrics
from mmte.utils.utils import DictAction, merge_config

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='mmte/configs/task/robustness/r4-adversarial-untarget.yaml', help='config file path')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    '''
    # List all available modules:
    pprint("models: {}".format(registry.list_chatmodels()))
    pprint("datasets: {}".format(registry.list_datasets()))
    pprint("methods: {}".format(registry.list_methods()))
    pprint("evaluators: {}".format(registry.list_evaluators()))
    pprint("metrics: {}".format(list(_supported_metrics.keys())))
    '''

    args = parse_args()
    config = args.config

    with open(config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        if hasattr(args, "cfg_options") and args.cfg_options is not None:
            cfg = merge_config(cfg, args.cfg_options)
        
        model_id = cfg.get('model_id')
        dataset_id = cfg.get('dataset_id')
        log_file = cfg.get('log_file')
        method_cfg = cfg.get('method_cfg', {})
        dataset_cfg = cfg.get('dataset_cfg', {})
        generation_kwargs = cfg.get('generation_kwargs', {})
        evaluator_seq_cfgs = cfg.get('evaluator_seq_cfgs', [])

        if 'max_new_tokens' not in generation_kwargs.keys():
            generation_kwargs['max_new_tokens'] = 50
        if 'do_sample' not in generation_kwargs.keys(): 
            generation_kwargs['do_sample'] = False

        cfg['generation_kwargs'] = generation_kwargs
        cfg['config_path'] = config

        pprint(cfg, width=150)

        if os.path.exists(log_file):
            print("{} is already exists.".format(log_file))
            exit()
        runner = BaseTask(dataset_id=dataset_id, model_id=model_id, method_cfg=method_cfg, dataset_cfg=dataset_cfg, generation_kwargs=generation_kwargs, log_file=log_file, evaluator_seq_cfgs=evaluator_seq_cfgs)
        runner.pipeline()