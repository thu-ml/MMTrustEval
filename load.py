from mmte.utils.utils import DictAction, merge_config
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='config file path')
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
    args = parse_args()
    config = args.config

    with open(config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        if hasattr(args, "cfg_options") and args.cfg_options is not None:
            cfg = merge_config(cfg, args.cfg_options)
        print(config)
        print(cfg)
        import pdb; pdb.set_trace()