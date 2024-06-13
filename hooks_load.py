import yaml
import importlib

def read_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def import_function(module_name, function_name):
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function

def get_hooks(config):
    hook_list = config.get('dynamic_hooks', None)
    if hook_list is None:
        return None

    hooks = {}
    for function_name, module_name in hook_list.items():
        if module_name and function_name:
            try:
                func = import_function(module_name, function_name)
                hooks[function_name] = func
            except (ImportError, AttributeError) as e:
                print(f"Error importing function {function_name} from module {module_name}: {e}")
        else:
            print("Function module and name must be specified in the config file.")
    
    return hooks

if __name__ == "__main__":
    config = read_config('hooks.yaml')
    hooks = get_hooks(config)
    if hooks:
        print(hooks['subtract'](10, 5))  
        print(hooks['add'](10, 5))       
