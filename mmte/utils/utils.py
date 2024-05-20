from argparse import Action, ArgumentParser, Namespace
from typing import Any, Sequence, Union, Dict
from mmte import lib_path
import requests
import copy
import os

def get_abs_path(rel):
    return os.path.join(lib_path, rel)


# Function to download an image from a URL
def download_image(url, path):
    if os.path.exists(path):
        # print(f"{path} already downloaded.")
        return True
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the download was successful
        with open(path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {path}")
        return True
    except requests.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return False

def get_task_type(task_id):
    """
    Determines the task type based on the suffix of the task_id.

    Args:
    - task_id (str): The task identifier, expected to end with a suffix indicating the type.

    Returns:
    - int: The numerical code representing the task type.

    Raises:
    - ValueError: If the task_id suffix is not recognized.
    """
    # Mapping of task_id suffixes to task_type values
    task_mapping = {
        'text': 0,
        'images': 1,
        'nature': 2,
        'noise': 3,
        'color': 4
    }

    # Extract the suffix from task_id
    suffix = task_id.split('-')[-1]
    
    # Retrieve and return the task type using the mapping dictionary
    if suffix in task_mapping:
        return task_mapping[suffix]
    else:
        raise ValueError(f"Unknown task_id suffix: {suffix}")



class DictAction(Action):
    """
    argparse action to split an argument into KEY=VALUE form
    on the first = and append to a dictionary. List options can
    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
    """

    @staticmethod
    def _parse_int_float_bool(val: str) -> Union[int, float, bool, Any]:
        """parse int/float/bool value in the string."""
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val == 'None':
            return None
        return val

    @staticmethod
    def _parse_iterable(val: str) -> Union[list, tuple, Any]:
        """Parse iterable values in the string.

        All elements inside '()' or '[]' are treated as iterable values.

        Args:
            val (str): Value string.

        Returns:
            list | tuple | Any: The expanded list or tuple from the string,
            or single value if no iterable values are found.

        Examples:
            >>> DictAction._parse_iterable('1,2,3')
            [1, 2, 3]
            >>> DictAction._parse_iterable('[a, b, c]')
            ['a', 'b', 'c']
            >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
            [(1, 2, 3), ['a', 'b'], 'c']
        """

        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]

        if is_tuple:
            return tuple(values)

        return values

    def __call__(self,
                 parser: ArgumentParser,
                 namespace: Namespace,
                 values: Union[str, Sequence[Any], None],
                 option_string: str = None):
        """Parse Variables in string and add them into argparser.

        Args:
            parser (ArgumentParser): Argument parser.
            namespace (Namespace): Argument namespace.
            values (Union[str, Sequence[Any], None]): Argument string.
            option_string (list[str], optional): Option string.
                Defaults to None.
        """
        # Copied behavior from `argparse._ExtendAction`.
        options = copy.copy(getattr(namespace, self.dest, None) or {})
        if values is not None:
            for kv in values:
                key, val = kv.split('=', maxsplit=1)
                options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)


def merge_config(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    dict1 = copy.deepcopy(dict1)
    for keys, value in dict2.items():
        key_list = keys.split('.')

        cur_dict = dict1
        for key_name in key_list[:-1]:
            if not hasattr(cur_dict, key_name):
                cur_dict[key_name] = {}
            cur_dict = cur_dict[key_name]
        
        cur_dict[key_list[-1]] = value
    
    return dict1