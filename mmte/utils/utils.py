import os
from mmte import lib_path
import requests

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


