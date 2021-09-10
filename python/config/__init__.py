import yaml

cate_2_part_num_dict = {'Laptop': 8, 'Dishwasher': 11, 'Bowl': 4, 'Bed': 24, 'Clock': 20, 'Bottle': 10, 'Mug': 4, 'Refrigerator': 13, 'Door': 8, 'Vase': 11, 'Display': 8, 'Lamp': 43, 'TrashCan': 16, 'Keyboard': 3, 'Scissors': 5, 'Hat': 8, 'Knife': 10, 'Faucet': 11, 'Table': 53, 'Earphone': 15, 'StorageFurniture': 36, 'Chair': 58, 'Bag': 5, 'Microwave': 12}

# General config
def load_config(path, default_path=None):
    ''' Loads config file.
    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg

def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.
    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used
    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v