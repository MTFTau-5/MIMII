import yaml


def yaml_parser(config_path = 'MIMII/config/traib.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)


    input_dim = config.get('input_dim')
    num_classes = config.get('num_classes')
    num_heads = config.get('num_heads')
    num_layers = config.get('num_layers')
    dim_feedforward = config.get('dim_feedforward')
    batch_size = config.get('batch_size')
    pkl_file_path = config.get('pkl_file_path')
    num_devices = config.get('num_devices')
    test_size = config.get('test_size')
    num_epochs = config.get('num_epochs')
    random_state = config.get('random_state')
    lr = config.get('lr')
    return (
        input_dim, num_classes, num_heads, num_layers, dim_feedforward,
        batch_size, pkl_file_path, num_devices, test_size, num_epochs, random_state, lr
    )
