from afa.datasets import ASTomSimDataset


def build(config):
    data_builders = {
        'AS-Tom-Sim': _build_as_tom_sim,
    }

    return data_builders[config.dataset_name](config)


def _build_as_tom_sim(config):
    data_train = ASTomSimDataset(config.dataset_path, 'train')
    data_val = ASTomSimDataset(config.dataset_path, 'val')
    data_test = ASTomSimDataset(config.dataset_path, 'test')

    return data_train, data_val, data_test
