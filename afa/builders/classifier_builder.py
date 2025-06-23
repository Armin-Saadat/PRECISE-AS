import torch
from afa.classifiers import ProtoASSimFormer, ProtoASSimMax


def build(config, logger, data_train, data_val):
    classifier_builders = {
        'proto_as_sim_former': _build_proto_as_sim_former,
        'proto_as_sim_max': _build_proto_as_sim_max,
    }
    classifier = classifier_builders[config.classifier_name](config, logger, data_train, data_val)

    return classifier


def _build_proto_as_sim_former(config, logger, data_train, data_val):
    classifier = ProtoASSimFormer(logger=logger)
    assert config.classifier_path is not None, f'You have to pass the checkpoint path for ProtoASSimFormer'
    classifier.load_state_dict(torch.load(config.classifier_path, map_location=torch.device(config.device)))
    classifier.eval()
    classifier.to(config.device)

    return classifier


def _build_proto_as_sim_max(config, logger, data_train, data_val):
    classifier = ProtoASSimMax(logger=logger)
    assert config.classifier_path is not None, f'You have to pass the checkpoint path for ProtoASSimMax'
    classifier.protoAS_model.load_state_dict(
        torch.load(config.classifier_path, map_location=torch.device(config.device)))
    classifier.eval()
    classifier.to(config.device)

    return classifier
