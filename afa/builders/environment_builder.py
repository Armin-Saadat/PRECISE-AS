from afa.environments import ASTomSimEnv


def build(config, logger, classifier, data):
    env_builders = {
        'AS-Tom-Sim': _build_as_tom_sim,
    }
    if config.env_name not in env_builders.keys():
        logger.error('No environment named {}.'.format(config.env_name))
    env = env_builders[config.env_name](config, logger, classifier, data)
    logger.warning('Environment {} is created successfully.'.format(config.env_name))

    return env


def _build_as_tom_sim(config, logger, classifier, data):
    return ASTomSimEnv(data=data,
                       n_actions=config.n_actions,
                       classifier=classifier,
                       lammy=config.lammy,
                       max_step=config.max_step,
                       device=config.device,
                       logger=logger)
