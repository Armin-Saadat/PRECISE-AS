from afa.agents import ASTomDDQNAgent


def build(config, logger):
    agent_builders = {
        'AS-Tom-DDQN': _build_as_tom_ddqn,
    }
    if config.agent_name not in agent_builders.keys():
        logger.error('No agent named {}.'.format(config.agent_name))
    agent = agent_builders[config.agent_name](config, logger)
    logger.warning('Agent {} is created successfully.'.format(config.agent_name))
    if config.agent_checkpoint_dir:
        agent.load(config.agent_checkpoint_dir, config.load_mode)
        logger.warning(f'Agent is loaded successfully from checkpoint at {agent.checkpoint_epoch} epoch.')

    return agent


def _build_as_tom_ddqn(config, logger):
    return ASTomDDQNAgent(state_shape=config.state_shape,
                          n_actions=config.n_actions,
                          lr=config.lr,
                          milestones=config.milestones,
                          gamma=config.gamma,
                          eps_start=config.eps_start,
                          eps_final=config.eps_final,
                          eps_decay=config.eps_decay,
                          memory_size=config.memory_size,
                          memory_bs=config.memory_bs,
                          discount_factor=config.discount_factor,
                          TAU=config.TAU,
                          device=config.device,
                          logger=logger)
