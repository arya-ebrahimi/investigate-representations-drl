from gymnasium.envs.registration import register

register(
    id='MazEnv-v0',
    entry_point='core.envs:MazEnv',
)