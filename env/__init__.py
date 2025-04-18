from gymnasium.envs.registration import register

register(
    id = 'Snake_v0',
    entry_point ="env.snake_v0:SnakeEnv",
)