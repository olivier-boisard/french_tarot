import gym


class FrenchTarotEnvironment(gym.Env):
    metadata = {"render.modes": ["human"]}

    def step(self, action):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def render(self, mode="human", close=False):
        raise NotImplementedError()
