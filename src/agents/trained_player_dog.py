from torch import nn

from agents.common import Agent, card_set_encoder
from environment import Card


class DogPhaseAgent(Agent):

    def __init__(self, device="cuda"):
        super(DogPhaseAgent, self).__init__(DogPhaseAgent._create_dqn().to(device))

    def get_action(self, observation):
        xx = card_set_encoder(observation)
        xx = self._policy_net(xx)
        return DogPhaseAgent._select_cards(xx, len(observation["original_dog"]))

    @staticmethod
    def _card_is_ok_in_dog(card):
        return "trump" not in card.value and "king" not in card.value and "excuse" not in card.value

    @staticmethod
    def _select_cards(xx, n_cards):
        card_list_set = [card for card in list(Card) if DogPhaseAgent._card_is_ok_in_dog(card)]
        assert xx.size(0) == len(card_list_set)
        new_dog = []
        for _ in range(n_cards):
            i = xx.argmax()
            xx[i] = 0
            new_dog.append(card_list_set[i])
        return new_dog

    @staticmethod
    def _create_dqn():
        return nn.Sequential(nn.Linear(78, 52), nn.Sigmoid())
