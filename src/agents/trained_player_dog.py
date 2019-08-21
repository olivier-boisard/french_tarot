import numpy as np
from torch import nn

from agents.common import Agent, card_set_encoder
from environment import Card, GamePhase


def _card_is_ok_in_dog(card):
    return "trump" not in card.value and "king" not in card.value and "excuse" not in card.value


class DogPhaseAgent(Agent):
    CARDS_OK_IN_DOG = [card for card in list(Card) if _card_is_ok_in_dog(card)]

    def __init__(self, device="cuda"):
        super(DogPhaseAgent, self).__init__(DogPhaseAgent._create_dqn().to(device))

    def get_action(self, observation):
        if observation["game_phase"] != GamePhase.DOG:
            raise ValueError("Game is not in dog phase")

        xx = card_set_encoder(observation)
        xx = self._policy_net(xx)
        return DogPhaseAgent._select_cards(xx, np.concatenate((observation["hand"], observation["original_dog"])),
                                           len(observation["original_dog"]))

    @staticmethod
    def _select_cards(xx, hand, n_cards):
        xx = xx.clone().detach()
        assert xx.size(0) == len(DogPhaseAgent.CARDS_OK_IN_DOG)
        mask = [card not in hand for card in DogPhaseAgent.CARDS_OK_IN_DOG]
        xx[mask] = 0

        new_dog = []
        for _ in range(n_cards):
            i = xx.argmax()
            xx[i] = 0
            new_dog.append(DogPhaseAgent.CARDS_OK_IN_DOG[i])
        return new_dog

    @staticmethod
    def _create_dqn():
        return nn.Sequential(nn.Linear(78, 52), nn.Sigmoid())
