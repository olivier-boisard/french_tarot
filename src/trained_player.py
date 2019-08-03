import numpy as np

from environment import Card


def bid_phase_observation_encoder(observation):
    return _encode_card_list(observation["hand"])


def _encode_card_list(card_list):
    deck = list(Card)
    return np.array([card in card_list for card in deck])
