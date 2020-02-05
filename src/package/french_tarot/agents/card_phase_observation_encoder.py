import numpy as np

from french_tarot.agents.encoding import encode_cards
from french_tarot.environment.core.core import CARDS, check_card_is_allowed
from french_tarot.environment.subenvironments.card.card_phase_observation import CardPhaseObservation
from french_tarot.exceptions import FrenchTarotException


class CardPhaseObservationEncoder:
    @staticmethod
    def encode(observation: CardPhaseObservation):
        return encode_cards(observation.player.hand)


def retrieve_allowed_cards(observation: CardPhaseObservation):
    card_list = np.array(CARDS)
    allowed_card_index = []
    for card in card_list:
        try:
            check_card_is_allowed(card, observation.played_cards_in_round, observation.player.hand)
            allowed_card_index.append(True)
        except FrenchTarotException:
            allowed_card_index.append(False)
    assert 1 <= np.sum(allowed_card_index) <= len(observation.player.hand)
    return card_list[allowed_card_index]
