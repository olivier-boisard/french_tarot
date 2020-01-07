from french_tarot.agents.encoding import encode_cards
from french_tarot.environment.subenvironments.card.card_phase_observation import CardPhaseObservation


class CardPhaseObservationEncoder:
    @staticmethod
    def encode(observation: CardPhaseObservation):
        return encode_cards(observation.player.hand)
