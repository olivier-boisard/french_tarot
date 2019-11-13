from french_tarot.agents.encoding import encode_cards
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation


class CardPhaseObservationEncoder:
    def encode(self, observation: CardPhaseObservation):
        return encode_cards(observation.player.hand)
