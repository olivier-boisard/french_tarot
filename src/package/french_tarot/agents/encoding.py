from french_tarot.environment.core.core import CARDS


def encode_cards(cards):
    return [float(card in cards) for card in CARDS]
