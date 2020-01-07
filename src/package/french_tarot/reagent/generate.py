from typing import List

from french_tarot.agents.trained_player_card import CardPhaseObservationEncoder
from french_tarot.play import play_episode
from french_tarot.reagent.card_phase import CardPhaseStateActionEncoder
from french_tarot.reagent.data import ReAgentDataRow


def play_episodes(n_rounds: int) -> List[ReAgentDataRow]:
    encoder = CardPhaseStateActionEncoder(CardPhaseObservationEncoder())
    output = []
    for _ in range(n_rounds):
        output.extend(play_episode(encoder))
        encoder.episode_done()
    return output
