from typing import Generator

from french_tarot.agents.card_phase_observation_encoder import CardPhaseObservationEncoder
from french_tarot.play import play_episode
from french_tarot.reagent.card_phase import CardPhaseStateActionEncoder
from french_tarot.reagent.data import ReAgentDataRow


def play_episodes() -> Generator[ReAgentDataRow, None, None]:
    encoder = CardPhaseStateActionEncoder(CardPhaseObservationEncoder())
    while True:
        yield play_episode(encoder)
        encoder.episode_done()
