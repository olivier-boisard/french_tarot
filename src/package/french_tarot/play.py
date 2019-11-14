import itertools

from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player_card import CardPhaseObservationEncoder
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from french_tarot.reagent.card_phase import CardPhaseStateActionEncoder


def play_round(encoder: CardPhaseStateActionEncoder(CardPhaseObservationEncoder())):
    player = RandomPlayer()
    environment = FrenchTarotEnvironment()
    observation = environment.reset()
    done = False

    reagent_entries = []
    while not done:
        action = player.get_action(observation)
        new_observation, reward, done, _ = environment.step(action)
        if isinstance(observation, CardPhaseObservation):
            reagent_entries.append(encoder.encode(observation, action, reward))
        observation = new_observation
    return reagent_entries


def play_rounds(n_rounds: int):
    encoder = CardPhaseStateActionEncoder(CardPhaseObservationEncoder())
    output = [play_round(encoder) for _ in range(n_rounds)]
    return list(itertools.chain(*output))
