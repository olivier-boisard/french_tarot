from typing import List

from french_tarot.agents.random_agent import RandomPlayer
from french_tarot.agents.trained_player_card import CardPhaseObservationEncoder
from french_tarot.environment.french_tarot import FrenchTarotEnvironment
from french_tarot.environment.subenvironments.card_phase import CardPhaseObservation
from french_tarot.reagent.card_phase import CardPhaseStateActionEncoder
from french_tarot.reagent.data import ReAgentDataRow


def play_episode(encoder: CardPhaseStateActionEncoder(CardPhaseObservationEncoder())):
    player = RandomPlayer()
    environment = FrenchTarotEnvironment()
    observation = environment.reset()
    done = False

    reagent_entries_per_player = []
    actions_in_round = []
    observations_in_round = []
    while not done:
        action = player.get_action(observation)
        new_observation, rewards, done, _ = environment.step(action)
        if isinstance(observation, CardPhaseObservation):
            actions_in_round.append(action)
            observations_in_round.append(observation)
            if rewards is not None:
                iterator = zip(observations_in_round, actions_in_round, rewards)
                for observation, action, reward in iterator:
                    entry = encoder.encode(observation.player.position_towards_taker, observation, action, reward)
                    reagent_entries_per_player.append(entry)
                actions_in_round = []
                observations_in_round = []
        observation = new_observation
    return reagent_entries_per_player


def play_episodes(n_rounds: int) -> List[ReAgentDataRow]:
    encoder = CardPhaseStateActionEncoder(CardPhaseObservationEncoder())
    output = []
    for _ in range(n_rounds):
        output.extend(play_episode(encoder))
        encoder.episode_done()
    return output
