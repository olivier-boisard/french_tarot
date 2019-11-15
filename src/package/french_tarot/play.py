import pyarrow as pa
import pyarrow.parquet as pq

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
    output = []
    for _ in range(n_rounds):
        output.extend(play_round(encoder))
        encoder.episode_done()
    return output


def create_batch(n_rounds, output_file_path: str):
    output = play_rounds(n_rounds)
    df = CardPhaseStateActionEncoder.convert_reagent_datarow_list_to_pandas_dataframe(output)

    print("Save batch at", output_file_path)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file_path)
