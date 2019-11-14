import os

from french_tarot.agents.trained_player_card import CardPhaseObservationEncoder
from french_tarot.play import play_round, play_rounds, create_batch
from french_tarot.reagent.card_phase import CardPhaseStateActionEncoder


def test_play_rounds():
    encoder = CardPhaseStateActionEncoder(CardPhaseObservationEncoder())
    round_1_output = play_round(encoder)
    encoder.episode_done()
    round_2_output = play_round(encoder)

    assert all(map(lambda row: row.mdp_id == 0, round_1_output))
    assert all(map(lambda row: row.mdp_id == 1, round_2_output))


def test_play_n_rounds():
    n_players = 4
    n_cards_played_per_player = 18
    n_rounds = 10
    expected_len = n_players * n_cards_played_per_player * n_rounds
    rounds_output = play_rounds(n_rounds)
    assert len(rounds_output) == expected_len


def test_create_batch(request):
    n_rounds = 10
    output_file_path = ".tmp.parquet"
    create_batch(n_rounds, output_file_path)
    request.addfinalizer(lambda: os.remove(output_file_path))

    assert os.path.isfile(output_file_path)
