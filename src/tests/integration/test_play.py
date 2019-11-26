from french_tarot.agents.trained_player_card import CardPhaseObservationEncoder
from french_tarot.play import play_episode, play_episodes
from french_tarot.reagent.card_phase import CardPhaseStateActionEncoder


def test_play_episode():
    encoder = CardPhaseStateActionEncoder(CardPhaseObservationEncoder())
    round_1_output = play_episode(encoder)
    encoder.episode_done()
    round_2_output = play_episode(encoder)

    assert len(set(map(lambda entry: entry.mdp_id, round_1_output))) == 4
    assert len(round_1_output) == 72
    assert len(set(map(lambda entry: entry.mdp_id, round_1_output + round_2_output))) == 8


def test_play_episodes():
    n_players = 4
    n_cards_played_per_player = 18
    n_rounds = 10
    expected_len = n_players * n_cards_played_per_player * n_rounds
    rounds_output = play_episodes(n_rounds)
    assert len(rounds_output) == expected_len
    assert len(set(map(lambda round_output: round_output.mdp_id, rounds_output))) == n_players * n_rounds
