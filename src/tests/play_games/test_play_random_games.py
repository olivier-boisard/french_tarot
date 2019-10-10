from french_tarot.play_games import play_random_games


def test_play_random_games(mocker):
    mocker.patch("argparse.ArgumentParser.parse_args", lambda _: mocker.Mock(initial_seed=0))
    mocker.patch('french_tarot.play_games.play_random_games._plot_scores')
    play_random_games._main(n_jobs=1, n_iterations=10)
