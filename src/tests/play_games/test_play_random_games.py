from french_tarot.play_games import play_random_games


def test_play_random_games(mocker):
    mocker.patch("argparse.ArgumentParser.parse_args", lambda _: mocker.Mock(initial_seed=0))
    play_random_games.main(n_jobs=1, n_iterations=10, plot_scores=False)
