from french_tarot.play_games import play_random_games


def test_play_random_games(mocker):
    mocker.patch("argparse.ArgumentParser.parse_args", lambda _: mocker.Mock(initial_seed=0))
    mocker.patch('play_games.play_random_games.N_ITERATIONS', 10)
    mocker.patch('play_games.play_random_games._print_final_scores')
    mocker.patch('play_games.play_random_games._plot_scores')
    mocker.patch('torch.utils.tensorboard.SummaryWriter')
    play_random_games._main()
