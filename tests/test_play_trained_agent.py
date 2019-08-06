import play_trained_agent


def test_play_trained_agent(mocker):
    mocker.patch('play_trained_agent.dump_and_display_results')
    mocker.patch('play_trained_agent.DEVICE', "cpu")
    play_trained_agent._main(10)
