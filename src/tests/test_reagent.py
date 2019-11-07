from src.tests.conftest import setup_environment


def test_encoder():
    environment, observation = setup_environment()
    action = observation.player.hand[0]
    reward = environment.step(action)[2]

    encode_card_phase()
