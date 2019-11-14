import numpy as np
import pytest

from french_tarot.agents.trained_player_card import CardPhaseObservationEncoder


def test_encoder_encode(card_phase_observation):
    encoder = CardPhaseObservationEncoder()
    output = encoder.encode(card_phase_observation)
    assert len(output) == 78
    assert np.min(output) == 0
    assert np.max(output) == 1
    assert np.sum(output) == 18


@pytest.mark.skip
def test_cuts_feature():
    raise NotImplementedError()


@pytest.mark.skip
def test_pees_feature():
    raise NotImplementedError()


@pytest.mark.skip
def test_remaining_figures_per_colors():
    raise NotImplementedError()


@pytest.mark.skip
def test_excuse_context():
    raise NotImplementedError()


@pytest.mark.skip
def test_n_big_trumps_still_in_game():
    raise NotImplementedError()


@pytest.mark.skip
def test_petit_still_in_game():
    raise NotImplementedError()


@pytest.mark.skip
def test_taker_feature():
    raise NotImplementedError()


@pytest.mark.skip
def test_n_trumps_still_in_game():
    raise NotImplementedError()
