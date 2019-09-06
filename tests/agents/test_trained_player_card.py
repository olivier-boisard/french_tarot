import numpy as np
import torch

from agents import trained_player_card
from environment import Card, FrenchTarotEnvironment


def create_observation():
    return {
        "plis": [],
        "current_player": 0,
        "played_cards": [],
        "hand": [Card.DIAMOND_1],
        "revealed_cards_in_dog": []
    }


def test_cuts_feature():
    observation = create_observation()
    observation["plis"] = [
        {"played_cards": [Card.CLOVER_1, Card.CLOVER_2, Card.CLOVER_3, Card.CLOVER_4], "starting_player": 0},
        {"played_cards": [Card.CLOVER_5, Card.CLOVER_6, Card.TRUMP_2, Card.CLOVER_7], "starting_player": 3},
        {"played_cards": [Card.HEART_1, Card.TRUMP_3, Card.HEART_3, Card.HEART_4], "starting_player": 2},
        {"played_cards": [Card.TRUMP_4, Card.TRUMP_5, Card.TRUMP_6, Card.TRUMP_7], "starting_player": 3},
    ]
    observation["current_player"] = 1
    observation["played_card"] = [Card.DIAMOND_1, Card.DIAMOND_2, Card.TRUMP_8]
    features = trained_player_card._extract_features(observation)
    assert len(features["cuts"] == FrenchTarotEnvironment().n_players)
    assert features["cuts"][1] == ["clover"]

    feature_vector = trained_player_card._encode_features(features)
    assert feature_vector.size(0) == 1
    assert feature_vector.size(1) == trained_player_card.FEATURE_VECTOR_SIZE

    expected_value = torch.zeros_like(feature_vector)
    expected_value[6] = 1
    expected_value[11] = 1
    assert feature_vector[:12] == expected_value[:12]


def test_pees_feature():
    raise NotImplementedError()


def test_taker_feature():
    observation = create_observation()
    features = trained_player_card._extract_features(observation)
    assert features["taker_position"] == 0
    feature_vector = trained_player_card._encode_features(features)
    assert feature_vector == torch.zeros_like(feature_vector)

    observation["current_player"] = 1
    expected_value = torch.zeros_like(feature_vector)
    expected_value[13] = 1
    assert feature_vector[12:15] == expected_value[12:15]


def test_n_trumps_still_in_game():
    observation = create_observation()
    observation["plis"] = [
        {"played_cards": [Card.CLOVER_1, Card.CLOVER_2, Card.CLOVER_3, Card.CLOVER_4], "starting_player": 0},
        {"played_cards": [Card.CLOVER_5, Card.CLOVER_6, Card.TRUMP_2, Card.CLOVER_7], "starting_player": 3},
        {"played_cards": [Card.HEART_1, Card.TRUMP_3, Card.HEART_3, Card.HEART_4], "starting_player": 2},
    ]
    observation["hand"] = [Card.TRUMP_1, Card.TRUMP_4, Card.CLOVER_10]
    observation["revealed_cards_in_dog"] = [Card.TRUMP_18, Card.TRUMP_19]
    features = trained_player_card._extract_features(observation)
    assert features["trumps_still_in_game"] == 15
    feature_vector = trained_player_card._encode_features(features)
    assert np.isclose(feature_vector[15], 15. / 21.)
