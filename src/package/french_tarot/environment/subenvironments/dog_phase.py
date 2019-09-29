from typing import List, Tuple

import numpy as np
from attr import dataclass

from french_tarot.environment.core import Card, is_oudler, count_trumps_and_excuse, get_card_set_point, Observation, \
    PlayerData
from french_tarot.environment.subenvironments.core import SubEnvironment
from french_tarot.exceptions import FrenchTarotException


@dataclass
class DogPhaseObservation(Observation):
    dog_size: int


class DogPhaseEnvironment(SubEnvironment):

    def __init__(self, hand: List[Card], original_dog: List[Card]):
        self.hand = hand + original_dog
        self._n_cards_in_dog = len(original_dog)
        self.new_dog = []
        self.current_player_id = 0  # TODO factorize current player in upperclass

    def reset(self):
        self.new_dog = []
        return self.observation

    @property
    def game_is_done(self):
        return False

    @property
    def observation(self) -> DogPhaseObservation:
        current_player_data = PlayerData(self.current_player_id, self.hand)
        return DogPhaseObservation(current_player_data, self._n_cards_in_dog)

    def step(self, dog: List[Card]) -> Tuple[DogPhaseObservation, float, bool, None]:
        if type(dog) != list:
            raise FrenchTarotException("Wrong type for 'action'")
        if len(set(dog)) != len(dog):
            raise FrenchTarotException("Duplicated cards in dog")
        if np.any(["king" in card.value for card in dog]):
            raise FrenchTarotException("There should be no king in dog")
        if np.any([is_oudler(card) for card in dog]):
            raise FrenchTarotException("There should be no oudler in dog")
        if np.any([card not in self.hand for card in dog]):
            raise FrenchTarotException("Card in dog not in taking player's hand")
        if len(dog) != self._n_cards_in_dog:
            raise FrenchTarotException("Wrong number of cards in dog")

        n_trumps_in_dog = np.sum(["trump" in card.value for card in dog])
        if n_trumps_in_dog > 0:
            n_trumps_in_taking_player_hand = count_trumps_and_excuse(self.hand)
            n_kings_in_taking_player_hand = np.sum(["king" in card.value for card in self.hand])
            allowed_trumps_in_dog = self._n_cards_in_dog - (
                    len(self.hand) - n_trumps_in_taking_player_hand - n_kings_in_taking_player_hand)
            if n_trumps_in_dog != allowed_trumps_in_dog:
                raise FrenchTarotException("There should be no more trumps in dog than needed")

        index_to_keep_in_hand = [card not in dog for card in self.hand]
        self.hand = list(np.array(self.hand)[index_to_keep_in_hand])
        reward = get_card_set_point(dog)
        self.new_dog = dog
        done = True
        info = None
        return self.observation, reward, done, info
