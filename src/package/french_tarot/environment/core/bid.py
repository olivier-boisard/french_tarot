from enum import IntEnum


class Bid(IntEnum):
    PASS = 0
    PETITE = 1
    GARDE = 2
    GARDE_SANS = 3
    GARDE_CONTRE = 4
