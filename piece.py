from enum import Enum


class Color(Enum):
    BLACK = 1
    RED = 2


class Piece:
    def __init__(self, color):
        self.color = color
        self.king = False

    def __repr__(self):
        return 'R' if self.color == Color.RED else 'B'

    def king_me(self):
        self.king = True
