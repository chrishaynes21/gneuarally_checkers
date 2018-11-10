from enum import Enum


class Color(Enum):
    BLACK = 1
    RED = 2


class Piece:
    def __init__(self, color):
        self.color = color
        self.king = False

    def __repr__(self):
        piece = 'R' if self.color == Color.RED else 'B'
        if self.king:
            piece += 'K'
        return piece

    def king_me(self):
        self.king = True
