from enum import Enum


class Color(Enum):
    BLACK = 1
    RED = 2


class Piece:
    def __init__(self, color, initial_position):
        self.color = color
        self.king = False
        self.position = initial_position

    def __repr__(self):
        piece = 'R' if self.color == Color.RED else 'B'
        if self.king:
            piece += 'K'
        return piece

    def king_me(self):
        self.king = True

    def to_number(self):
        number = 3 if self.king else 1
        if self.color == Color.BLACK:
            number *= -1
        return number