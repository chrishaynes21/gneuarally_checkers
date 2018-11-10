import numpy as np
from piece import Piece, Color


class Board:
    def __init__(self):
        self.draught = np.empty(shape=(8, 8), dtype=object)

        # Initialize all pieces
        for row_index in range(len(self.draught)):
            if row_index in [0, 2, 6]:  # Right aligned rows
                for col_index in [1, 3, 5, 7]:
                    self.draught[row_index, col_index] = Piece(Color.BLACK) if row_index != 6 else Piece(Color.RED)
            elif row_index in [1, 5, 7]:  # Left aligned rows
                for col_index in [0, 2, 4, 6]:
                    self.draught[row_index, col_index] = Piece(Color.RED) if row_index != 1 else Piece(Color.BLACK)

    def __repr__(self):
        format_string = '{:2} | {:2} | {:2} | {:2} | {:2} | {:2} | {:2} | {:2}\n'
        final_string = ''
        for row in self.draught:
            final_string += format_string.format(*[str(piece) if piece is not None else '-' for piece in row])
        return final_string

    def printState(self):
        print(self)


if __name__ == '__main__':
    board = Board()
    board.printState()
