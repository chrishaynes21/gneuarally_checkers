import numpy as np
from piece import Piece, Color


class Board:
    def __init__(self):
        self.draught = np.empty(shape=(8, 8), dtype=object)
        self.turn = Color.RED

        # Initialize all pieces
        for row_index in range(len(self.draught)):
            if row_index in [0, 2, 6]:  # Right aligned rows
                for col_index in [1, 3, 5, 7]:
                    self.draught[row_index, col_index] = Piece(Color.BLACK, (row_index, col_index)) if row_index != 6 \
                        else Piece(Color.RED, (row_index, col_index))
            elif row_index in [1, 5, 7]:  # Left aligned rows
                for col_index in [0, 2, 4, 6]:
                    self.draught[row_index, col_index] = Piece(Color.RED, (row_index, col_index)) if row_index != 1 \
                        else Piece(Color.BLACK, (row_index, col_index))

    def __repr__(self):
        # Formatting for a row
        format_string = '{:2} | {:2} | {:2} | {:2} | {:2} | {:2} | {:2} | {:2}\n'
        final_string = ''
        for row in self.draught:
            final_string += format_string.format(*[str(piece) if piece is not None else '-' for piece in row])
        return final_string

    def validMoves(self):
        valid_pieces = [piece for piece in self.draught.flat if piece is not None and piece.color == self.turn]
        valid_moves = []
        for piece in valid_pieces:
            piece_moves = []
            if piece.color == Color.BLACK:
                piece_moves.append(self.checkSpace(piece.position, 1, -1))
                piece_moves.append(self.checkSpace(piece.position, 1, 1))
            elif piece.color == Color.RED:
                piece_moves.append(self.checkSpace(piece.position, -1, -1))
                piece_moves.append(self.checkSpace(piece.position, -1, 1))
            valid_moves.extend([(piece.position, move) for move in piece_moves if move is not None])
        return valid_moves

    def checkSpace(self, position, row_mod, col_mod):
        row, col = position
        new_row = row + row_mod
        new_col = col + col_mod
        if 0 <= new_row <= 7 and 0 <= new_col <= 7:
            if self.draught[new_row, new_col] is None:
                return new_row, new_col
        return None

    def printState(self):
        print(self)

    # Only can be run on adjacent opponent pieces
    def canBeJumped(self, attacker, defender):
        if attacker[0] < defender[0]:  # Attacking from top to bottom
            if attacker[1] < defender[1]:  # Attacking from left to right
                return self.draught[attacker[0] + 2, attacker[1] + 2] is None
            else:  # Attacking from right to left
                return self.draught[attacker[0] + 2, attacker[1] - 2] is None
        else:  # Attacking from bottom to top
            if attacker[1] < defender[1]:  # Attacking from left to right
                return self.draught[attacker[0] - 2, attacker[1] + 2] is None
            else:  # Attacking from right to left
                return self.draught[attacker[0] - 2, attacker[1] - 2] is None


if __name__ == '__main__':
    board = Board()
    board.printState()
    print(board.validMoves())
