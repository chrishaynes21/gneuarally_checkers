import numpy as np

from piece import Piece, Color


class Board:
    def __init__(self):
        self.draught = np.empty(shape=(8, 8), dtype=object)
        self.turn = Color.RED

        # Initialize all pieces
        for row_index in range(len(self.draught)):
            if row_index in [0, 2, 6]:  # Left aligned rows
                for col_index in [1, 3, 5, 7]:
                    self.draught[row_index, col_index] = Piece(Color.BLACK, (row_index, col_index)) if row_index != 6 \
                        else Piece(Color.RED, (row_index, col_index))
            elif row_index in [1, 5, 7]:  # Right aligned rows
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

    def printState(self):
        print(self)

    def stateMoveTuple(self, move):
        return str(self), move

    def makeMove(self, move):
        move_len = len(move[1])
        piece = self.draught[move[0][0], move[0][1]]

        # Check if any jumps occur in the move, if so remove that piece
        for i in range(0, move_len):
            prev = move[1][i - 1] if i > 0 else move[0]
            if abs(move[1][i][0] - prev[0]) == 2:
                self.draught[(move[1][i][0] + prev[0]) // 2, (move[1][i][1] + prev[1]) // 2] = None

        # Move piece and make previous space None
        self.draught[move[1][move_len - 1][0], move[1][move_len - 1][1]] = piece
        self.draught[move[0][0], move[0][1]] = None
        piece.position = move[1][move_len - 1][0], move[1][move_len - 1][1]

        # Check if any piece needs a king
        if self.turn == Color.RED and piece.position[0] == 0:
            piece.king_me()
        elif self.turn == Color.BLACK and piece.position[0] == 7:
            piece.king_me()

        # Change turn
        self.turn = Color.RED if self.turn == Color.BLACK else Color.BLACK

    def validMoves(self):
        # Get all valid pieces
        valid_pieces = [piece for piece in self.draught.flat if piece is not None and piece.color == self.turn]
        valid_moves = []
        for piece in valid_pieces:
            piece_moves = []
            if piece.color == Color.BLACK and not piece.king:  # Black can move down a board, left, and right
                piece_moves.append(self.checkSpace(piece.position, 1, -1))
                piece_moves.append(self.checkSpace(piece.position, 1, 1))
            elif piece.color == Color.RED and not piece.king:  # Red can move up a board, left, and right
                piece_moves.append(self.checkSpace(piece.position, -1, -1))
                piece_moves.append(self.checkSpace(piece.position, -1, 1))
            else:  # A king can move in any direction
                piece_moves.append(self.checkSpace(piece.position, 1, -1, king=True))
                piece_moves.append(self.checkSpace(piece.position, 1, 1, king=True))
                piece_moves.append(self.checkSpace(piece.position, -1, -1, king=True))
                piece_moves.append(self.checkSpace(piece.position, -1, 1, king=True))
            # Filter out illegal moves
            valid_moves.extend([(piece.position, tuple(move)) for move in piece_moves if move is not None])
        return valid_moves

    # A helper function to check if a space can be moved to. If occupied, it checks if it can begin a jump sequence
    # It returns the valid move for moving in that direction, a list of tuples that represent positions along the way
    def checkSpace(self, position, row_mod, col_mod, king=False, recurse=False):
        new_row = position[0] + row_mod
        new_col = position[1] + col_mod
        # Bounds check the new space (does not include bounds checks for a jump)
        if 0 <= new_row <= 7 and 0 <= new_col <= 7:
            if self.draught[new_row, new_col] is None:  # If the space is not occupied
                if not recurse:  # If recursion, do not allow for a space to be unoccupied, it must be a jump
                    return [(new_row, new_col)]
            else:
                # If a space can be jumped, search for another possible jump. The jumps MUST be made if a jump is started
                if self.draught[new_row, new_col].color != self.turn and self.canBeJumped(position, (new_row, new_col)):
                    search = []  # Holds moves from search, populated with results of searching a new move
                    jump_position = (new_row + row_mod, new_col + col_mod)
                    jumps = [jump_position]  # Initialize array with first legal jump
                    if not king:  # Since normal pieces can only attack forward, check left and right of the new space
                        search.append(self.checkSpace(jump_position, row_mod, col_mod, recurse=True))
                        search.append(self.checkSpace(jump_position, row_mod, -col_mod, recurse=True))
                    else:  # King's can attack backwards, but cannot jump a space twice, thus (-row, -col) is omitted
                        search.append(self.checkSpace(jump_position, row_mod, col_mod, king=True, recurse=True))
                        search.append(self.checkSpace(jump_position, row_mod, -col_mod, king=True, recurse=True))
                        search.append(self.checkSpace(jump_position, -row_mod, col_mod, king=True, recurse=True))
                    # Filter out illegal moves and extend the jumps array with valid more valid jumps
                    search = [path for path in search if path is not None]
                    if len(search) > 0:  # Additional jumps exist, add the items to the list using .extend()
                        for move in search:
                            if move is not None:
                                jumps.extend(move)
                        return jumps
                    else:  # No more jumps are possible, so just return this one
                        return [jump_position]
        return None

    # Only can be run on adjacent opponent pieces
    def canBeJumped(self, attacker, defender):
        if attacker[0] < defender[0]:  # Attacking from top to bottom
            if defender[0] < 7:
                if attacker[1] < defender[1]:  # Attacking from left to right
                    return defender[1] < 7 and self.draught[attacker[0] + 2, attacker[1] + 2] is None
                else:  # Attacking from right to left
                    return defender[1] > 0 and self.draught[attacker[0] + 2, attacker[1] - 2] is None
            else:
                return False
        else:  # Attacking from bottom to top
            if defender[0] > 0:
                if attacker[1] < defender[1]:  # Attacking from left to right
                    return defender[1] < 7 and self.draught[attacker[0] - 2, attacker[1] + 2] is None
                else:  # Attacking from right to left
                    return defender[1] > 0 and self.draught[attacker[0] - 2, attacker[1] - 2] is None
            else:
                return False


if __name__ == '__main__':
    board = Board()
    board.printState()
    while True:
        decision = input('Turn: {} Continue? Y/N: '.format(board.turn))
        if decision == 'N' or decision == 'n':
            break
        moves = board.validMoves()
        if len(moves) == 0:
            print('Game over, {} won!'.format('Black' if Color.RED == board.turn else 'Red'))
        for i in range(0, len(moves)):
            print('Index:{:3} Move: {}'.format(i, moves[i]))
        move_index = int(input('Move index: '))
        board.makeMove(moves[move_index])
        board.printState()
