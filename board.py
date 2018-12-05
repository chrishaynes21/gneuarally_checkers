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
        final_string = '     ' + format_string.format(*list(range(0,8)))
        row_count = 0
        for row in self.draught:
            final_string += ' {} | '.format(row_count)
            row_count += 1
            final_string += format_string.format(*[str(piece) if piece is not None else '-' for piece in row])
        return final_string

    def printState(self):
        print(self)

    def numberizeRow(self, row):
        numericRow =[]
        for man in row:
            if man is None:
                numericRow.append(0)
            else:
                numericRow.append(man.to_number())
        return tuple(numericRow)

    def draughtToVector(self):
        rows = [self.numberizeRow(row) for row in self.draught]
        vector = []
        for row in rows:
            vector.extend(row)
        return vector

    def stateMoveVectorForNN(self, move):
        vector = self.draughtToVector()
        vector.extend(move[0])
        vector.extend(move[1][0])
        return vector

    def stateMoveTuple(self, move):
        return str(self), move

    def setBoard(self, board):
        self.draught = board

    def makeMove(self, move, change_turn=True):
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

        # Change turn if told to
        if change_turn:
            self.change_turn()

    def change_turn(self):
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

    def isOver(self):
        origTurn = self.turn

        self.turn = Color.RED
        redMoves = self.validMoves()

        self.turn = Color.BLACK
        blackMoves = self.validMoves()

        self.turn = origTurn

        # TODO: handle tie game
        if len(redMoves) == 0:
            # RED can't move, BLACK wins
            return True, Color.BLACK
        if len(blackMoves) == 0:
            # BLACK can't move, RED wins
            return True, Color.RED

        return False, None

    def getUtility(self, type='nm', maximizePlayer=Color.RED):
        isOver, winner = self.isOver()

        if type == 'nm':
            if isOver:
                if winner == Color.RED:
                    return 2 if self.turn is Color.RED else -2
                if winner == Color.BLACK:
                    return 2 if self.turn is Color.BLACK else -2
                else:
                    return 0
        else: # type == 'mm'
            if isOver:
                if maximizePlayer == Color.RED:
                    return 2 if winner == Color.RED else -2
                else: # maximizePlayer == Color.BLACK
                    return 2 if winner == Color.BLACK else -2

        #
        # Game is not over, count number of red and black pieces.
        # When counting, kings count for 2 and regular pieces count
        # for one. Whichever color has a higher value gets a utility
        # of 1 or -1
        #

        redPieces = 0
        blackPieces = 0

        for i in range(8):
            for j in range(8):
                piece = self.draught[i][j]

                if piece != None:
                    if piece.color == Color.BLACK:
                        if piece.king:
                            blackPieces += 2
                        else:
                            blackPieces += 1
                    else: # piece == Color.RED
                        if piece.king:
                            redPieces += 2
                        else:
                            redPieces += 1

        if type == 'nm':
            if redPieces > blackPieces:
                return 1 if self.turn is Color.RED else -1
            elif blackPieces > redPieces:
                return 1 if self.turn is Color.BLACK else -1
            else:
                return 0
        else: # type == 'mm'
            if redPieces > blackPieces:
                return 1 if maximizePlayer == Color.RED else -1
            elif blackPieces > redPieces:
                return 1 if maximizePlayer == Color.BLACK else -1
            else:
                return 0

    # A helper function to check if a space can be moved to. If occupied, it checks if it can begin a jump sequence
    # It returns the valid move for moving in that direction, a list of tuples that represent positions along the way
    def checkSpace(self, position, row_mod, col_mod, king=False, recurse=False, positions=None):
        if positions is None:
            positions = []
        new_row = position[0] + row_mod
        new_col = position[1] + col_mod
        # Bounds check the new space (does not include bounds checks for a jump)
        if 0 <= new_row <= 7 and 0 <= new_col <= 7:
            if self.draught[new_row, new_col] is None:  # If the space is not occupied
                if not recurse:  # If recursion, do not allow for a space to be unoccupied, it must be a jump
                    return [(new_row, new_col)]
            elif position not in positions:
                # If a space can be jumped, search for another possible jump. The jumps MUST be made if a jump is started
                if self.draught[new_row, new_col].color != self.turn and self.canBeJumped(position, (new_row, new_col)):
                    search = []  # Holds moves from search, populated with results of searching a new move
                    jump_position = (new_row + row_mod, new_col + col_mod)
                    positions.extend([tuple(jump_position)])  # Initialize array with first legal jump
                    if not king:  # Since normal pieces can only attack forward, check left and right of the new space
                        search.append(self.checkSpace(jump_position, row_mod, col_mod, recurse=True, positions=positions))
                        search.append(self.checkSpace(jump_position, row_mod, -col_mod, recurse=True, positions=positions))
                    else:  # King's can attack backwards, but cannot jump a space twice, thus (-row, -col) is omitted
                        search.append(self.checkSpace(jump_position, row_mod, col_mod, True, True, positions))
                        search.append(self.checkSpace(jump_position, row_mod, -col_mod, True, True, positions))
                        search.append(self.checkSpace(jump_position, -row_mod, col_mod, True, True, positions))
                    # Filter out illegal moves and extend the jumps array with valid more valid jumps
                    search = [path for path in search if path is not None]
                    if len(search) > 0:  # Additional jumps exist, add the items to the list using .extend()
                        for move in search:
                            if move is not None and position not in positions:
                                positions.extend(move)
                        return positions
                    else:  # No more jumps are possible, so just return this one
                        return positions
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
    # board = Board()
    # board.printState()
    # while True:
    #     decision = input('Turn: {} Continue? Y/N: '.format(board.turn))
    #     if decision == 'N' or decision == 'n':
    #         break
    #     moves = board.validMoves()
    #     if len(moves) == 0:
    #         print('Game over, {} won!'.format('Black' if Color.RED == board.turn else 'Red'))
    #     for i in range(0, len(moves)):
    #         print('Index:{:3} Move: {}'.format(i, moves[i]))
    #     move_index = int(input('Move index: '))
    #     board.makeMove(moves[move_index])
    #     board.printState()

    board = Board()
    piece = Piece(Color.RED, [1, 1])
    piece2 = Piece(Color.BLACK, [3, 3])
    draught = np.empty(shape=(8, 8), dtype=object)
    draught[1, 1] = piece
    draught[3, 3] = piece2
    board.setBoard(draught)
    board.printState()
    assert board.isOver() == (False, None)

    draught[1, 1] = None
    board.printState()
    assert board.isOver() == (True, Color.BLACK)
    board.turn = Color.RED
    assert board.getUtility() == -1
    board.turn = Color.BLACK
    assert board.getUtility() == 1

    draught[1, 1] = piece
    draught[3, 3] = None
    board.printState()
    assert board.isOver() == (True, Color.RED)
    board.turn = Color.RED
    assert board.getUtility() == 1
    board.turn = Color.BLACK
    assert board.getUtility() == -1