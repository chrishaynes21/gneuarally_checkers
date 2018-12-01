# function MiNlMAX-DECisiON(gam<?) returns an operator
# for each op in OPERATORSfgame] do
# VALUE[op] — MINIMAX-VALUE(APPLY(op, game), game)
# end
# return the op with the highest VALUE[op]

from piece import Color
from copy import copy, deepcopy

inf = float('infinity')

def minimax(board, depthLeft, isMax=True):

    isOver, _ = board.isOver()

    if isOver or depthLeft == 0:
        maximizePlayer = None

        if isMax:
            maximizePlayer = board.turn
        else:
            maximizePlayer = Color.RED if board.turn == Color.BLACK else Color.BLACK

        return board.getUtility(type='mm', maximizePlayer=maximizePlayer), None

    if isMax:
        bestValue = -inf
        bestMove = None

        for move in board.validMoves():
            # Make copy of board to make the move on so original board remains the same
            tempBoard = deepcopy(board)

            print("Move: ", move)
            print("Piece: ", board.draught[move[0][0], move[0][1]])

            # Apply a move to current state
            tempBoard.makeMove(move)

            #print('trying max\n',tempBoard)
            value, _ = minimax(tempBoard, depthLeft - 1, False)

            if value > bestValue:
                # Value for this move is better than moves tried so far from this state.
                bestValue = value
                #print("new best value max! Old Val: ", bestValue, "New Val: ", value)
                bestMove = move

                if bestValue == 1:
                    # victory found, no need to check other states
                    break;

        return bestValue, bestMove

    else:
        bestValue = inf
        bestMove = None

        for move in board.validMoves():
            # Make copy of board to make the move on so original board remains the same
            tempBoard = deepcopy(board)

            # Apply a move to current state
            tempBoard.makeMove(move)

            #print('trying min\n',tempBoard)
            value, _ = minimax(tempBoard, depthLeft - 1, True)

            if value < bestValue:
                # Value for this move is better than moves tried so far from this state.
                #print("new best value min! Old Val: ", bestValue, "New Val: ", value)
                bestValue = value
                bestMove = move

                if bestValue == -1:
                    # victory found, no need to check other states
                    break;

        return bestValue, bestMove

if __name__ == '__main__':
    from board import Board
    from piece import Color, Piece
    import numpy as np

    # Setup
    board = Board()
    board.printState()

    ## Game 1
    # piece = Piece(Color.RED, [3, 2])
    # piece2 = Piece(Color.BLACK, [2, 1])
    # draught = np.empty(shape=(8, 8), dtype=object)
    # draught[3, 2] = piece
    # draught[2, 1] = piece2
    # board.setBoard(draught)
    # board.printState()
    #
    # value, move = minimax(board, 9)
    # board.makeMove(move)
    # print(value)
    # board.printState()

    # Game 2
    piece = Piece(Color.RED, [3, 4])
    piece4 = Piece(Color.RED, [3, 2])
    piece5 = Piece(Color.RED, [2, 5])
    piece2 = Piece(Color.BLACK, [2, 3])
    piece3 = Piece(Color.BLACK, [0, 3])
    draught = np.empty(shape=(8, 8), dtype=object)
    draught[3, 4] = piece
    draught[3, 2] = piece4
    draught[2, 5] = piece4
    draught[2, 3] = piece2
    draught[0, 3] = piece3
    board.setBoard(draught)
    board.printState()

    isMax = True
    isOver, _ = board.isOver()
    while not isOver:
        value, move = minimax(board, 5, isMax)
        print("move: ", move)
        if move is None:
            print('move is None. Stopping')
            break
        board.makeMove(move)
        print("\nPlayer", board.turn, "to", move, "for value", value)
        print(board)
        isMax = not isMax