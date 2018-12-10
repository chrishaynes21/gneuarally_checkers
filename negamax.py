#
# This code based off of code from notebook '12 Negamax'
# http://nbviewer.jupyter.org/url/www.cs.colostate.edu/~anderson/cs440/notebooks/12%20Negamax.ipynb
#

from copy import copy, deepcopy

inf = float('infinity')

def negamax(board, depthLeft, alpha, beta):
    # If at terminal state or depth limit, return utility value and move None
    isOver, _ = board.isOver()

    if isOver or depthLeft == 0:
        return board.getUtility(), None

    # Find best move and its value from current state
    bestValue = -inf
    bestMove = None

    for move in board.validMoves():
        # Make copy of board to make the move on so original board remains the same
        tempBoard = deepcopy(board)

        # Apply a move to current state
        tempBoard.makeMove(move)

        # print('trying',tempBoard)
        # Use depth-first search to find eventual utility value and back it up.
        #  Negate it because it will come back in context of next player
        value, _ = negamax(tempBoard, depthLeft - 1, -beta, -alpha)
        value = - value

        if value > bestValue:
            # Value for this move is better than moves tried so far from this state.
            bestValue = value
            bestMove = move

            if bestValue == 2:
                # victory found, no need to check other states
                break;

        if bestValue > alpha:
            alpha = bestValue

        if alpha >= beta:
            #print("pruning alpha: ", alpha, "beta: ", beta)
            break;

    return bestValue, bestMove

if __name__ == '__main__':
    from board import Board
    from piece import Color, Piece
    import numpy as np

    # Setup
    board = Board()
    board.printState()

    # Game 1
    piece = Piece(Color.RED, [3, 2])
    piece2 = Piece(Color.BLACK, [2, 1])
    draught = np.empty(shape=(8, 8), dtype=object)
    draught[3, 2] = piece
    draught[2, 1] = piece2
    board.setBoard(draught)
    board.printState()

    value, move = negamax(board, 9, -inf, inf)
    board.makeMove(move)
    print(value)
    board.printState()

    # # Game 2
    # piece = Piece(Color.RED, [3, 4])
    # piece4 = Piece(Color.RED, [3, 2])
    # piece5 = Piece(Color.RED, [2, 5])
    # piece2 = Piece(Color.BLACK, [2, 3])
    # piece3 = Piece(Color.BLACK, [0, 3])
    # draught = np.empty(shape=(8, 8), dtype=object)
    # draught[3, 4] = piece
    # draught[3, 2] = piece4
    # draught[2, 5] = piece5
    # draught[2, 3] = piece2
    # draught[0, 3] = piece3
    # board.setBoard(draught)
    # board.printState()
    #
    # print("turn: ", board.turn)
    # print("Utility", board.getUtility("nm"))
    #
    # isOver, _ = board.isOver()
    # while not isOver:
    #     value, move = negamax(board, 5, -inf, inf)
    #     print("move: ", move)
    #     if move is None:
    #         print('move is None. Stopping')
    #         break
    #     print("\nPlayer", board.turn, "to", move, "for value", value)
    #     board.makeMove(move)
    #     print(board)
    #     isOver, _ = board.isOver()

    # Game 3
    # board = Board()
    # board.printState()
    #
    # isOver, _ = board.isOver()
    # while not isOver:
    #     if board.turn == Color.BLACK:
    #         move = board.validMoves()[int(len(board.validMoves()) / 2)]
    #         print("\nPlayer", board.turn, "to", move)
    #         board.makeMove(move)
    #     else:
    #         value, move = negamax(board, 10, -inf, inf)
    #         print("move: ", move)
    #         if move is None:
    #             print('move is None. Stopping')
    #             break
    #         print("\nPlayer", board.turn, "to", move, "for value", value)
    #         board.makeMove(move)
    #     print(board)
    #     isOver, _ = board.isOver()