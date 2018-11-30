# function MiNlMAX-DECisiON(gam<?) returns an operator
# for each op in OPERATORSfgame] do
# VALUE[op] — MINIMAX-VALUE(APPLY(op, game), game)
# end
# return the op with the highest VALUE[op]

from piece import Color
from copy import copy, deepcopy

inf = float('infinity')

def minimax(board, depthLeft, isMax):

    isOver, _ = board.isOver()

    if isOver or depthLeft == 0:
        maximizePlayer = None

        if isMax:
            maximizePlayer = board.turn
        else:
            maximizePlayer = Color.RED if board.turn == Color.BLACK else Color.BLACK

        return board.getUtility(type='mm', maximizePlayer=maximizePlayer), None

    if isMax:
        bestVal = -inf
        bestMove = None

        for move in board.getMoves():
            # Make copy of board to make the move on so original board remains the same
            tempBoard = deepcopy(board)

            # Apply a move to current state
            tempBoard.makeMove(move)

            value, _ = minimax(tempBoard, depthLeft - 1, False)

            if value > bestValue:
                # Value for this move is better than moves tried so far from this state.
                bestValue = value
                bestMove = move

        return bestValue, bestMove

    else:
        bestVal = inf
        bestMove = None

        for move in board.getMoves():
            # Make copy of board to make the move on so original board remains the same
            tempBoard = deepcopy(board)

            # Apply a move to current state
            tempBoard.makeMove(move)

            value, _ = minimax(tempBoard, depthLeft - 1, True)

            if value < bestValue:
                # Value for this move is better than moves tried so far from this state.
                bestValue = value
                bestMove = move

        return bestValue, bestMove