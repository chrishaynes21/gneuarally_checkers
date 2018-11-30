#
# This code based off of code from notebook '12 Negamax'
# http://nbviewer.jupyter.org/url/www.cs.colostate.edu/~anderson/cs440/notebooks/12%20Negamax.ipynb
#

from copy import copy, deepcopy

inf = float('infinity')

def negamax(board, depthLeft):
    # If at terminal state or depth limit, return utility value and move None
    isOver, _ = board.isOver

    if isOver or depthLeft == 0:
        return board.getUtility(), None

    # Find best move and its value from current state
    bestValue = -inf
    bestMove = None

    for move in board.getMoves():
        # Make copy of board to make the move on so original board remains the same
        tempBoard = deepcopy(board)

        # Apply a move to current state
        tempBoard.makeMove(move)

        # print('trying',game)
        # Use depth-first search to find eventual utility value and back it up.
        #  Negate it because it will come back in context of next player
        value, _ = negamax(tempBoard, depthLeft - 1)
        value = - value

        if value > bestValue:
            # Value for this move is better than moves tried so far from this state.
            bestValue = value
            bestMove = move

    return bestValue, bestMove