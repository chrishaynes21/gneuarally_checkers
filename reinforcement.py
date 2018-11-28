from board import Board
from copy import copy
import random
import numpy as np


def epsilonGreedy(epsilon, Q, board):
    valid_moves = board.validMoves()
    rand = np.random.uniform()
    if rand < epsilon:
        # Random Move
        return valid_moves[random.randint(0, len(valid_moves) - 1)]
    else:
        return greedy(valid_moves, Q, board)


def greedy(valid_moves, Q, board):
    Qs = np.array([Q.get(board.stateMoveTuple(m), 0) for m in valid_moves])
    return valid_moves[np.argmin(Qs)]


def finished(board):
    return len(board.validMoves()) == 0


def trainQ(nRepititions, learningRate, epsilonDecayFactor):
    epsilons = np.zeros(nRepititions)
    epsilon = 1.0
    Q = {}
    outcomes = []

    for nGames in range(nRepititions):
        epsilon *= epsilonDecayFactor
        epsilons[nGames] = epsilon
        step = 0

        # Initialize new board
        board = Board()
        done = False
        while not done:
            step += 1

            # Red will make a move
            move = epsilonGreedy(epsilon, Q, board)
            board_new = copy(board)
            board_new.makeMove(move)
            if board.stateMoveTuple(move) not in Q:
                Q[board.stateMoveTuple(move)] = 0

            if finished(board_new):
                # Red has won the checkers match
                Q[board.stateMoveTuple(move)] = 1
                done = True
                outcomes[nGames] = 1
            else:
                # Black will make a move
                black_move = np.random.choice(board_new.validMoves())
                board_new.makeMove(black_move)
                if finished(board_new):
                    # Red has won
                    Q[board.stateMoveTuple(move)] += learningRate * (-1 - Q[board.stateMoveTuple(move)])
                    done = True
                    outcomes[nGames] = -1
            if step > 1:
                Q[board_old.stateMoveTuple(move_old)] += \
                    learningRate * (Q[board.stateMoveTuple(move)] - Q[board_old.stateMoveTuple(move_old)])

            board_old, move_old = board, move
            board = board_new
    return Q, outcomes


def testQ(Q, maxSteps):
    states = []
    done = False
    board = Board()
    step = 0

    while not done and step < maxSteps:
        step += 1

        # Red will make a move
        move = greedy(board.validMoves(), Q, board)
        board.makeMove(move)
        states.append(str(board))

        if finished(board):
            print('Black won, read it and weep')
            done = True
        else:
            # Black will take a turn
            black_move = np.random.choice(board.validMoves())
            board.makeMove(black_move)
            if finished(board):
                done = True
                print('Red won, Chris sucks at programming')
    return states


if __name__ == '__main__':
    Q_ret, steps_ret = trainQ(50, 0.5, 0.7)
    print(steps_ret)
    steps = testQ(Q_ret, 20)
