from board import Board
from copy import deepcopy
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
            board_new = deepcopy(board)
            board_new.makeMove(move)
            if board.stateMoveTuple(move) not in Q:
                Q[board.stateMoveTuple(move)] = 0

            if finished(board_new):
                # Red has won the checkers match
                Q[board.stateMoveTuple(move)] = 1
                done = True
                outcomes.append(1)
            else:
                # Black will make a move
                valid_moves = board_new.validMoves()
                black_move = valid_moves[random.randint(0, len(valid_moves) - 1)]
                board_new.makeMove(black_move)
                if finished(board_new):
                    # Red has won
                    Q[board.stateMoveTuple(move)] += learningRate * (-1 - Q[board.stateMoveTuple(move)])
                    done = True
                    outcomes.append(-1)
            if step > 1:
                Q[board_old.stateMoveTuple(move_old)] += \
                    learningRate * (Q[board.stateMoveTuple(move)] - Q[board_old.stateMoveTuple(move_old)])
            board_old, move_old = deepcopy(board), move
            board = board_new
    return Q, outcomes


def useQ(Q, maxSteps):
    states = []
    done = False
    board = Board()
    step = 0

    while not done and step < maxSteps:
        step += 1

        # Red will make a move
        valid_moves = board.validMoves()
        move = greedy(valid_moves, Q, board)
        board.makeMove(move)
        states.append(str(board))

        if finished(board):
            print('Black won, read it and weep')
            done = True
        else:
            # Black will take a turn
            valid_moves = board.validMoves()
            black_move = valid_moves[random.randint(0, len(valid_moves) - 1)]
            board.makeMove(black_move)
            if finished(board):
                done = True
                print('Red won, Chris sucks at programming')
    return states


if __name__ == '__main__':
    Q_ret, outcomes = trainQ(50, 0.5, 0.7)
    steps = useQ(Q_ret, 1000)
    for step in steps:
        print(step)
