import random
from copy import deepcopy

import numpy as np

from board import Board


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
    return valid_moves[np.argmax(Qs)]


def finished(board):
    return len(board.validMoves()) == 0


def trainQ(nRepititions, learningRate, epsilonDecayFactor, propagationDecayFactor):
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
        board_smts = []
        done = False
        while not done:
            step += 1

            # Red will make a move
            move = epsilonGreedy(epsilon, Q, board)
            board_new = deepcopy(board)
            board_new.makeMove(move)
            if board.stateMoveTuple(move) not in Q:
                Q[board.stateMoveTuple(move)] = 0.0

            if finished(board_new):
                # Red has won the checkers match
                Q[board.stateMoveTuple(move)] = 1.0
                done = True
                outcomes.append(1)
            else:
                # Black will make a move
                valid_moves = board_new.validMoves()
                black_move = valid_moves[random.randint(0, len(valid_moves) - 1)]
                board_new.makeMove(black_move)
                if finished(board_new):
                    # Black has won
                    Q[board.stateMoveTuple(move)] += learningRate * (-1.0 - Q[board.stateMoveTuple(move)])
                    done = True
                    outcomes.append(-1)
            board_smts.append(board.stateMoveTuple(move))
            if step > 1:
                back_propagate_reinforcement(Q, board_smts, learningRate, propagationDecayFactor)
            board = board_new
    return Q, outcomes


def back_propagate_reinforcement(Q, smt_list, learningRate, propagationDecayFactor):
    new_smt = None
    for smt in reversed(smt_list):
        if new_smt is None:  # 1st case, already at current value
            new_smt = smt
        else:  # All other cases
            Q[smt] += learningRate * (Q[new_smt] - Q[smt])
            learningRate *= propagationDecayFactor
            new_smt = smt


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
            print('Red won, read it and weep')
            done = True
        else:
            # Black will take a turn
            valid_moves = board.validMoves()
            black_move = valid_moves[random.randint(0, len(valid_moves) - 1)]
            board.makeMove(black_move)
            if finished(board):
                done = True
                print('Black won, Chris sucks at programming')
    return states


if __name__ == '__main__':
    Q_ret, outcomes = trainQ(1000, 1, 0.7, 0.9)
    for smt, value in Q_ret.items():
        print('{} {}'.format(smt, value))
    steps = useQ(Q_ret, 1000)
    for step in steps:
        print(step)
