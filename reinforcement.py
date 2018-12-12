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


def trainQ(nRepititions, learningRate, epsilonDecayFactor, propagationDecayFactor=0.0):
    epsilons = np.zeros(nRepititions)
    epsilon = 1.0
    Q = {}
    outcomes = []

    for nGames in range(nRepititions):
        print(nGames)
        epsilon *= epsilonDecayFactor
        epsilons[nGames] = epsilon
        step = 0

        # Initialize new board
        board = Board()
        board_move_tuples = []
        done = False
        count = 0
        while not done and count < 1000:
            step += 1

            # Red will make a move
            move = epsilonGreedy(epsilon, Q, board)
            board_new = deepcopy(board)
            board_new.makeMove(move)
            if move not in Q:
                Q[move] = 0.0

            if finished(board_new):
                # Red has won the checkers match
                Q[move] = 1.0
                done = True
                outcomes.append(1)
            else:
                # Black will make a move
                valid_moves = board_new.validMoves()
                black_move = valid_moves[random.randint(0, len(valid_moves) - 1)]
                board_new.makeMove(black_move)
                if finished(board_new):
                    # Black has won
                    Q[move] += (-2.0 - Q[move])
                    done = True
                    outcomes.append(-1)
            board_move_tuples.append(move)
            if step > 1:
                backPropagateReinforcement(Q, board_move_tuples, learningRate, propagationDecayFactor)
            board = board_new
            count += 1
    return Q, outcomes


def backPropagateReinforcement(Q, move_tuple_list, learningRate, propagationDecayFactor):
    new_move_tuple = None
    for move_tuple in reversed(move_tuple_list):
        if new_move_tuple is None:  # 1st case, already at current value
            new_move_tuple = move_tuple
        else:  # All other cases
            Q[move_tuple] += learningRate * (Q[new_move_tuple] - Q[move_tuple])
            learningRate *= propagationDecayFactor
            if learningRate < 0.00000001:
                break
            new_move_tuple = move_tuple


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
    Q_ret, outcomes = trainQ(10000, 0.7, 0.85, 0.2)
    for move_tuple, value in sorted(Q_ret.items()):
        print('{} {}'.format(move_tuple, value))
    steps = useQ(Q_ret, 1000)
    for step in steps:
        print(step)
