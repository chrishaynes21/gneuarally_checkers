from reinforcement import *
from NeuralReinforcement import *
from minimax import *
from negamax import *
import numpy as np
from nnutils import *
import threading as tr


def finished(state):
    return len(state.validMoves()) == 0


def playGame(redMoveFunction, blackMoveFunction, maxSteps, printResult, Q = None):
    states = []
    done = False
    state = Board()
    step = 0
    win = 0

    while not done and step < maxSteps:
        step += 1

        if Q is not None:
            red_move = redMoveFunction(state, Q)
        else:
            red_move = redMoveFunction(state)
        state.makeMove(red_move)
        states.append(str(state))

        if finished(state):
            done = True
            win = 1
        else:
            black_move = blackMoveFunction(state)
            if black_move is None:
                done = True
                win = 1
            else:
                state.makeMove(black_move)
            if finished(state):
                done = True
                win = 0
    if printResult:
        return win, states
    else:
        return win


def testAI(redMoveFunction, blackMoveFunction, trails, maxSteps = 1000, printResult = False, Q = None):
    results = []
    for _ in range(trails):
        results.append(playGame(redMoveFunction, blackMoveFunction, maxSteps, printResult, Q))

    print("Red won {00:.2%} of games!".format(np.mean(results)))
    print("With function", redMoveFunction)
    return results


def reinforcementMoveFunction(state, Q):
    valid_moves = state.validMoves()
    move = greedy(valid_moves, Q, state)
    return move


def nnReinforcementMoveFunction(state, Q):
    move, _ = epsilonGreedy(Q, state, 0)
    return move


def negamaxMoveFunction(state):
    _, move = negamax(state, 9, -inf, inf)
    return move


def minimaxMoveFunction(state):
    _, move = minimax(state, 9, -inf, inf)
    return move


def randomMoveFunction(state):
    valid_moves = state.validMoves()
    random_move = valid_moves[random.randint(0, len(valid_moves) - 1)]
    return random_move

if __name__ == '__main__':

    # Train the traditional reinforcement strategy
    Q_ret, outcomes = trainQ(100, 0.5, 0.7, 0.2)
    print("Trained traditional Q table")
    # Test the traditional reinforcement strategy against random opponent
    print("Testing Q table against random moves")
    testAI(reinforcementMoveFunction, randomMoveFunction, 1000, 1000, False, Q_ret)

    # Load the really big network from disk
    Qnet = loadNetwork("1000.qnet")
    print("Saved Neural Network loaded from disk")
    print("Testing NN against random moves")
    testAI(nnReinforcementMoveFunction, randomMoveFunction, 1000, 1000, False, Qnet)

    #Play each against negamax and minimax
    # do each on a thread
    #rLmmThread = tr.Thread(testAI, reinforcementMoveFunction, minimaxMoveFunction, 1, 1500, False, Q_ret)
    rLMM = tr.Thread(target=testAI, args=[reinforcementMoveFunction, minimaxMoveFunction, 1, 1500, False, Q_ret])
    nnMM = tr.Thread(target=testAI, args=[nnReinforcementMoveFunction, minimaxMoveFunction, 1, 1500, False, Qnet])

    rLNM = tr.Thread(target=testAI, args=[reinforcementMoveFunction, minimaxMoveFunction, 1, 1500, False, Q_ret])
    nnNM = tr.Thread(target=testAI, args=[nnReinforcementMoveFunction, minimaxMoveFunction, 1, 1500, False, Qnet])

    # GO!
    rLMM.start(), nnMM.start(), rLNM.start(), nnNM.start()

    print("Started playing Q table and network against minimax and negamax on multiple threads")

    # Wait...
    rLMM.join(), nnMM.join(), rLNM.join(), nnNM.join()

    print("All finished up!")