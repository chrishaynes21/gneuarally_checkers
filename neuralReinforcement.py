import numpy as np
import random
import neuralnetworks as nn
from board import *
from copy import deepcopy
import matplotlib.pyplot as plt

def epsilonGreedy(Qnet, state, epsilon):
    moves = state.validMoves()
    if np.random.uniform() < epsilon:
        move = moves[np.random.choice(range(len(moves)))] #take the randome move
        X = state.stateMoveVectorForNN(move)
        # without this, we give it a list. Needs an np array.
        X = np.array(X)
        # expects a 2d array. We want one row of a that array, so reshape first.
        X = X.reshape(1, 68)
        Q = Qnet.use(X) if Qnet.Xmeans is not None else 0 #checks to see if initialized
    else:
        qs = []
        for m in moves:
            X = state.stateMoveVectorForNN(m)
            X = np.array(X)
            X = X.reshape(1, 68)
            qs.append(Qnet.use(X)) if Qnet.Xmeans is not None else 0
        move = moves[np.argmax(qs)]
        Q = np.max(qs)
    return move, Q


# Shorthand for validMove length.
# Not sure how this handles draws, it might not at all.
def finished(state):
    return len(state.validMoves()) == 0


# Trains the Qnet with the given parameters.
# nBatches is the number of times to train the Qnet. nRepsPerBatch is the number of games for each of those batches.
# hiddenLayers is a number or list structure for the hiddenlayers i.e.: [10,10,10]
# nReplays isn't setup or tested yet, set it to 0.
# epsilon is the amount of random moves to take at the start.
# epsilonDecayFactor is the rate at which to decrease epsilon.
def trainQnet(nBatches, nRepsPerBatch, hiddenLayers, nIterations, nReplays, epsilon, epsilonDecayFactor):
    outcomes = np.zeros(nBatches*nRepsPerBatch) # holds number of steps to victory. Should hold the number of outcome, win loss or draw
    # Create a 68 to one mapping. Checker state and move pair to one Q value
    # Uses hiddenLayers
    Qnet = nn.NeuralNetwork(68, hiddenLayers, 1)
    Qnet._standardizeT = lambda x: x
    Qnet._unstandardizeT = lambda x: x
    samples = [] # I know it looks like its double initialized. But this is necessary
    #Counts the total number of reps
    repk = -1

    # Big batch, each of these creates something on which to train the Q network
    for batch in range(nBatches):
        # decay epsilon after first repitition, then cap it at .01
        if batch > 0:
            epsilon *= epsilonDecayFactor
            epsilon = max(0.01, epsilon)

        samples = []
        samplesNextStateForReplay = []

        #Simulate #reps games
        for rep in range(nRepsPerBatch):
            repk += 1
            step = 0
            done = False

            state = Board()  # create a new board to represent the state
            move, _ = epsilonGreedy(Qnet, state, epsilon) # Different than Qdict reinforcement, move first
            # Red goes first!

            while not done:
                step += 1

                # Make this move to get to nextState. Find the board state for the next move.
                stateNext = deepcopy(state)
                stateNext.makeMove(move)

                # step reinforcement is zero to let it only learn to win.
                # could give it some reinforcements though. Leave this here to allow that
                r = 0
                # Qnext is none at first, later we check if its none to equivalent of temporal difference.
                Qnext = None

                # Now check to see if the game is over. Red just played,
                # so we shouldn't need to check if red is the winner.
                if finished(stateNext):  # GG, red won
                    # goal found. Q is one for winners
                    # could try a larger reinforcement....
                    Qnext = 1
                    done = True
                    outcomes[repk] = 1
                    if rep % 10 == 0 or rep == nRepsPerBatch - 1:
                        print('Red won: batch={:d} rep={:d} epsilon={:.3f} steps={:d} outcome={:d}'.format(batch, repk, epsilon,
                                                                                    step, int(outcomes[repk])), end=', ')
                else:
                    # blacks turn
                    # choose a random choice for black.
                    blackMoves = stateNext.validMoves()
                    moveBlack = blackMoves[np.random.choice(range(len(blackMoves)))]
                    stateNext.makeMove(moveBlack)
                    if finished(stateNext):  # BG, red lost
                        Qnext = -1  # <-  negative reinforcement for loss
                        outcomes[repk] = -1
                        done = True
                        if rep % 10 == 0 or rep == nRepsPerBatch - 1:
                            print('Black won: batch={:d} rep={:d} epsilon={:.3f} steps={:d} outcome={:d}'.format(batch, repk, epsilon,
                                                                                     step, int(outcomes[repk])), end=', ')

                # At this point, were back at red's turn and can get the q from epsilon greedy if not found
                if Qnext is None:
                    moveNext, Qnext = epsilonGreedy(Qnet, stateNext, epsilon)
                else:
                    if len(stateNext.validMoves()) > 0:
                        moveNext, _ = epsilonGreedy(Qnet, stateNext, epsilon)
                    else:
                        moveNext = ((0, 0), [(0, 0)])  #placeholder, really there isn't a next move in this case because we lost.

                # append a vector, reinforcement, Qnext list to samples.
                samples.append([*state.stateMoveVectorForNN(move), r, Qnext])
                # Don't worry about what this does, not 100% necessary
                samplesNextStateForReplay.append([*stateNext.stateMoveVectorForNN(moveNext), *moveNext])

                state = deepcopy(stateNext)
                move = deepcopy(moveNext)

            # Train on samples collected from batch.
            npsamples = np.array(samples)
            X = npsamples[:, :68]  # X is the first part of the samples, the state input to the nn
            T = npsamples[:, 68:69] + npsamples[:, 69:70]  # This is the target, reinforcemen(0 for now) plus Q
            Qnet.train(X, T, nIterations, verbose=False)

            # Experience Replay: Train on recent samples with updates to Qnext.
            # Not 100% needed, could just use the top part.
            #samplesNextStateForReplay = np.array(samplesNextStateForReplay)
            for replay in range(nReplays):
                # for sample, stateNext in zip(samples, samplesNextStateForReplay):
                # moveNext, Qnext = epsilonGreedy(Qnet, stateNext, epsilon, validMovesF)
                # sample[6] = Qnext
                # print('before',samples[:5,6])
                QnextNotZero = npsamples[:, 6] != 0
                npsamples[QnextNotZero, 6:7] = Qnet.use(samplesNextStateForReplay[QnextNotZero, :])
                # print('after',samples[:5,6])
                T = npsamples[:, 5:6] + npsamples[:, 6:7]
                Qnet.train(X, T, nIterations, verbose=False)

    print('DONE')
    Qnet.outcomes = outcomes
    Qnet.samples = samples
    return Qnet


# Simulates a game using Q against a random opponent.
# Returns 1|0 for win|loss, and the string list of states if printResult = True
def useQ(Qnet, maxSteps, printResult = False):
    states = []
    done = False
    state = Board()
    step = 0
    win = 0

    while not done and step < maxSteps:
        step += 1

        # Red will make a move
        # Using epsilonGreedy with no random choice move.
        move, _ = epsilonGreedy(Qnet, state, 0)
        state.makeMove(move)
        states.append(str(state))

        if finished(state):
            done = True
            win = 1
        else:
            # Black will take a turn
            valid_moves = state.validMoves()
            black_move = valid_moves[random.randint(0, len(valid_moves) - 1)]
            state.makeMove(black_move)
            if finished(state):
                done = True
                win = 0
    if printResult:
        return win, states
    else:
        return win


# Returns a list of results that isn't quite necessary. Also prints out the percent of games won in #trials
def testQ(Qnet, trials, maxSteps = 1000, printResult = False):
    # a list of zeros and ones, zero = loss or incomplete, 1 = victory
    results = []
    for i in range(trials):
        results.append(useQ(Qnet, maxSteps, printResult))

    print("Red won {00:.2%} of games!".format(np.mean(results)))
    return results


# Plots the outcomes of the results of a trainQ.
# Use this to see if the the outcomes are learning to win. So far my tweaking of epsilon hasn't gotten there yet.
def plotOutcomes(outcomes, binRate = 10):
    winRate = []
    box = []
    for i in range(len(outcomes)):
        box.append(outcomes[i])
        if i % binRate == 0:
            winRate.append(np.mean(box))
            box = []
    plt.plot(winRate)

if __name__ == "__main__":
    print("Creating a sample Qnet with parameters 15, 10, [100, 100, 150], 20, 0, 1, .99")
    print("This took about a minute on my 8 core 3.9ghz cpu, so fair warning it takes awhile.")
    Qnet, outcomes, samples = trainQnet(15, 10, [100, 100, 100], 20, 0, 1, .99)
    print("Qnet trained! Here is a plot of outcomes")
    plotOutcomes(outcomes)
    print("Now Testing Qnet with 500 simulated games against a random opponent. Can also take awhile")
    results = testQ(Qnet, 500)
