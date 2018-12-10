import NeuralReinforcement as nr
import numpy as np
import time
import datetime
import dill as pickle # this library is like regular pickle but tastier. Serializes lambdas.
import random
from board import *
import matplotlib.pyplot as plt


def loadNetwork(filename):
    networkDir = "qnets/"
    filename = networkDir + filename
    with open(filename, 'rb') as network_output_file:
        return pickle.load(network_output_file)

def saveNetwork(filename, Qnet):
    networkDir = "qnets/"
    filename = networkDir + filename
    with open(filename, 'wb') as network_output_file:
        pickle.dump(Qnet, network_output_file)

def trainNN(nBatches, nRepsPerBatch, hiddenLayers, nIterations, epsilonDecayFactor):
    epsilon = 1 #haven't needed to change this
    nReplays = 0 # not used
    # nBatches, nRepsPerBatch, hiddenLayers, nIterations, nReplays, epsilon, epsilonDecayFacto
    start = time.time()
    Qnet = nr.trainQnet(nBatches, nRepsPerBatch, hiddenLayers, 
                                           nIterations, nReplays, epsilon, epsilonDecayFactor)
    end = time.time()
    print("Training took {}!".format(str(datetime.timedelta(seconds = end-start))))
    return Qnet

def plotOutcomes(outcomes, binRate = 10):
    winRate = []
    box = []
    for i in range(len(outcomes)):
        box.append(outcomes[i])
        if i % binRate == 0:
            winRate.append(np.mean(box))
            box = []
    plt.plot(winRate)

def finished(state):
    return len(state.validMoves()) == 0

# plays a game of two random opponents. 
# I used a loop to do this around 100 times to see the results
# Found that checkers seems to be a fair game, about 50/50 win rete
def randomQ(maxSteps):
    states = []
    done = False
    state = Board()
    step = 0

    while not done and step < maxSteps:
        step += 1

        # Red will make a move
        valid_moves = state.validMoves()
        move = valid_moves[random.randint(0, len(valid_moves) - 1)]
        state.makeMove(move)
        states.append(str(state))

        if finished(state):
            done = True
            return 1
        else:
            # Black will take a turn
            valid_moves = state.validMoves()
            black_move = valid_moves[random.randint(0, len(valid_moves) - 1)]
            state.makeMove(black_move)
            if finished(state):
                done = True
                return 0
    return states

## Example of net.
## if this doesn't load, don't train it. It took 5 hours on my 8 core cpu. 
##try:
##    qnet1000 = loadNetwork("1000.qnet")
##except:
##    qnet1000 = trainNN(2, 500, [1000], 10, .97)
##    saveNetwork("1000.qnet", qnet1000)

## Example of testing
## (network, number of games, maxsteps per game)
## prints out percent of wins. this one wins 98%
## result = nr.testQ(qnet1000, 1000, maxsteps = 1500)

