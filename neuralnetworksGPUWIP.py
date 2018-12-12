
import numpy as np
import scaledconjugategradient as scg
import mlutils as ml  # for draw()
from copy import copy
import torch

class NeuralNetwork:

    def __init__(self, ni, nhs, no, gpu = False):
        try:
            nihs = [ni] + list(nhs)
        except:
            nihs = [ni] + [nhs]
            nhs = [nhs]

        self.Vs = [1/np.sqrt(nihs[i]) *
                       np.random.uniform(-1, 1, size=(1+nihs[i], nihs[i+1])) for i in range(len(nihs)-1)]
        self.W = 1/np.sqrt(nhs[-1]) * np.random.uniform(-1, 1, size=(1+nhs[-1], no))
       # else:
       #     self.Vs = torch.from_numpy([1/np.sqrt(nihs[i]) *
       #                np.random.uniform(-1, 1, size=(1+nihs[i], nihs[i+1])) for i in range(len(nihs)-1)]).cuda()
       #     self.W = torch.from_numpy(1/np.sqrt(nhs[-1]) * np.random.uniform(-1, 1, size=(1+nhs[-1], no))).cuda()
        self.ni, self.nhs, self.no = ni, nhs, no
        self.Xmeans = None
        self.Xstds = None
        self.Tmeans = None
        self.Tstds = None
        self.trained = False
        self.reason = None
        self.errorTrace = None
        self.numberOfIterations = None
        self.gpu = gpu

    def __repr__(self):
        str = 'NeuralNetwork({}, {}, {})'.format(self.ni, self.nhs, self.no)
        # str += '  Standardization parameters' + (' not' if self.Xmeans == None else '') + ' calculated.'
        if self.trained:
            str += '\n   Network was trained for {} iterations. Final error is {}.'.format(self.numberOfIterations,
                                                                                           self.errorTrace[-1])
        else:
            str += '  Network is not trained.'
        return str
            
    def standardizeX(self, X):
        result = (X - self.Xmeans) / self.XstdsFixed
        result[:, self.Xconstant] = 0.0
        return result
    
    def unstandardizeX(self, Xs):
        return self.Xstds * Xs + self.Xmeans
    
    def standardizeT(self, T):
        result = (T - self.Tmeans) / self.TstdsFixed
        result[:, self.Tconstant] = 0.0
        return result
    
    def unstandardizeT(self, Ts):
        return self.Tstds * Ts + self.Tmeans
   
    def pack(self, Vs, W):
        return np.hstack([V.flat for V in Vs] + [W.flat])

    def unpack(self,w):
        first = 0
        numInThisLayer = self.ni
        for i in range(len(self.Vs)):
            self.Vs[i][:] = w[first:first+(numInThisLayer+1)*self.nhs[i]].reshape((numInThisLayer+1, self.nhs[i]))
            first += (numInThisLayer+1) * self.nhs[i]
            numInThisLayer = self.nhs[i]
        self.W[:] = w[first:].reshape((numInThisLayer+1, self.no))

    def train(self, X, T, nIterations=100, verbose=False, weightPrecision=0, errorPrecision=0, saveWeightsHistory=False):

        if type(T) == np.ndarray:
            #if self.Xmeans is None:
            #    self.Xmeans = X.mean(axis=0)
            #    self.Xstds = X.std(axis=0)
            #    self.Xconstant = self.Xstds == 0
            #    self.XstdsFixed = copy(self.Xstds)
            #    self.XstdsFixed[self.Xconstant] = 1
            #X = self.standardizeX(X)

            if T.ndim == 1:
                T = T.reshape((-1,1))

            #if self.Tmeans is None:
            #    self.Tmeans = T.mean(axis=0)
            #    self.Tstds = T.std(axis=0)
            #    self.Tconstant = self.Tstds == 0
            #    self.TstdsFixed = copy(self.Tstds)
            #    self.TstdsFixed[self.Tconstant] = 1
            #T = self.standardizeT(T)
        else:
            ## Tensor version
            #if self.Xmeans is None:
            #    self.Xmeans = X.mean().item()
            #    self.Xstds = X.std(False).item()
            #    self.Xconstant = self.Xstds == 0
            #    self.XstdsFixed = copy(self.Xstds)
            #X = self.standardizeX(X)

            if T.dim() == 1:
                T = T.reshape((-1, 1))

            #if self.Tmeans is None:
            #    self.Tmeans = T.mean().item()
            #    self.Tstds = T.std(False).item()
            #    self.Tconstant = self.Tstds == 0
            #    self.TstdsFixed = list(copy(self.Tstds))
            #    self.TstdsFixed[self.Tconstant] = 1
            #T = self.standardizeT(T)

        def objectiveF(w):
            self.unpack(w)
            Zprev = X
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                if self.gpu:
                    Zprev = np.tanh(Zprev.cuda() @ torch.from_numpy(V[1:,:] + V[0:1,:]).cuda())
                else:
                    Zprev = np.tanh(Zprev @ V[1:,:] + V[0:1,:])  # handling bias weight without adding column of 1's

            if self.gpu:
                Zprev = Zprev.cuda()
                Y = Zprev @ torch.from_numpy( self.W[1:,:] + self.W[0:1,:]).cuda()
                return np.mean(((T - Y) ** 2).cpu().numpy())
            else:
                Y = Zprev @ self.W[1:,:] + self.W[0:1,:]
                return np.mean((T-Y)**2)

        def gradF(w):
            self.unpack(w)
            Zprev = X
            Z = [Zprev]
            for i in range(len(self.nhs)):
                V = self.Vs[i]
                if self.gpu:
                    Zprev = np.tanh(Zprev.cuda() @ torch.from_numpy( V[1:,:] + V[0:1,:]).cuda())
                else:
                    Zprev = np.tanh(Zprev @ V[1:,:] + V[0:1,:])
                Z.append(Zprev)
            if self.gpu:
                Zprev = Zprev.cuda()
                Y = Zprev @ torch.from_numpy(self.W[1:,:] + self.W[0:1,:]).cuda()
            else:
                Y = Zprev @ self.W[1:,:] + self.W[0:1,:]
            delta = -(T - Y) / (X.shape[0] * T.shape[1])
            if self.gpu:
                dW = 2 * np.vstack((np.ones((1, delta.shape[0])) @ delta,
                                    Z[-1].cuda().t() @ delta))
            else:
                dW = 2 * np.vstack(( np.ones((1,delta.shape[0])) @ delta,
                                 Z[-1].T @ delta ))
            dVs = []
            if self.gpu:
                delta = (1 - Z[-1] ** 2).cuda() * (delta @ torch.from_numpy(self.W[1:, :]).t().cuda())
            else:
                delta = (1 - Z[-1]**2) * (delta @ self.W[1:,:].T)
            for Zi in range(len(self.nhs), 0, -1):
                Vi = Zi - 1 # because X is first element of Z
                if self.gpu:
                    dV = 2 * np.vstack((np.ones((1, delta.shape[0])) @ delta,
                                        Z[Zi - 1].t().cuda() @ delta))
                    dVs.insert(0, dV)
                    delta = (delta @ torch.from_numpy(self.Vs[Vi][1:, :]).t().cuda()) * (1 - Z[Zi - 1] ** 2).cuda()
                else:
                    dV = 2 * np.vstack(( np.ones((1,delta.shape[0])) @ delta,
                                     Z[Zi-1].T @ delta ))
                    dVs.insert(0,dV)
                    delta = (delta @ self.Vs[Vi][1:,:].T) * (1 - Z[Zi-1]**2)
            return self.pack(dVs, dW)


        scgresult = scg.scg(self.pack(self.Vs, self.W), objectiveF, gradF,
                            xPrecision = weightPrecision,
                            fPrecision = errorPrecision,
                            nIterations = nIterations,
                            verbose=verbose,
                            ftracep=True,
                            xtracep=saveWeightsHistory)

        self.unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace'] # * self.Tstds # to unstandardize the MSEs
        self.numberOfIterations = len(self.errorTrace)
        self.trained = True
        self.weightsHistory = scgresult['xtrace'] if saveWeightsHistory else None
        return self

    def use(self, X, allOutputs=False):
        Zprev = X
        Z = [Zprev]
        for i in range(len(self.nhs)):
            V = self.Vs[i]
            Zprev = np.tanh( Zprev @ V[1:,:] + V[0:1,:])
            Z.append(Zprev)
        Y = Zprev @ self.W[1:,:] + self.W[0:1,:]
        Y = Y
        return (Y, Z[1:]) if allOutputs else Y

    def getNumberOfIterations(self):
        return self.numberOfIterations
    
    def getErrors(self):
        return self.errorTrace
        
    def getWeightsHistory(self):
        return self.weightsHistory
        
    def draw(self, inputNames = None, outputNames = None):
        ml.draw(self.Vs, self.W, inputNames, outputNames)
