import numpy as np
import matplotlib.pyplot as plt
import math

class S1:
    
    def __init__(self, dx=0.01, rg=(-1, 1)):
        self.dx = dx
        self.N = int((rg[1] - rg[0])/dx)
        
    def numerov(self, potential, psi0, energy):
        psi = psi0.copy()
        f1 = 2 * (potential[0] - energy)
        q0 = psi0[0]
        q1 = psi[1] * (1 - (dx**2) * f1)
        for ix in range(2, self.N):
            q2 = 2.0*q1 - q0 + (dx**2) * f1 * psi[ix-1]
            q0, q1 = q1, q2
            f1 = 2 * (potential[ix] - energy)
            psi[ix] = q1/(1.0 - (dx**2)*f1)
        psi /= psi.sum()
        return psi
    
    def deviation(self, psi, psi0):
        return psi[self.N-1] - psi0[self.N-1]
    
    def shooting(self, potential, psi0, E0=1.0, dE=0.01, eps=0.001):
        eval_psi = lambda energies: map(lambda en: self.numerov(potential, psi0, en), energies)
        eval_dev = lambda wfs: map(lambda wf: self.deviation(wf, psi0), wfs)
            
        E = [E0, E0 + dE]
        psi = eval_psi(E)
        d = eval_dev(psi)
        
        while d[0] * d[1] > 0:
            E[1] += dE
            psi = [psi[1], self.numerov(potential, psi0, E[1])]
            d = [d[1], self.deviation(psi[1], psi0)]
    
        E = [E[1] - dE, E[1] - dE/2, E[1]]
        psi = eval_psi(E)
        d = eval_dev(psi)
        
        while math.fabs(d[1]) > eps:
            print('deviation: ' + str(d[1]))
            if d[0] * d[1] < 0:
                E[1:] = [(E[0] + (E[1] - E[0])/2), E[1]]
            elif d[1] * d[2] < 0:
                E[:2] = [E[1], (E[1] + (E[2] - E[1])/2)]
            else:
                raise Exception('Energy step too large; contains multiple sign-flips')
            psi = eval_psi(E)
            d = eval_dev(psi)
        
        return psi[1]
        

class SeparableS2:
    
    def __init__(self, dx=0.01, dy=0.01, rg=((-1, 1), (-1, 1))):
        self.xs = S1(dx=dx, rg=rg[0])
        self.ys = S1(dx=dy, rg=rg[1])
        
    def shooting(self, potential, psi0, E0, dE=0.01, eps=0.001):
        psi_x = self.xs.shooting(potential[0], psi0[0], E0[0], dE, eps)
        psi_y = self.ys.shooting(potential[1], psi0[1], E0[1], dE, eps)
        return np.outer(psi_x, psi_y)