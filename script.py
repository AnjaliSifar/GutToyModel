


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product

a = list(product(np.linspace(0,1,11).round(1),np.linspace(0,1,11).round(1),np.linspace(0,1,11).round(1)))
len(a)
rtable = {}

dopamine_b1 = np.zeros(1331)
for i,kk in enumerate(a):
    dopamine_b1[i] = np.round(2*kk[0]+0.6*kk[1]+0.1*kk[2],2)

dopamine_b2 = np.zeros(1331)
for i,kk in enumerate(a):
    dopamine_b2[i] = np.round(5*kk[0]+0.1*kk[1]+0.2*kk[2],2)
    
dopamine_b3 = np.zeros(1331)
for i,kk in enumerate(a):
    dopamine_b3[i] = np.round(1*kk[0]+0.1*kk[1]+0.01*kk[2],2)
    
  
max_dop = np.zeros(len(a))

for ii,ai in enumerate(a):
    rtable[ai,'bacteria_1'] = dopamine_b1[ii]
    rtable[ai,'bacteria_2'] = dopamine_b2[ii]
    rtable[ai,'bacteria_3'] = dopamine_b3[ii]
    max_dop[ii] = dopamine_b1[ii] +dopamine_b2[ii]+dopamine_b3[ii]

    
max(max_dop),max_dop[-1],max_dop

b =(tuple(np.random.choice(np.linspace(0,1,11),3).round())),np.random.choice(['bacteria_1','bacteria_2','bacteria_3'])

b,rtable.get(b)


class Bacteria():
    def __init__(self,t):
        self.population_size = np.random.randint(100,200)
        self.type = t

class Organism():
    
    def __init__(self,n):
        self.nutrients = n
        self.brain = Brain(self)
        self.gut = [Bacteria('bacteria_'+str(i+1)) for i in range(3)]
       
        
    def run(self):
        self.brain.setState()
        for i in range(100000):
            self.brain.takeAction()
        
act = list(product(np.linspace(0,0.1,2),np.linspace(-0.1,0.1,2),np.linspace(-0.1,0.1,2)))
# act.remove((0.0,0.0,0.0))
act = tuple(act)

    
class Brain():
    
    def __init__(self,organism):
        self.organism = organism
#          check what to name this 
        self.actions = act
        self.state = (0,0,1)
        
        self.alpha = 1
        self.gamma = 0.1
        self.epsilon = 0.01
        self.qtable = {}

    def setState(self):
        self.state = self.organism.nutrients
        
    def takeAction(self):
        all_q = [self.qtable.get((self.state,a),0.0) for a in self.actions]
        all_max_ind = np.where(all_q == np.max(all_q))[0]
        action = self.actions[np.random.choice(all_max_ind)]
        
#         use exploration
        if np.random.random()< self.epsilon:
            action = tuple(np.random.choice([0.1,0.0,-0.1], 3))
            
            
        self.ostate = self.state
        self.oaction = action
        self.state = tuple(np.round(np.clip(np.array(self.ostate) + np.array(action),0,1),1))

        # this is the summation of dopamine from all bacterium, say         
        self.oreward = np.sum([rtable.get((self.ostate,b.type)) for b in self.organism.gut])
#         print(""" While selecting an action: 
#         old state = {0}
#         action taken = {1}
#         new state = {2}

#         reward = {3}
#         """.format(self.ostate, action,self.state,self.oreward,self.state))
        
        self.learn()
        
      
 def learn(self):
        old_q = self.qtable.get((self.ostate,self.oaction),0.0)
#         print("""
#         old_q  = {0}
#         self state = {4}, old state = {5}
#         qtable = {3}
#         """.format(old_q,len(self.actions),self.actions,
#                   self.qtable.get((self.state,self.actions[0]),np.nan),
#                    self.state,self.ostate))

        max_new_q = np.max([self.qtable.get((self.state,a),0.0) for a in self.actions])
        
        
        newq = old_q + self.alpha*(self.oreward + self.gamma*max_new_q - old_q)
        
        self.qtable[(self.ostate,self.oaction)] = np.round(newq,3)
        
        
#         print("""
#                 Old Q = {0}
#         max New Q = {1}
#         New Q = {2}

#         Q table = {3}
#     """.format(old_q,max_new_q,newq,self.qtable))

o = Organism((0.2,0.4,0.3))
o.run()
o.brain.qtable
