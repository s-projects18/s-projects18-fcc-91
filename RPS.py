import numpy as np
import itertools
import helper as hlp
import sys

# prev_play = guess = 'R' | 'P' | 'S'
bt = hlp.Batches()
weight1=0
weight2=0

# TODO: poor player: 
# at move 751 predictions get wrong for a while
# no error, no warnings...
'''
701 [0.0, 0.99999, 0.00034]
751 [0.0, 0.0, 0.0]
801 [0.0, 0.0, 1.0]
851 [0.0, 0.0, 0.0]
901 [0.0, 1.0, 0.0]
'''
def player(prev_play, opponent_history=[]):
  # --------- CONFIG ---------
  # the more, the better and slower
  n = 1000      # train on every n-th move
  epochs = 10 # how often run through traning loop

  global weight1, weight2
  info = False
  
  d = [[0,0,0]]
  i = "RPS".index(prev_play)
  d[0][i] = 1

  # TEST with cartesian -> out of memory
  # cartesian product (order matters, with replacement)
  dataY = list(itertools.product([0,1], repeat=3))
  dataY = list(filter(lambda x: x[i]==1, dataY))
  dataY = [list(x) for x in dataY] # tuples 2 list

  for dy in dataY:
    bt.addRows(d, [dy])  

  # ????
  # model fails at quincy/poor2, but basically works with poor!
  #bt.addRows(d, d)
  X, y = bt.getData()

  # we will retrain our model not on every move
  pr = []

  if len(X) % n == 0:
    weight1, weight2 = hlp.train(X, y, False, False, epochs)
    info = str(len(X))

  i_pred = np.array([1,1,1])
  pr = hlp.predict([weight1, weight2], i_pred)
  #print(pr)
  #sys.exit()

  # TODO
  # check foreach move prob and return a move
  # eg if most prob move is R the return P
  # -> extremely slow! 
  if pr[0]>pr[1] and pr[0]>pr[2]: # R
    guess = 'P'
  elif pr[1]>pr[0] and pr[1]>pr[2]: # P
    guess = 'S'
  else: # S
    guess = 'R'

  pr2 = [round(x,5) for x in pr]
  if info != False:
    print(info, pr2)
  return guess

# random strategy: about 50%
# I don't see whether my move was successful,
# cause my moves are not stored
''''
def player(prev_play, opponent_history=[]):
  l = ['R', 'P','S']
  guess = l[np.random.randint(0,2)]
  return guess
'''

# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.
'''
def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)

    guess = "R"
    if len(opponent_history) > 2:
        guess = opponent_history[-2]

    return guess
'''
