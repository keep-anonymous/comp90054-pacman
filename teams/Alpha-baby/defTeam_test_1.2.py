# baselineTeam.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions
import game
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
 
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()    
    features['successorScore'] = -len(foodList)#self.getScore(successor)

    # Compute distance to the nearest food

    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1}

class DefensiveReflexAgent(ReflexCaptureAgent):
  
  def __init__(self, index):
      CaptureAgent.__init__(self, index)
      # Variables used to verify if the agent os locked
      
      self.defendingFood=[]
      self.index = index
      self.target = None
      self.flag = 0

  def getBorder(self,gameState):
  
    mid = gameState.data.layout.width / 2

    if self.red:
      mid = mid - 1
    else:
      mid = mid + 1

    legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    border = [p for p in legalPositions if p[0] == mid]
    return border


  def isPacman(self, gameState, action):
    successor = self.getSuccessor(gameState, action)
    myState = successor.getAgentState(self.index)
    return myState.isPacman

  

  def chooseAction(self, gameState):
     # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # our patrol points probabilities.
    
    myFood= self.getFoodYouAreDefending(gameState).asList()
    border = self.getBorder(gameState)
    mypos = gameState.getAgentPosition(self.index)
    homePos = gameState.getInitialAgentPosition(self.index)
    actions = gameState.getLegalActions(self.index)
    
    

    if self.flag==0:
      self.flag=1
      self.defendingFood=myFood
     

    if mypos == self.target:
      self.target = None


    x = self.getOpponents(gameState)
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = filter(lambda x: x.isPacman and x.getPosition() != None, enemies)
    invaderPos = [agent.getPosition() for agent in invaders]
        

    if len(self.defendingFood)>len(myFood):
      print "protect food!"
      target=list(set(self.defendingFood)-set(myFood))[0]
      self.defendingFood=myFood
      self.target=target

    elif len(invaders) > 0:
      followTarget = min(invaderPos, key=lambda x: self.getMazeDistance(mypos,x))

      if gameState.getAgentState(self.index).scaredTimer > 0:
        print "escape!"
        gActions = []
        values = []
        for a in actions:
          newstate = gameState.generateSuccessor(self.index, a)
          if not (a == Directions.STOP or self.isPacman(gameState, a)):
            new = newstate.getAgentPosition(self.index)
            gActions.append(a)
            values.append(self.getMazeDistance(followTarget,new))
        
  
        bestaction = min(values)
        tie = filter(lambda x: x[0] == bestaction, zip(values, gActions))
        followAction = random.choice(tie)[1]
        escapeAction = Actions.reverseDirection(followAction)
        if (escapeAction not in actions) or (self.isPacman(gameState, escapeAction)):
          for a in actions:
            if not (a==followAction or self.isPacman(gameState, a)):
              return a
        else:
          return escapeAction
          
      else:
        print "follow the invader!!!"
        self.target = followTarget

   
    if self.target == None and len(self.getFoodYouAreDefending(gameState).asList()) <= 6:
      food = self.getFoodYouAreDefending(gameState).asList() \
             + self.getCapsulesYouAreDefending(gameState)
      self.target = random.choice(food)

    elif self.target == None:
      print "patrol around the border~"
      
      self.target = random.choice(border)

  
    
    goodActions = []
    fvalues = []
    for a in actions:
      new_state = gameState.generateSuccessor(self.index, a)
      if not (a == Directions.STOP or self.isPacman(gameState, a)):
        newpos = new_state.getAgentPosition(self.index)
        goodActions.append(a)
        fvalues.append(self.getMazeDistance(self.target,newpos))

  
    best = min(fvalues)
    ties = filter(lambda x: x[0] == best, zip(fvalues, goodActions))

    # print 'eval time for defender agent %d: %.4f' % (self.index, time.time() - start)
    
    return random.choice(ties)[1]
    


