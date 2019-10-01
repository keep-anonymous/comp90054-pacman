# myteam.py
''' Xinjie Lan edited this second version for offensive agent using Q learning technique
    This version implemented approximate Q learning with updated weights. Several features are defined
    Future works such as determine situations will be fulfilled. '''

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
    #weights & features for offensiveAgent
    self.weights = {'carrying':1.0, 'successorScore': -1.0, 'getFood': 1.0
                    , 'getCaplual': 0.5, 'enemyOneStepToPacman': -0.5, 'towardToGhost': -1.0,'distanceToFood': -1.0,
                    'back': -1.0, 'stop': -2.0}

    self.epsilon = 0.05
    self.alpha = 0.5
    self.discountFactor = 0.9
		#Open weights file if it exists, otherwise start with empty weights.
		#NEEDS TO BE CHANGED BEFORE SUBMISSION
    try:
      with open('weights.txt', "r") as file:
        self.weights = eval(file.read())
    except IOError:
          return
		
    #features: succesorScore, getFood, getCaplual, enemyOneStepToPacman, towardToGhost

    #attributes for offensiveAgent when learning(refer to reinforce learning project)
    

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
  def getQVal(self,gameState,action):
    features = self.getFeatures(gameState,action)
    #need more work
    value = 0.0
    
    for feature in features:
      #print"feature", feature
      #print"self.weights", self.weights[feature]
      product = features[feature]*self.weights[feature]
      #print"feature",feature
      #print"product of single feature", product
      value += product
    #print"feature value",features
    #print"weights", self.weights
    return value
  '''def evaluate(self, gameState, action):
    features = self.getFeatures(gameState,action)
    #need more work
    return features * self.weights'''

  def getHighestQWithAction(self, gameState):
    qValues = []
    actions = gameState.getLegalActions(self.index)
    actions.remove(Directions.STOP)
    if len(actions) == 0:
      return None
    else:
      for action in actions:
        #print"action", action
        qValues.append((self.getQVal(gameState,action),action))
        #print"qValues", qValues
        maxQ = max(qValues)
        #print"maxQ", maxQ
      return maxQ
  def getPolicy(self,gameState):
    #get the best action with respect to highest Q
    
    action = self.getHighestQWithAction(gameState)
    '''while action[1] == "Stop":
      action = self.getHighestQWithAction(gameState)'''
    self.weights = self.updateQ(gameState,action[1])
    #print"weights", self.weights
    #print"action in policy", action
    return action[1]

  def chooseAction(self,gameState):
    actions = gameState.getLegalActions(self.index)
    action = None
    if len(actions) !=0:
      probability = util.flipCoin(self.epsilon)
      if probability:
        #print"random"
        action = random.choice(actions)
      else:
        action = self.getPolicy(gameState)
    action = self.getPolicy(gameState)
    #print"action",action
    return action
  def updateQ(self,gameState,action):
    
    weights = self.weights
    nextState = self.getSuccessor(gameState, action)
    reward = abs(nextState.getScore() - gameState.getScore())
    if reward !=0:
      
      print"reward", reward
    features = self.getFeatures(gameState,action)
    Q = self.getHighestQWithAction(nextState)
    #print"Q", Q[0]
    currentQ = self.getQVal(gameState,action)
    #nextQ = self.getQval(nextState, action)
    #print"curretQ",currentQ
    for feature in features: 
      weights[feature] = weights[feature]+ (self.alpha*(reward+self.discountFactor*Q[0]- currentQ)*features[feature])
      #approximate Q value refer to lec slide12
    print"weights", weights
    return weights      
    



      
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    defendingFoodList = self.getFoodYouAreDefending(gameState).asList()
    walls = gameState.getWalls()
    myPosition = successor.getAgentState(self.index).getPosition()
    nextMePosition = successor.getAgentState(self.index).getPosition()
    InitialPosition = gameState.getInitialAgentPosition(self.index)
    #print"foodList",foodList
    capsules = gameState.getCapsules()
    #print"capsules", capsules
    features['getCaplual'] = 0.0
    #distanceToCapsules = self.getMazeDistance(successor.getAgentPosition(self.index),capsules)
    #print"successorPosition", successor.getAgentPosition(self.index)
    

      
    
    
    #features['successorScore'] = self.getScore(successor)
    features['successorScore'] = -len(foodList)
    #print"self.score", self.getScore(successor)
    #print"feature successorScore", features['successorScore']

    
    if action[1] == Directions.STOP:
      feature['stop'] = 2.0
    
    features['enemyOneStepToPacman'] = -1.0#detect the postion of the ghost

    enemies = []
    enemyGhost = []
    enemyPacman = []
    for opponent in self.getOpponents(gameState):
      enemy = gameState.getAgentState(opponent)
      enemies.append(enemy)
    #print"enemies", enemies
    enemyGhost = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    enemyPacman = [a for a in enemies if a.isPacman and a.getPosition() != None]
    x, y = gameState.getAgentPosition(self.index)
    dx, dy = Actions.directionToVector(action)
    xAfterMove, yAfterMove = int(x + dx), int(y + dy)
    nextPosition = (xAfterMove,yAfterMove)
    ghostPositions = []
    enemiesInvisible = False

    
    enemyGhostPosition = [Ghost.getPosition() for Ghost in enemyGhost]
    features['getFood'] = 1.0
    features['back'] = 1.1
    if len(enemyGhostPosition) == 0:
      #print"empty"
      #print"enemyGhostPosition",enemyGhostPosition
      enemiesInvisible = True
    if enemiesInvisible:
      #print"enemiesInvisible", enemiesInvisible
      for food in foodList:
        if nextMePosition == food:
          features['getFood'] += 1.0
      if len(foodList) > 0: # This should always be True,  but better safe than sorry
      
        minDistance = min([self.getMazeDistance(myPosition, food) for food in foodList])
        features['distanceToFood'] = float(minDistance)*20 /(walls.width * walls.height)
        #features['distanceToFood'] += 1.0
        #features['carrying'] = successor.getAgentState(self.index).numCarrying
      
    else:
      if len(foodList) > 0: # This should always be True,  but better safe than sorry
      
          #minDistance = min([self.getMazeDistance(myPosition, food) for food in defendingFoodList])
          #features['back'] = float(minDistance) /(walls.width * walls.height)+1.0
        features['back'] = float(self.getMazeDistance(myPosition, InitialPosition))/(walls.width * walls.height)+1.0
      if gameState.getAgentState(self.index).isPacman:
        
        if len(capsules) >0:
          distanceToCapsules = min(self.getMazeDistance(successor.getAgentPosition(self.index),capsule) for capsule in capsules)
          #if successor.getAgentPosition(self.index) == capsules:#successor
          features['getCaplual'] = distanceToCapsules
        features['enemyOneStepToPacman'] = -3.0
        
        features['back'] = float(self.getMazeDistance(myPosition, InitialPosition))/(walls.width * walls.height)+1.0
        #features['back'] = self.getMazeDistance(myPosition, InitialPosition)
      else:
        features['distanceToFood'] += 1.0
        if not successor.getAgentState(self.index).isPacman:
          features['back'] = float(self.getMazeDistance(myPosition, InitialPosition))/(walls.width * walls.height)+1.0
      if not successor.getAgentState(self.index).isPacman:
        features['back'] += 1
    #need a feature for scared ghost      
      
    '''if len(foodList) ==0:
      features['back'] == 1000.0'''
    
    #features: succesorScore, getFood, getCaplual, enemyOneStepToPacman, towardToGhost
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      
        minDistance = min([self.getMazeDistance(myPosition, food) for food in foodList])
        features['distanceToFood'] = float(minDistance) /(walls.width * walls.height)
    #print"distanceToFood", features['distanceToFood']
    features.divideAll(10.0)
    print"features", features
    return features
  #def astarFood(self,gameState):
  # Update weights file at the end of each game
  def final(self, gameState):
    print self.weights
    file = open('weights.txt', 'w')
    file.write(str(self.weights))
    

  '''def getWeights(self, gameState, action):
    return {'successorScore': 10000, 'distanceToFood': -1}'''


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -100, 'onDefense': 1000, 'invaderDistance': -1, 'stop': -100, 'reverse': -2}
