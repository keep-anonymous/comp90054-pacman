# myteam.py
''' Author is Xinjie Lan
    This is the 10th version for offensive agent using approximate Q learning technique.
    Pacman will focus cap first in this version
    
    
    
     
    
    
    

    
    there are some future works 
                                
                                1 reward function can be studied more
                                2 possible potential-based reward shaping method
                                3 reconstruct the code
                                4 Consider the tradeoff when carrying food vs time and depth of enemy territory
                                5 Consider the equiblirum state when pacman and ghost fighting on boarders 
                              
    '''

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
sys.path.append('teams/Alpha-baby/')
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
    self.weights = {'carrying':0.0, 'successorScore': 0.0, 'getFood': 0.0
                    , 'getCaplual': 0.0, 'enemyOneStepToPacman': 0.0, 'towardToGhost': 0.0,'distanceToFood': 0.0,
                    'back': 0.0, 'stop': 0.0, 'eatGhost':0.0, 'reverse': -2.0}
    #self.weights = {'successorScore': 0.0, 'distanceToFood': 0.0}
    
    self.epsilon = 0.05
    self.alpha = 0.5
    self.discountFactor = 0.5

    self.reward = 0
		
    try:
      with open('./teams/Alpha-baby/weights.txt', "r") as file:
        #print"done reading weights"
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

    """if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction"""

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
 

  def getHighestQWithAction(self, gameState):
    qValues = []
    actions = gameState.getLegalActions(self.index)
    #actions.remove(Directions.STOP)
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
    print''
    return action[1]

  def chooseAction(self,gameState):
    actions = gameState.getLegalActions(self.index)
    action = None
    foodLeft = len(self.getFood(gameState).asList())
    if len(actions) !=0:
      probability = util.flipCoin(self.epsilon)
      if probability:
        #print"random"
        action = random.choice(actions)
      else:
        action = self.getPolicy(gameState)
    # if no other weights can lead the pacman go back to base, this method will be used
    """if foodLeft <= 2:
      bestDist = 9999
      for action2 in actions:
        successor = self.getSuccessor(gameState, action2)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < bestDist:
          bestAction = action2
          bestDist = dist
      return bestAction"""
    action = self.getPolicy(gameState)
    #print"action",action
    return action
  def updateQ(self,gameState,action):
     
    
    weights = self.weights
    nextState = self.getSuccessor(gameState, action)
    foodList = self.getFood(nextState).asList()
    
    
    #lastFoodList = self.getPreviousObservation().getFood(gameState).asList()
    #lastState = self.getPreviousObervation()
    #print"last state",lastState
    #reward = 0
    """if gameState.getAgentState(self.index).isPacman:
      if nextState.getAgentState(self.index).isPacman:
        reward = -abs(nextState.getScore() - gameState.getScore())
        if reward ==0:
      
          reward = -float(len(foodList))
      else:
        reward = -5
    else:
      reward = -100"""
    
    '''currentFoodList = self.getFood(gameState).asList()
    if len(foodList)- len(currentFoodList)<0:
      reward = -4
    else:
      reward = -5'''
    '''if len(foodList) == 0:
    reward = 1000'''

    x, y = gameState.getAgentPosition(self.index)
    dx, dy = Actions.directionToVector(action)
    xAfterMove, yAfterMove = int(x + dx), int(y + dy)
    nextPosition = (xAfterMove,yAfterMove)

    if gameState.hasFood(xAfterMove,yAfterMove):
      reward = 1
      
    else:
      reward = -1
    if self.getScore(nextState) - self.getScore(gameState)<=0:
      reward -= 1
    else:
      reward +=1
    
    #print'rewa', reward
    #print'self reward', self.reward
    """if self.getScore(gameState)<=0:
      
    else:
      self.reward += 1"""


    #realReward = max(self.reward,reward)
    #print"reward", realReward
    features = self.getFeatures(gameState,action)
    Q = self.getHighestQWithAction(nextState)
    #print"Q", Q[0]
    currentQ = self.getQVal(gameState,action)
    #nextQ = self.getQval(nextState, action)
    #print"curretQ",currentQ
    for feature in features: 
      weights[feature] = weights[feature]+ (self.alpha*(reward+self.discountFactor*Q[0]-currentQ)*features[feature])
      #approximate Q value refer to lec slide12

    #print"weights", weights
    return weights      
    



      
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    defendingFoodList = self.getFoodYouAreDefending(gameState).asList()
    walls = gameState.getWalls()
    myPosition = successor.getAgentState(self.index).getPosition()
    myCurrentPos =  gameState.getAgentState(self.index).getPosition()
    nextMePosition = successor.getAgentState(self.index).getPosition()
    InitialPosition = gameState.getInitialAgentPosition(self.index)
    currentCarry = gameState.getAgentState(self.index).numCarrying

    mid = gameState.data.layout.width / 2

    if gameState.isOnRedTeam(self.index):
      mid = mid - 1
    else:
      mid = mid + 1

    legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    border = [p for p in legalPositions if p[0] == mid]

    distanceToBorder = min([self.getMazeDistance(successor.getAgentPosition(self.index),borderPos) for borderPos in border])











    
    #print"foodList",foodList
    capsules = gameState.getCapsules()
    defendingCap = self.getCapsulesYouAreDefending(gameState)
    for defCap in defendingCap:
      for cap in capsules:
        if defCap == cap:
          tempCap = defCap
      capsules.remove(tempCap)
      #capsules.remove(defCap)
    print'capsules',capsules
    
    if len(foodList) > 0: # This should always be True,  but better safe than sorry      
      minDistance = min([self.getMazeDistance(myPosition, food) for food in foodList])
      features['distanceToFood'] = float(minDistance) /(walls.width * walls.height)
    
    #print"capsules", capsules
    #features['getCaplual'] = 0.0
    #distanceToCapsules = self.getMazeDistance(successor.getAgentPosition(self.index),capsules)
    #print"successorPosition", successor.getAgentPosition(self.index)
    
    features['enemyOneStepToPacman'] = 0.0#detect the postion of the ghost


    blueFood = gameState.getBlueFood().asList()
    redFood = gameState.getRedFood().asList()
    if gameState.isOnRedTeam(self.index):
      
      #print"bluefood",blueFood
      
      if len(blueFood) != 0:
        features['successorScore'] = -float(len(foodList))/len(blueFood)
    else:
      if len(redFood) != 0:
        features['successorScore'] = -float(len(foodList))/len(redFood)
    """if gameState.getScore() == 0:
      features['successorScore'] = -len(foodList)
    else:
      features['successorScore'] = gameState.getScore()"""
    


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

    ranges = []
    enemyGhostPosition = [Ghost.getPosition() for Ghost in enemyGhost]
    enemyPacmanPosition = [Pacman.getPosition() for Pacman in enemyPacman]

    #features['getFood'] = 1.0
    #features['back'] = -1.1
    if len(enemyGhostPosition) == 0:
      #print"empty"
      #print"enemyGhostPosition",enemyGhostPosition
      enemiesInvisible = True
    if enemiesInvisible:
      #print"enemiesInvisible", enemiesInvisible
      for food in foodList:
        if nextMePosition == food:
          features['getFood'] = 1.0
    else:
      # enemy in range
      #scaredTime = 0
      for ghostNearby in enemyGhost:
        if ghostNearby.scaredTimer > 10:
          isScared = True
          #scaredTime = ghostNearby.scaredTimer
        else:
          isScared = False
      #print'scare time', scaredTime
      if isScared:
        #startEatingGhost = time.time()
        distanceToGhost = min([self.getMazeDistance(successor.getAgentPosition(self.index),ghostPosition) for ghostPosition in enemyGhostPosition])
        print'eat ghost'
        features['eatGhost'] = float(distanceToGhost*10)/(walls.width * walls.height)
        #features['back'] = 1.0
      else:
        #ghost in range but not scared



      
              #if len(foodList) > 0: # This should always be True,  but better safe than sorry
        
              #minDistance = min([self.getMazeDistance(myPosition, food) for food in defendingFoodList])
              #features['back'] = float(minDistance) /(walls.width * walls.height)+1.0
        #features['back'] = -float(self.getMazeDistance(myPosition, InitialPosition))/(walls.width * walls.height)
        
        if gameState.getAgentState(self.index).isPacman:
          
          
            if len(capsules) >0:
              
              print'cap',defendingCap
              distanceToCapsules = min(self.getMazeDistance(successor.getAgentPosition(self.index),capsule) for capsule in capsules)
              #if successor.getAgentPosition(self.index) == capsules:#successor
              features['getCaplual'] = float(distanceToCapsules*10)/(walls.width * walls.height)
            
            
            #print'testing detour'
            if currentCarry != 0:
                print'back'
                print'carrying dot', currentCarry
                #features['back'] = -float(self.getMazeDistance(myPosition, distanceToBorder))/(walls.width * walls.height)
                features['back'] = -float(distanceToBorder)/(walls.width * walls.height)
            
          
            if (min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition])) ==1:
              print'enemy one step close'
              
              features['enemyOneStepToPacman'] = -float(min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition]))/(walls.width * walls.height)
              rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
              if action == rev and currentCarry !=0:
                features['reverse'] = 1
                #features['distanceToFood'] = 0
                #features['successorScore'] = 0
    
                print'detour'
              elif action == rev and currentCarry == 0:
                features['reverse'] = 1
                features['enemyOneStepToPacman'] = 0
                
                print'detour to get food'
              else:
                #if (min([self.getMazeDistance(myCurrentPos, ghostPos) for ghostPos in enemyGhostPosition])) ==1:
                print'back when one step away'
                features['reverse'] = 1
                #features['distanceToFood'] = 0
                #features['successorScore'] = 0
                #features['back'] =  -float(self.getMazeDistance(myPosition, InitialPosition))/(walls.width * walls.height)
                #features['enemyOneStepToPacman'] = 0
            
              #print'run'
              #features['enemyOneStepToPacman'] = -float(self.getMazeDistance(myPosition, InitialPosition))/(walls.width * walls.height)
                
            

              
                
            
            
              
            
              #features['back'] = -float(self.getMazeDistance(myPosition, InitialPosition))/(walls.width * walls.height)
              #features['enemyOneStepToPacman'] = -float(self.getMazeDistance(myPosition, InitialPosition))/(walls.width * walls.height)
            
              
        else:
          #print'run from my side'
          if min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition]) <4:
            #features['back'] = -float(self.getMazeDistance(myPosition, distanceToBorder))/(walls.width * walls.height)
            features['back'] = -float(distanceToBorder)/(walls.width * walls.height)
            
            print'close'
          
            #if min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition]) == 1:
            #print'enter dont run'
            features['enemyOneStepToPacman'] = -float(min(self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition))/(walls.width * walls.height)
            rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if action == rev:
              features['reverse'] = 1
            #features['stop']= 1
              
            #features['back'] = -float(self.getMazeDistance(myPosition, InitialPosition))/(walls.width * walls.height)
          #features['back'] = self.getMazeDistance(myPosition, InitialPosition)

    
    
      
            
            
    
       #print"self.score", self.getScore(successor)
    #print"feature successorScore", features['successorScore']
    if len(foodList) <=2 or currentCarry >= 6:
      features['distanceToFood'] = 0
      features['successorScore'] = 0
      print'head home'
      isScared1 = False
      #features['back'] = -float(self.getMazeDistance(myPosition, distanceToBorder))/(walls.width * walls.height)
      features['back'] = -float(distanceToBorder)/(walls.width * walls.height)

      
      #features['stop'] = -1
      if len(enemyGhostPosition) != 0:
        if (min([self.getMazeDistance(myCurrentPos, ghostPos) for ghostPos in enemyGhostPosition])) ==1 or (min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition]))==1:
          #features['back'] = -1.1
          if len(capsules) >0:
              #print'cap'
              distanceToCapsules = min(self.getMazeDistance(successor.getAgentPosition(self.index),capsule) for capsule in capsules)
              #if successor.getAgentPosition(self.index) == capsules:#successor
              features['getCaplual'] = float(distanceToCapsules)/(walls.width * walls.height)
          
          print'reverse'
          features['enemyOneStepToPacman'] = -float(min(self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition))/(walls.width * walls.height)
          rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
          if action == rev:
            print'testing reverse'
            features['reverse'] = 1
            #features['getFood'] = 0.0
      

      """for ghostNearby in enemyGhost:
        if ghostNearby.scaredTimer > 20:
          isScared1 = True
          #scaredTime = ghostNearby.scaredTimer
        else:
          isScared1 = False
          #print'scare time', scaredTime
      if isScared1:
        #startEatingGhost = time.time()
        distanceToGhost = min([self.getMazeDistance(successor.getAgentPosition(self.index),ghostPosition) for ghostPosition in enemyGhostPosition])
        
        features['eatGhost'] = -float(distanceToGhost)/(walls.width * walls.height)"""
      
        

    features.divideAll(10.0)
    #print"features",features
    return features

  #def astarFood(self,gameState):
  




  
  # Update weights file at the end of each game
  def final(self, gameState):
    #print self.weights
    file = open('./teams/Alpha-baby/weights.txt', 'w')
    file.write(str(self.weights))
    

  '''def getWeights(self, gameState, action):
    return {'successorScore': 10000, 'distanceToFood': -1}'''

class DefensiveReflexAgent(ReflexCaptureAgent):
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

    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        # Variables used to verify if the agent os locked
        self.foodNum = 999
        self.trappedTime = 0
        self.defendingFood=[]
        self.flag=0
        self.target=()

    def evaluate(self, gameState, action):
        """
    Computes a linear combination of features and feature weights
    """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    

    def chooseAction(self, gameState):
        """
    Picks among the actions with the highest Q(s,a).
    """
        actions = gameState.getLegalActions(self.index)
        #remove the stop action
        actions.remove(Directions.STOP)
        

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
    def getBorder(self,gameState):

      mid = gameState.data.layout.width/2

      if self.red:
        mid = mid - 1
      else:
        mid = mid + 1

      legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
      border = [p for p in legalPositions if p[0] == mid]
      return border


    def getFeatures(self, gameState, action):
      features = util.Counter()
      successor = self.getSuccessor(gameState, action)
      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()
      myFood=self.getFoodYouAreDefending(gameState).asList()
      borderPos=random.choice(self.getBorder(gameState))
      

      if self.flag==0:
        self.flag=1
        self.defendingFood=myFood
        self.target=borderPos

      #locate the target in the dispeared food area
      if len(self.defendingFood)>len(myFood):
        target=list(set(self.defendingFood)-set(myFood))[0]
        self.defendingFood=myFood
        self.target=target
      
      
      if myState.isPacman:
       features['onDefense'] = 0
      else:
       features['onDefense'] = 1
      opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      invaders = [i for i in opponents if i.isPacman and i.getPosition() != None]
      features['numInvaders'] = len(invaders)

      if len(invaders) > 0:
        invaderDist = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
        
        #follow the invader
        features['invaderDist'] = min(invaderDist)
        
        if gameState.getAgentState(self.index).scaredTimer > 0:
          reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
          features['reverse'] = 1
            
          print'reverse'

        self.target=borderPos
        
        


      if len(invaders) == 0:
        #protect food or patrol around border
        features['ToFoodOrBorder'] = self.getMazeDistance(myPos,self.target)
      else:
        features['ToFoodOrBorder'] = 0

      

      return features

    def getWeights(self, gameState, action):
      return { 'onDefense': 100, 'invaderDist': -10,'reverse': -10,
              'ToFoodOrBorder': -20,'numInvaders':-1000}





