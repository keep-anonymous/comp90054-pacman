# myteam.py
''' The author of attacking agentis XINJIE LAN and the author of defensive agent is LIUYI CHAI
    This is the 15th version for offensive agent using approximate Q learning technique.
    This version edited that after eat capluse, pacman will first eat nearby ghost then eat food then go back to home. Although this was desined in 14th version, there are still some bugs
    Also fixed the issue atart will be called even when ghost is scared.
    This version should be the final version for the competition. Reuslts will be observed for the last pre-competition if anything needs to change. 
    Defence not change in this version

    

  

    The author of Defensive agent is LiuyiChai. her ghost will go directly when dots disappeared 
    
    
    
     
    
    
    

    
    there are some future works  
                                
                                1 reward function can be studied more
                                2 possible potential-based reward shaping method
                                3 reconstruct the code
                               
                                
                              
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
    self.weights = {'carrying':-1.0957026245589823, 'successorScore': 34.932620346482636, 'getFood': 0.0
                    , 'getCaplual': -10.372649567819039, 'enemyOneStepToPacman': 3.073831162329436, 'towardToGhost': 0.0,'distanceToFood': -11.972100433027324,
                    'back': 20.489084718452254, 'stop': 0.0, 'eatGhost':-19.741091990680903, 'reverse': -6.672254703081991}
    #self.weights = {'successorScore': 0.0, 'distanceToFood': 0.0}
    
    self.epsilon = 0.05
    self.alpha = 0.5
    self.discountFactor = 0.5
    self.farPoint = (0,0)
    self.aSt = False
    self.actionList = []
    self.stopAction = False
    self.skipAstar = False
    self.nextPositionGlo = (0,0)

    self.detourFood = False

    self.reward = 0
		
    """try:
      with open('weights.txt', "r") as file:
        #print"done reading weights"
        self.weights = eval(file.read())
    except IOError:
          return"""
		
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
    #print''
    move = action[1]
    if self.aSt:
      move = Directions.STOP
      
    return move
    

  def chooseAction(self,gameState):
    start = time.time()
   
    
    actions = gameState.getLegalActions(self.index)
    action = None
    foodLeft = len(self.getFood(gameState).asList())

    myCurrentPos =  gameState.getAgentState(self.index).getPosition()
    InitialPosition = gameState.getInitialAgentPosition(self.index)
    enemies = []
    enemyGhost = []
    enemyPacman = []
    for opponent in self.getOpponents(gameState):
      enemy = gameState.getAgentState(opponent)
      enemies.append(enemy)
    #print"enemies", enemies
    enemyGhost = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    enemyPacman = [a for a in enemies if a.isPacman and a.getPosition() != None]
    
    ghostPositions = []
    disToG = 6666
    enterBorder = False
    

    ranges = []
    enemyGhostPosition = [Ghost.getPosition() for Ghost in enemyGhost]
    enemyPacmanPosition = [Pacman.getPosition() for Pacman in enemyPacman]
    



    mid = gameState.data.layout.width / 2

    if gameState.isOnRedTeam(self.index):
      mid = mid - 1
    else:
      mid = mid + 1

    legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    border = [p for p in legalPositions if p[0] == mid]

    if len(enemyGhostPosition) >0 and not gameState.getAgentState(self.index).isPacman:
      disToG = min([self.getMazeDistance(myCurrentPos, ghostPos) for ghostPos in enemyGhostPosition])
      for scaredEnemy in enemyGhost:
        if scaredEnemy.scaredTimer>0:
          enterBorder = True
    
    #print'position in action',myCurrentPos
    #print'If Astar111',self.aSt
    if disToG <5 and not enterBorder:
      randomPoint = random.choice(border)
      while randomPoint == myCurrentPos:
        randomPoint = random.choice(border)
      self.farPoint = randomPoint
      self.aSt = True
      #print'If Astar',self.aSt
      #print'current pacman position',myCurrentPos
      self.stopAction = True
    
    actionList = []
    if myCurrentPos!= self.farPoint:
      
      if self.aSt:
        bestDist = 9999
        #for action2 in actions:
        #successor = self.getSuccessor(gameState, action2,self.farPoint)
        #pos2 = successor.getAgentPosition(self.index)
        #action3 = None
        #print'self.actionList outside if-else',self.actionList
        if len(self.actionList) == 0:
          #self.actionList.remove('Stop')
          self.actionList = self.aStar(gameState,self.farPoint,myCurrentPos,start)
          
          #self.actionList = action3
          #print'self.actionList',self.actionList
          if len(self.actionList) != 0:
            if self.actionList[0] == 9999:
              self.aSt = False
              self.skipAstar = True
              self.actionList.remove(9999)
              print'skip astar first'
        elif self.actionList[0] == 9999:
          self.aSt = False
          self.skipAstar = True
          self.actionList.remove(9999)
          print'skip a star'
        else:
          #actionList = self.actionList
          #if len(self.actionList)>0:
          bestAction = self.actionList[0]
          print 'eval time in A STAR for agent %d: %.4f' % (self.index, time.time() - start)
          print'bestAction',bestAction
            
            
          self.actionList.remove(bestAction)

          if myCurrentPos == InitialPosition or len(self.actionList) == 0:
            self.aSt = False
          else:
            #for move in actions:
            if bestAction in actions:
              return bestAction
            else:
              self.aSt = False
            #return bestAction
      else:
        self.aSt = False
          
          
          #bestDist = dist
          #print'bestAction',bestAction
      
          #return bestAction
    else:
      self.aSt = False
        

    
    
    action = self.getPolicy(gameState)
    """if self.stopAction and not self.skipAstar:
      print'stop for a sec'
      action = Directions.STOP
      self.stopAction = False"""
      
    """elif self.aSt:
      action = Directions.STOP
      self.stopAction = False
      print'stop second time'"""
    #print"final action",action
    print 'eval time in Q-Learning for agent %d: %.4f' % (self.index, time.time() - start)
    return action
  def updateQ(self,gameState,action):
     
    
    weights = self.weights
    nextState = self.getSuccessor(gameState, action)
    foodList = self.getFood(nextState).asList()

    successor = self.getSuccessor(gameState, action)
    myCurrentPos =  gameState.getAgentState(self.index).getPosition()

    enemies = []
    enemyGhost = []
    enemyPacman = []
    for opponent in self.getOpponents(gameState):
      enemy = gameState.getAgentState(opponent)
      enemies.append(enemy)
    #print"enemies", enemies
    enemyGhost = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    enemyPacman = [a for a in enemies if a.isPacman and a.getPosition() != None]
    enemyGhostPosition = [Ghost.getPosition() for Ghost in enemyGhost]
    enemyPacmanPosition = [Pacman.getPosition() for Pacman in enemyPacman]
    if len(enemyGhostPosition) >0:
      distanceToEnemyGhost = (min([self.getMazeDistance(myCurrentPos, ghostPos) for ghostPos in enemyGhostPosition]))
    else:
      distanceToEnemyGhost = 9999
      

    x, y = gameState.getAgentPosition(self.index)
    dx, dy = Actions.directionToVector(action)
    xAfterMove, yAfterMove = int(x + dx), int(y + dy)
    nextPosition = (xAfterMove,yAfterMove)

    if gameState.hasFood(xAfterMove,yAfterMove):
      wallCount = 0
      if gameState.hasWall(xAfterMove+1,yAfterMove):
        #print'right wall at reward'
        wallCount +=1
      if gameState.hasWall(xAfterMove-1,yAfterMove):
        #print'left wall at reward'
        wallCount += 1
      if gameState.hasWall(xAfterMove,yAfterMove+1):
        #print'top wall at reward'
        wallCount += 1
      if gameState.hasWall(xAfterMove,yAfterMove-1):
        #print'bottom wall at reward'
        wallCount += 1
      if wallCount>=3 and distanceToEnemyGhost <=2:
        reward = -1
      else:
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
    #print'action infeatures',action
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
    #print'border',border
    #print'myposition',myPosition

    distanceToBorder = min([self.getMazeDistance(myPosition,borderPos) for borderPos in border])


    otherFood = []
    for nFood in foodList:
      wallCount1 = 0
      foodX, foodY = nFood
      if gameState.hasWall(foodX+1,foodY):
        #print'right wall'
        wallCount1 +=1
      if gameState.hasWall(foodX-1,foodY):
        #print'left wall'
        wallCount1 += 1
      if gameState.hasWall(foodX,foodY+1):
        #print'top wall'
        wallCount1 += 1
      if gameState.hasWall(foodX,foodY-1):
        #print'bottom wall'
        wallCount1 += 1
      if wallCount1<3:


      
        otherFood.append(nFood)



    #print"foodList",foodList
    capsules = gameState.getCapsules()
    defendingCap = self.getCapsulesYouAreDefending(gameState)
    for defCap in defendingCap:
      for cap in capsules:
        if defCap == cap:
          tempCap = defCap
      capsules.remove(tempCap)
      #capsules.remove(defCap)
    #print'capsules',capsules
    
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

    currentX, currentY = myPosition

    
    ghostPositions = []
    enemiesInvisible = False

    ranges = []
    enemyGhostPosition = [Ghost.getPosition() for Ghost in enemyGhost]
    enemyPacmanPosition = [Pacman.getPosition() for Pacman in enemyPacman]

    eatMoreDot = False
    dontEatSecondCap = False
    isScared = False
    escapeDistance = 9999

    #features['getFood'] = 1.0
    #features['back'] = -1.1
    if len(enemyGhostPosition) > 0:
      escapeDistance = min([self.getMazeDistance(successor.getAgentPosition(self.index),ghostPosition) for ghostPosition in enemyGhostPosition]) 
      #print"empty"
      #print"enemyGhostPosition",enemyGhostPosition
      enemiesInvisible = True
    
    if escapeDistance <10 or successor.getAgentPosition(self.index) == InitialPosition:
      #print"escapeDistance",escapeDistance
      # enemy in range
      #scaredTime = 0
      for ghostNearby in enemyGhost:
        if ghostNearby.scaredTimer >0:
          dontEatSecondCap = True
          
        if ghostNearby.scaredTimer > 8:
          eatMoreDot = True
          
        
        if ghostNearby.scaredTimer > 30:
          isScared = True
          #scaredTime = ghostNearby.scaredTimer
        else:
          isScared = False
      #print'scare time', scaredTime
      if isScared:
        #startEatingGhost = time.time()
        distanceToGhost = min([self.getMazeDistance(successor.getAgentPosition(self.index),ghostPosition) for ghostPosition in enemyGhostPosition])
        #print'eat ghost'
        features['eatGhost'] = float(distanceToGhost*10)/(walls.width * walls.height)
        #features['back'] = 1.0
      else:
        
        if gameState.getAgentState(self.index).isPacman and not eatMoreDot:
          
          
            if len(capsules) >0 and not dontEatSecondCap:
              
              #print'cap',defendingCap
              distanceToCapsules = min(self.getMazeDistance(successor.getAgentPosition(self.index),capsule) for capsule in capsules)
              #if successor.getAgentPosition(self.index) == capsules:#successor
              features['getCaplual'] = float(distanceToCapsules*10)/(walls.width * walls.height)
            
            
            #print'testing detour'
            if currentCarry != 0:
                print'back'
                print'carrying dot', currentCarry
                #features['back'] = -float(self.getMazeDistance(myPosition, distanceToBorder))/(walls.width * walls.height)
                features['back'] = -float(distanceToBorder)/(walls.width * walls.height)
            
            distanceToEnemyGhost = (min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition]))
            if  distanceToEnemyGhost ==1:
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
                #features['enemyOneStepToPacman'] = 0
               
                
                print'detour to get food'
              else:
                #if (min([self.getMazeDistance(myCurrentPos, ghostPos) for ghostPos in enemyGhostPosition])) ==1:
                print'back when one step away'
                features['reverse'] = 1
               
            if len(foodList) > 0: # This should always be True,  but better safe than sorry      
              minFoodDistance = min([(self.getMazeDistance(myPosition, food),food) for food in foodList])
            if len(minFoodDistance)>0:
                
              if minFoodDistance[0] <4 and minFoodDistance[1] not in otherFood  and distanceToEnemyGhost <=2 and action != Directions.STOP :
                features['back'] = 0
                #print'food distance',minFoodDistance[0]
                wallCount = 0
              
                  
                foodX, foodY = minFoodDistance[1]
                if gameState.hasWall(foodX+1,foodY):
                  #print'right wall'
                  wallCount +=1
                if gameState.hasWall(foodX-1,foodY):
                  #print'left wall'
                  wallCount += 1
                if gameState.hasWall(foodX,foodY+1):
                  #print'top wall'
                  wallCount += 1
                if gameState.hasWall(foodX,foodY-1):
                  #print'bottom wall'
                  wallCount += 1
                if wallCount>=3:
                  #print'FOOD IN CORNER'
                  #print'otherFood',otherFood
                  #print'my position when other food', myPosition
                  newFoodX, newFoodY = minFoodDistance[1]
                  


                  if gameState.hasFood(newFoodX+1,newFoodY):
                    dangerFood0 =(newFoodX+1,newFoodY)
                    #print'dangerFood0',dangerFood0
                    if dangerFood0 in otherFood:
                      otherFood.remove(dangerFood0)
                  
                  if gameState.hasFood(newFoodX-1,newFoodY):
                    dangerFood1 =(newFoodX-1,newFoodY)
                    #print'dangerFood1',dangerFood1
                    if dangerFood1 in otherFood:
                      otherFood.remove(dangerFood1)
                  if gameState.hasFood(newFoodX,newFoodY+1):
                    dangerFood2 =(newFoodX,newFoodY+1)
                    #print'dangerFood2',dangerFood2
                    if dangerFood2 in otherFood:
                      otherFood.remove(dangerFood2)
                    
                  if gameState.hasFood(newFoodX,newFoodY-1):
                    dangerFood3 =(newFoodX,foodY-1)
                    #print'dangerFood3',dangerFood3
                    if dangerFood3 in otherFood:
                      otherFood.remove(dangerFood3)
                  
                  
                  #otherFood = foodList
                  #foodList.remove(minFoodDistance[1])
                  
                  #otherFood.remove(minFoodDistance[1])
                  #print'foodList',foodList
                  #print'otherFood',otherFood
                  
                  if len(otherFood) >0:
                    minOtherFoodDistance = min([(self.getMazeDistance(myPosition, food),food) for food in otherFood])
                  
                    features['carrying'] = float(minOtherFoodDistance[0])/(walls.width * walls.height)

                    if  distanceToEnemyGhost >=2:
                      features['enemyOneStepToPacman'] = 0
                      features['reverse'] = 0

                    print'second dot',minOtherFoodDistance[1]
                  
                    #features['distanceToFood'] = float(minOtherFoodDistance[0])/(walls.width * walls.height)
                    features['distanceToFood'] = 0
                  
                    
                

                      
                    if gameState.isOnRedTeam(self.index):
        
                      #print"bluefood",blueFood
                      blueFood.remove(minFoodDistance[1])
                      
        
                      if len(blueFood) != 0:
                        features['successorScore'] = -float(len(otherFood))/len(blueFood)
                    else:
                      redFood.remove(minFoodDistance[1])
                      if len(redFood) != 0:
                        features['successorScore'] = -float(len(otherFood))/len(redFood)
                    #features['successorScore'] = 0"""
                      
                    
                            
        else:
          #print'run from my side'
          if eatMoreDot:
            print'eat'
            
          else:
            if len(enemyGhostPosition) >0:
          
              if min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition]) <3:
                #features['back'] = -float(self.getMazeDistance(myPosition, distanceToBorder))/(walls.width * walls.height)
                features['back'] = -float(distanceToBorder)/(walls.width * walls.height)
                
                print'close'
              
                #if min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition]) == 1:
                #print'enter dont run'
                
                print'run when cap almost gone'
                features['enemyOneStepToPacman'] = -float(min(self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition))/(walls.width * walls.height)
                rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
                if action == rev:
                  features['reverse'] = 1
          
         
    if not eatMoreDot:
      if len(foodList) <=2 or currentCarry >= 6:
        features['distanceToFood'] = 0
        features['successorScore'] = 0
        print'head home'
        isScared1 = False
        #features['back'] = -float(self.getMazeDistance(myPosition, distanceToBorder))/(walls.width * walls.height)
        #print'distance to boorder',distanceToBorder
        features['back'] = -float(distanceToBorder)/(walls.width * walls.height)

        
        #features['stop'] = -1
        if len(enemyGhostPosition) != 0 and not dontEatSecondCap:
          if (min([self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition]))==1:
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
          """if (min([self.getMazeDistance(myCurrentPos, ghostPos) for ghostPos in enemyGhostPosition])) ==1:
            #features['enemyOneStepToPacman'] = -float(min(self.getMazeDistance(myPosition, ghostPos) for ghostPos in enemyGhostPosition))/(walls.width * walls.height)
            rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if action == rev:
              print'testing escape'
              features['reverse'] = 1"""
            
            
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
  def aStar(self,gameState,goal,iniPos,start):
    
    legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    

    mid = gameState.data.layout.width / 2

    if gameState.isOnRedTeam(self.index):
      mid = mid - 1
    else:
      mid = mid + 1

    legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
    border = [p for p in legalPositions if p[0] == mid]

    if gameState.isOnRedTeam(self.index):
      side = [p for p in legalPositions if p[0] <= mid]
    else:
      side = [p for p in legalPositions if p[0] >= mid]
    if iniPos not in side:
      side.append(iniPos)

    #print'side', side
    #print'border',border
    #successor = self.getSuccessor(gameState, action)
    enemies = []
    enemyGhost = []
    enemyPacman = []
    
    for opponent in self.getOpponents(gameState):
      enemy = gameState.getAgentState(opponent)
      enemies.append(enemy)
    #print"enemies", enemies
    enemyGhost = [a for a in enemies if not a.isPacman and a.getPosition() != None]

    enemyGhostPosition = [Ghost.getPosition() for Ghost in enemyGhost]
    myPosition = gameState.getAgentState(self.index).getPosition()
    escape = True
    Cost = []
    priority = 0
    frontier = util.PriorityQueue()
    Action = []
    successors = []
    ss = []
    newPos = []
    ss.append(gameState)
    #frontier.push((gameState.getAgentPosition(self.index),Action,Cost),priority)
    frontier.push((iniPos,Action,Cost),priority)
    visited = []
    #visited[problem.getStartState()] = True
    actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

    #distanceToCloseGhost = min(self.getMazeDistance(successor.getAgentPosition(self.index),ghostPosition) for ghostPosition in enemyGhostPosition)
            
    currentAction = {}
    costOfAction = 0
    
    while not frontier.isEmpty():
        '''priority = 0'''
        '''print "isEmpty", frontier.isEmpty()'''
        current = frontier.pop()
        #print"current in astar", current
        #currentState = current[0].getAgentPosition(self.index)
        currentState = current[0]
        print "current", currentState
        currentAction = current[1]
        
        '''print "currentAction0", currentAction'''
        '''currentAction[current[1]] = current[1]'''
        currentCost = current[2]
        
        #print "printCost", currentCost
        #print " currentAction", currentAction 
        #for ghostPosition in enemyGhostPosition:
        '''if len(enemyGhostPosition) !=0:
          escape = False;'''
        #print'currentState',currentState
        #print'goal',goal
        
        print 'eval time in A STAR action for agent %d: %.4f' % (self.index, time.time() - start)
        aStarTime = time.time() -start
        if aStarTime >=0.8:
          return [9999]
        if (currentState == goal):
                #print"return len of action" 
                '''print "next[1]", next[1]'''
                #print "currentCost", currentAction
                #self.expanded = visited
                #self.aSt = False
                return currentAction
            
        if (currentState not in visited) and (currentState in side) and (currentState in legalPositions):
            #visited[currentState] = True
            #actions = gameState.getLegalActions(self.index)
            #print'actions',actions
            for action in actions:
              
              
              #successors.append((newState,action))
              x,y = currentState
             
              #print'xy',x,y
              dx, dy = Actions.directionToVector(action)
              #print'dxdy',dx,dy
              nextx, nexty = int(x + dx), int(y + dy)
              #successor =  Configuration.generateSuccessor( gameState.getAgentState(self.index).configuration,(xAfterV,yAfterV)) 
              
                
              #successors.append((successor,action))
              #ss.append(nextSuc)
              #print'successor',successor
              
            #for suce in successors:
              #print'next',next[0]
              #nextAction = suce[1]
              #for next in successors:
              
              #position = successor.getPosition()
              position = (nextx,nexty)
              #print'position',position
              #nextAction = next[1]
              if (position in side) and (position in legalPositions):
                newAction = currentAction+[action]
                nextAction = action
                costOfAction = 1
                newPos.append((position,nextAction,costOfAction))
                
              else:
                continue
            for next in newPos:
                
              if (next[0] not in visited):
                
                
                  '''print "next in a star", next'''
                  
                  ''' print "next2", next[2]'''
                    
                  #print "currentCost", currentCost
                  #currentCost.append(len(currentAction))
                  heuristic = self.getMazeDistance(next[0], goal)
                  #heuristic = 1
                  
                  priority = heuristic
                  #priority = problem.getCostOfActions(currentAction+[next[1]]) + heuristic(next[0],problem)                 
                  frontier.push((next[0],currentAction+[next[1]],currentCost+[next[2]]),priority)
        #visited[currentState] = True#if put inside of the loop last problem wrong
        visited.append(currentState)
      
    return [9999]
  

  




  
  # Update weights file at the end of each game
  """def final(self, gameState):
    #print self.weights
    file = open('weights.txt', 'w')
    file.write(str(self.weights))"""
    

  '''def getWeights(self, gameState, action):
    return {'successorScore': 10000, 'distanceToFood': -1}'''

class DefensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index):
        CaptureAgent.__init__(self, index)
        # Variables used to verify if the agent os locked
        self.foodNum = 999
        self.trappedTime = 0
        self.defendingFood=[]
        self.flag=0
        self.target=()
        self.isTargetToFood = False
    def getBorder(self,gameState):

      mid = gameState.data.layout.width/2

      if self.red:
        mid = mid - 1
      else:
        mid = mid + 1

      legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
      border = [p for p in legalPositions if p[0] == mid]
      return border

    
    def aStar(self,gameState,goal,iniPos,start):
    
        legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        

        mid = gameState.data.layout.width / 2

        if gameState.isOnRedTeam(self.index):
          mid = mid - 1
        else:
          mid = mid + 1

        legalPositions = [p for p in gameState.getWalls().asList(False) if p[1] > 1]
        border = [p for p in legalPositions if p[0] == mid]

        if gameState.isOnRedTeam(self.index):
          side = [p for p in legalPositions if p[0] <= mid]
        else:
          side = [p for p in legalPositions if p[0] >= mid]
        if iniPos not in side:
          side.append(iniPos)

        enemies = []
        enemyGhost = []
        enemyPacman = []
        
        for opponent in self.getOpponents(gameState):
          enemy = gameState.getAgentState(opponent)
          enemies.append(enemy)
        #print"enemies", enemies
        enemyGhost = [a for a in enemies if not a.isPacman and a.getPosition() != None]

        enemyGhostPosition = [Ghost.getPosition() for Ghost in enemyGhost]
        myPosition = gameState.getAgentState(self.index).getPosition()
        escape = True
        Cost = []
        priority = 0
        frontier = util.PriorityQueue()
        Action = []
        successors = []
        ss = []
        newPos = []
        ss.append(gameState)
        #frontier.push((gameState.getAgentPosition(self.index),Action,Cost),priority)
        frontier.push((iniPos,Action,Cost),priority)
        visited = []
        #visited[problem.getStartState()] = True
        actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

        #distanceToCloseGhost = min(self.getMazeDistance(successor.getAgentPosition(self.index),ghostPosition) for ghostPosition in enemyGhostPosition)
                
        currentAction = {}
        costOfAction = 0
        
        while not frontier.isEmpty():
           
            current = frontier.pop()
            currentState = current[0]
            print "current", currentState
            currentAction = current[1]
          
            currentCost = current[2]
           
            print 'eval time in A STAR action for agent %d: %.4f' % (self.index, time.time() - start)
            aStarTime = time.time() -start
            if aStarTime >=0.8:
                #if len(currentAction)!=0:
                return [9999]
            #else:
            #return [9999]
            if (currentState == goal):
                    
                    return currentAction
                
            if (currentState not in visited) and (currentState in side) and (currentState in legalPositions):
                
                for action in actions:
                  
                  x,y = currentState
                 
                  #print'xy',x,y
                  dx, dy = Actions.directionToVector(action)
                  #print'dxdy',dx,dy
                  nextx, nexty = int(x + dx), int(y + dy)
                  
                  position = (nextx,nexty)
                  
                  if (position in side) and (position in legalPositions):
                    newAction = currentAction+[action]
                    nextAction = action
                    costOfAction = 1
                    newPos.append((position,nextAction,costOfAction))
                    
                  else:
                    continue
                for next in newPos:
                    
                  if (next[0] not in visited):
                      heuristic = self.getMazeDistance(next[0], goal)
                      #heuristic = 1
                      
                      priority = heuristic
                      #priority = problem.getCostOfActions(currentAction+[next[1]]) + heuristic(next[0],problem)                 
                      frontier.push((next[0],currentAction+[next[1]],currentCost+[next[2]]),priority)
            #visited[currentState] = True#if put inside of the loop last problem wrong
            visited.append(currentState)
          
        return [9999]
      
    def getFeatures(self, gameState, action):
      features = util.Counter()
      successor = self.getSuccessor(gameState, action)
      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()
      myCuPos = gameState.getAgentState(self.index).getPosition()
      myFood=self.getFoodYouAreDefending(gameState).asList()
      borderPos=random.choice(self.getBorder(gameState))
     
      
      

      if self.flag==0:
        self.flag=1
        self.defendingFood=myFood
        self.target=borderPos

      #locate the food area where the pacdots disapeared
      if len(self.defendingFood)>len(myFood):

        #print "protect food!!!!!!!!!!!!!!!!!!!"
        target=list(set(self.defendingFood)-set(myFood))[0]
        self.defendingFood=myFood
        self.target=target
        self.isTargetToFood = True
      
      #when enemy lose pacdots
      if len(self.defendingFood)<len(myFood):
        #print "renew the defending food"
        self.defendingFood = myFood
      
      
      if myState.isPacman:
       print "ispacman"
       features['onDefense'] = 0
      else:
       features['onDefense'] = 1

      '''for b in self.getBorder(gameState):
        if self.red:
          if myPos[0] <= b[0]:
            features['behindBorderLine'] = 1
          else:
            features['behindBorderLine'] = 0
        else:
          if myPos[0] >= b[0]:
            features['behindBorderLine'] = 1
          else:
            features['behindBorderLine'] = 0'''
            
      opponents = [successor.getAgentState(i) for i in self.getOpponents(successor)]
      nearOpponents = [i for i in opponents if (not i.isPacman) and (i.getPosition() != None)]
      invaders = [i for i in opponents if i.isPacman and i.getPosition() != None]
      features['numInvaders'] = len(invaders)
      
      
        
      if len(invaders) > 0:
        invaderDist = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]

        invaderDistClose = [self.getMazeDistance(myCuPos, a.getPosition()) for a in invaders]
        
        #follow the invader
        features['invaderDist'] = min(invaderDist)
        
        if min(invaderDist)==1 and gameState.getAgentState(self.index).scaredTimer > 0:
          features['onDefense'] = 0
          reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
          if action ==reverse:
            features['reverse'] = 1
          
            
          
        if min(invaderDistClose)==1 and gameState.getAgentState(self.index).scaredTimer > 0:
          features['onDefense'] = 0
          reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
          if action ==reverse:
            features['reverse'] = 1
            
          
        self.target=borderPos
        self.isTargetToFood=False
        
        


      if len(invaders) == 0:
        #when there is no invader and no dot disapears, stay close to the opnt
        if len(nearOpponents) > 0 and self.isTargetToFood==False:
          #print "stick to opponent"
          nearDist = [self.getMazeDistance(myPos, a.getPosition()) for a in nearOpponents]
          features['nearDist'] = min(nearDist)

        else:
          #protect food or patrol around border
          if self.isTargetToFood == True and len(nearOpponents) > 0:
            #print "here[1]"
            features['onDefense'] = 0
            reverse = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
            if action ==reverse:
              features['reverse'] = 1
          #print "patrol or protect"
          features['ToFoodOrBorder'] = self.getMazeDistance(myPos,self.target)
      else:
        features['ToFoodOrBorder'] = 0
    
      return features



    def getWeights(self, gameState, action):
      return { 'onDefense': 100, 'invaderDist': -10, 'nearDist':-10, 'opponentDist':-10, 'reverse': -10,
              'ToFoodOrBorder': -20,'numInvaders':-1000}
    def chooseAction(self, gameState):
        """
    Picks among the actions with the highest Q(s,a).
    """
        actions = gameState.getLegalActions(self.index)
        clock = time.time()
        myCurrentPos =  gameState.getAgentState(self.index).getPosition()
        InitialPosition = gameState.getInitialAgentPosition(self.index)
        enemies = []
        enemyGhost = []
        enemyPacman = []
        for opponent in self.getOpponents(gameState):
          enemy = gameState.getAgentState(opponent)
          enemies.append(enemy)
        #print"enemies", enemies
        enemyGhost = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        enemyPacman = [a for a in enemies if a.isPacman and a.getPosition() != None]

      
        #remove the stop action
        actions.remove(Directions.STOP)

        #remove the pacman action under some situations
        if gameState.getAgentState(self.index).scaredTimer <= 0 and self.isTargetToFood == False:
            for a in actions:
              successor = self.getSuccessor(gameState, a)
              myState = successor.getAgentState(self.index)
              if myState.isPacman:
                  actions.remove(a)

        #use astar to find the disapeared dot
        actionList=[]
        if self.isTargetToFood == True and len(enemyPacman)==0:
            actionList = self.aStar(gameState, self.target, myCurrentPos, clock)

            print "actionList:",actionList
            if len(actionList) >0:
              if actionList[0]!=9999:
                  return actionList[0]
            
                      

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




