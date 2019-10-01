# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
import game
from game import Grid
import distanceCalculator
import random, time, util, sys
from game import Directions,Actions
from util import nearestPoint


#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first='DefensiveReflexAgent', second='OffensiveReflexAgent'):
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

    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

    def registerInitialState(self, gameState):
        """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

        '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

        '''
    Your initialization code goes here, if you need any.
    '''

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
        self.isTargetToFood = False

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
        
      
        #remove the stop action
        actions.remove(Directions.STOP)

        if gameState.getAgentState(self.index).scaredTimer <= 0 and self.isTargetToFood == False:
            for a in actions:
              successor = self.getSuccessor(gameState, a)
              myState = successor.getAgentState(self.index)
              if myState.isPacman:
                  actions.remove(a)

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
