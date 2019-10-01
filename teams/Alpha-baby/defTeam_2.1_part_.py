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
        
        if min(invaderaDist)==1 and gameState.getAgentState(self.index).scaredTimer > 0:
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





