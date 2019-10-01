# myTeam.py
"""
The author of agents is Yufei Gao
This is the 3rd version for agents using expectiNegamax search, which is a variant
of minmax algorithm.

"""
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
# Pieter Abbeel (pabbeel@cs.berkeley.edu).# myteam.py


from captureAgents import CaptureAgent
from capture import SIGHT_RANGE
import random, time, util,sys
import game
import distanceCalculator
from game import Directions, Actions
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent',
               second = 'DefensiveReflexAgent'):
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


class ReflexCaptureAgent(CaptureAgent):


  SEARCH_DEPTH = 5
  TERMINAL_STATE_VALUE = -1000000

  def registerInitialState(self, gameState):
    if self.red:
      CaptureAgent.registerTeam(self, gameState.getRedTeamIndices())
    else:
      CaptureAgent.registerTeam(self, gameState.getBlueTeamIndices())
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

    self.legalPositions = gameState.data.layout.walls.asList(False)

    self.positionBeliefs = {}
    for opponent in self.getOpponents(gameState):
      self.initialBeliefsDistributions(opponent)

  def stateIsTerminal(self, agent, gameState):

    return len(gameState.getLegalActions(agent)) == 0

  def getAgentPosition(self, agent, gameState):

    pos = gameState.getAgentPosition(agent)
    if pos:
      return pos
    else:
      return self.guessPosition(agent)

  def agentIsPacman(self, agent, gameState):

    agentPos = self.getAgentPosition(agent, gameState)
    return (gameState.isRed(agentPos) != gameState.isOnRedTeam(agent))

  def getOpponentDistances(self, gameState):

    return [(a, self.distancer.getDistance(
             self.getAgentPosition(self.index, gameState),
             self.getAgentPosition(a, gameState)))
            for a in self.getOpponents(gameState)]

  def initialBeliefsDistributions(self, agent):

    self.positionBeliefs[agent] = util.Counter()
    for p in self.legalPositions:
      self.positionBeliefs[agent][p] = 1.0

  def chooseAction(self, gameState):

    myPosition = gameState.getAgentState(self.index).getPosition()
    noisyDistances = gameState.getAgentDistances()
    probableState = gameState.deepCopy()

    for opponent in self.getOpponents(gameState):
      pos = gameState.getAgentPosition(opponent)
      if pos:
        self.enemyFixedPosition(opponent, pos)
      else:
        self.elapseTime(opponent, gameState)
        self.observe(opponent, noisyDistances[opponent], gameState)

    self.displayDistributionsOverPositions(self.positionBeliefs.values())
    for opponent in self.getOpponents(gameState):
      probablePosition = self.guessPosition(opponent)
      c = game.Configuration(probablePosition, Directions.STOP)
      probableState.data.agentStates[opponent] = game.AgentState(
        c, probableState.isRed(probablePosition) != probableState.isOnRedTeam(opponent))

    bestValue, bestAction = float("-inf"), None
    for opponent in self.getOpponents(gameState):
      value, action = self.expectNegamax(opponent,probableState,self.SEARCH_DEPTH,1,returnAction=True)
      if value > bestValue:
        bestValue, bestAction = value, action

    return action

  def enemyFixedPosition(self, agent, position):
    newBeliefs = util.Counter()
    newBeliefs[position] = 1.0
    self.positionBeliefs[agent] = newBeliefs

  def elapseTime(self, agent, gameState):

    newBeliefs = util.Counter()
    for (preX, preY), preProbability in self.positionBeliefs[agent].items():
      newDist = util.Counter()
      for p in [(preX - 1, preY), (preX + 1, preY),
                (preX, preY - 1), (preX, preY + 1)]:
        if p in self.legalPositions:
          newDist[p] = 1.0
      newDist.normalize()
      for newPosition, newProbability in newDist.items():
        newBeliefs[newPosition] += newProbability * preProbability

    lastObserved = self.getPreviousObservation()
    if lastObserved:
      lostFood = [food for food in self.getFoodYouAreDefending(lastObserved).asList()
                  if food not in self.getFoodYouAreDefending(gameState).asList()]
      for f in lostFood:
        newBeliefs[f] = 1.0/len(self.getOpponents(gameState))

    self.positionBeliefs[agent] = newBeliefs


  def observe(self, agent, noisyDistance, gameState):
    myPosition = self.getAgentPosition(self.index, gameState)
    partnerPositions = [self.getAgentPosition(partner, gameState)
                         for partner in self.getTeam(gameState)]
    newBeliefs = util.Counter()

    for p in self.legalPositions:
      if any([util.manhattanDistance(partnerPos, p) <= SIGHT_RANGE
              for partnerPos in partnerPositions]):
        newBeliefs[p] = 0.0
      else:
        trueDistance = util.manhattanDistance(myPosition, p)
        positionProbability = gameState.getDistanceProb(trueDistance, noisyDistance)
        newBeliefs[p] = positionProbability * self.positionBeliefs[agent][p]

    if not newBeliefs.totalCount():
      self.initialBeliefsDistributions(agent)
    else:
      newBeliefs.normalize()
      self.positionBeliefs[agent] = newBeliefs

  def guessPosition(self, agent):

    return self.positionBeliefs[agent].argMax()
  def evaluate(self, gameState):

    util.raiseNotDefined()

  def expectNegamax(self, opponent, state, depth, mark, returnAction=False):
    if mark == 1:
      agent = self.index
    else:
      agent = opponent

    bestAction = None
    if self.stateIsTerminal(agent, state) or depth == 0:
      bestValue = mark * self.evaluate(state)
    else:
      actions = state.getLegalActions(agent)
      actions.remove(Directions.STOP)
      bestValue = float("-inf") if agent == self.index else 0
      for action in actions:
        successor = state.generateSuccessor(agent, action)
        value = -self.expectNegamax(opponent, successor, depth - 1, -mark)
        if agent == self.index and value > bestValue:
          bestValue, bestAction = value, action
        elif agent == opponent:
          bestValue += value/len(actions)

    if agent == self.index and returnAction:
      return bestValue, bestAction
    else:
      return bestValue


class OffensiveReflexAgent(ReflexCaptureAgent):
  def evaluate(self, gameState):
    myPosition = self.getAgentPosition(self.index, gameState)
    food = self.getFood(gameState).asList()

    targetFood = None
    maxDist = 0

    opponentDistances = self.getOpponentDistances(gameState)
    opponentDistance = min([dist for id, dist in opponentDistances])

    if not food or gameState.getAgentState(self.index).numCarrying > self.getScore(gameState) > 0:
      return 20 * self.getScore(gameState) \
             - self.distancer.getDistance(myPosition, gameState.getInitialAgentPosition(self.index)) \
             + opponentDistance

    for f in food:
      d = min([self.distancer.getDistance(self.getAgentPosition(o, gameState), f)
              for o in self.getOpponents(gameState)])
      if d > maxDist:
        targetFood = f
        maxDist = d
    if targetFood:
      foodDist = self.distancer.getDistance(myPosition, targetFood)
    else:
      foodDist = 0

    distanceFromStart = abs(myPosition[0] - gameState.getInitialAgentPosition(self.index)[0])
    if not len(food):
      distanceFromStart *= -1

    return 2 * self.getScore(gameState) \
           - 100 * len(food) \
           - 2 * foodDist \
           + opponentDistance \
           + distanceFromStart

  

class DefensiveReflexAgent(ReflexCaptureAgent):

  def stateIsTerminal(self, agent, gameState):
    return self.agentIsPacman(self.index, gameState) or \
      ReflexCaptureAgent.stateIsTerminal(self, agent, gameState)

  def evaluate(self, gameState):
    if self.agentIsPacman(self.index, gameState):
      return ReflexCaptureAgent.TERMINAL_STATE_VALUE

    myPosition = self.getAgentPosition(self.index, gameState)
    shieldedFood = self.getFoodYouAreDefending(gameState).asList()
    opponentPositions = [self.getAgentPosition(opponent, gameState)
                         for opponent in self.getOpponents(gameState)]
    if len(shieldedFood):
      opponentDistances = util.Counter()
      opponentTotalDistances = util.Counter()

      for f in shieldedFood:
        for o in opponentPositions:
          distance = self.distancer.getDistance(f, o)
          opponentDistances[(f, o)] = distance
          opponentTotalDistances[o] -= distance

      threateningOpponent = opponentTotalDistances.argMax()
      atRiskFood, shortestDist = None, float("inf")
      for (food, opponent), dist in opponentDistances.iteritems():
        if opponent == threateningOpponent and dist < shortestDist:
          atRiskFood, shortestDist = food, dist

      return len(shieldedFood) \
             - 2 * self.distancer.getDistance(myPosition, atRiskFood) \
             - self.distancer.getDistance(myPosition, threateningOpponent)
    else:
      return -min(self.getOpponentDistances(gameState), key=lambda t: t[1])[1]
