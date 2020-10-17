# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        from util import manhattanDistance

        food = newFood.asList()
        foodDistances = []
        ghostDistances = []
        count = 0 

        for item in food:
            foodDistances.append(manhattanDistance(newPos,item))

        for i in foodDistances:
            if i <= 4:
                count += 1
            elif i > 4 and i <= 15:
                count += 0.2
            else:
                count += 0.15

        for ghost in successorGameState.getGhostPositions():
            ghostDistances.append(manhattanDistance(ghost,newPos))

        for ghost in successorGameState.getGhostPositions():
            if ghost == newPos:
                count = 2 - count

            elif manhattanDistance(ghost,newPos) <= 3.5:
                count = 1 - count

        return successorGameState.getScore() + count

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        def miniMax(gameState,agent,depth):
            result = []

            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState),0

            if depth == self.depth:
                return self.evaluationFunction(gameState),0

            if agent == gameState.getNumAgents() - 1:
                depth += 1

            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index
            else:
                nextAgent = agent + 1

            for action in gameState.getLegalActions(agent):
                if not result:
                    nextValue = miniMax(gameState.generateSuccessor(agent,action),nextAgent,depth)

                    result.append(nextValue[0])
                    result.append(action)
                else:
                    previousValue = result[0]
                    nextValue = miniMax(gameState.generateSuccessor(agent,action),nextAgent,depth)
                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                    else:
                        if nextValue[0] < previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
            return result

        return miniMax(gameState,self.index,0)[1]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def AB(gameState,agent,depth,a,b):
            result = []

            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState),0

            if depth == self.depth:
                return self.evaluationFunction(gameState),0

            if agent == gameState.getNumAgents() - 1:
                depth += 1

            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index

            else:
                nextAgent = agent + 1

            for action in gameState.getLegalActions(agent):
                if not result: 
                    nextValue = AB(gameState.generateSuccessor(agent,action),nextAgent,depth,a,b)
                    result.append(nextValue[0])
                    result.append(action)

                    if agent == self.index:
                        a = max(result[0],a)
                    else:
                        b = min(result[0],b)
                else:
                    if result[0] > b and agent == self.index:
                        return result

                    if result[0] < a and agent != self.index:
                        return result

                    previousValue = result[0]
                    nextValue = AB(gameState.generateSuccessor(agent,action),nextAgent,depth,a,b)

                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            a = max(result[0],a)

                    else:
                        if nextValue[0] < previousValue:
                            result[0] = nextValue[0]
                            result[1] = action
                            # b may change #
                            b = min(result[0],b)
            return result

        return AB(gameState,self.index,0,-float("inf"),float("inf"))[1]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectiMax(gameState,agent,depth):
            result = []

            if not gameState.getLegalActions(agent):
                return self.evaluationFunction(gameState),0

            if depth == self.depth:
                return self.evaluationFunction(gameState),0

            if agent == gameState.getNumAgents() - 1:
                depth += 1

            if agent == gameState.getNumAgents() - 1:
                nextAgent = self.index

            else:
                nextAgent = agent + 1

            for action in gameState.getLegalActions(agent):
                if not result: # First move
                    nextValue = expectiMax(gameState.generateSuccessor(agent,action),nextAgent,depth)
                    if(agent != self.index):
                        result.append((1.0 / len(gameState.getLegalActions(agent))) * nextValue[0])
                        result.append(action)
                    else:
                        result.append(nextValue[0])
                        result.append(action)
                else:
                    previousValue = result[0]
                    nextValue = expectiMax(gameState.generateSuccessor(agent,action),nextAgent,depth)

                    if agent == self.index:
                        if nextValue[0] > previousValue:
                            result[0] = nextValue[0]
                            result[1] = action

                    else:
                        result[0] = result[0] + (1.0 / len(gameState.getLegalActions(agent))) * nextValue[0]
                        result[1] = action
            return result

        return expectiMax(gameState,self.index,0)[1]

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    food = currentGameState.getFood().asList()
    ghosts = currentGameState.getGhostStates()
    pacmanPosition = currentGameState.getPacmanPosition()
    activeGhosts = []
    scaredGhosts = []
    totalCapsules = len(currentGameState.getCapsules())
    totalFood = len(food)
    myEval = 0 

    for ghost in ghosts:
        if ghost.scaredTimer:
            scaredGhosts.append(ghost)
        else:
            activeGhosts.append(ghost)

    myEval += 1.5 * currentGameState.getScore()

    myEval += -10 * totalFood

    myEval += -20 * totalCapsules

    foodDistances = []
    activeGhostsDistances = []
    scaredGhostsDistances = []

    for item in food:
        foodDistances.append(manhattanDistance(pacmanPosition,item))

    for item in activeGhosts:
        scaredGhostsDistances.append(manhattanDistance(pacmanPosition,item.getPosition()))

    for item in scaredGhosts:
        scaredGhostsDistances.append(manhattanDistance(pacmanPosition,item.getPosition()))

    for item in foodDistances:
        if item < 3:
            myEval += -1 * item
        if item < 7:
            myEval += -0.5 * item
        else:
            myEval += -0.2 * item

    for item in scaredGhostsDistances:
        if item < 3:
            myEval += -20 * item
        else:
            myEval += -10 * item

    for item in activeGhostsDistances:
        if item < 3:
            myEval += 3 * item
        elif item < 7:
            myEval += 2 * item
        else:
            myEval += 0.5 * item

    return myEval

# Abbreviation
better = betterEvaluationFunction

