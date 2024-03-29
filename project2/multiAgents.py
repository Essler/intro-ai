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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        walls = successorGameState.getWalls()
        if newPos in walls:
            return 999999

        closestFoodDistance = 999999
        for x in range(0, newFood.width):
            for y in range(0, newFood.height):
                if newFood[x][y] and newPos != (x, y):
                    foodDistance = manhattanDistance(newPos, (x, y))
                    if foodDistance < closestFoodDistance:
                        closestFoodDistance = foodDistance

        averageGhostDistance = 0
        for ghostState in newGhostStates:
            ghostDistance = manhattanDistance(newPos, ghostState.configuration.pos)
            if ghostState.scaredTimer > 0:
                # ghostDistance = 0  # If ghost is scared, being close is OK
                ghostDistance = -ghostDistance
            averageGhostDistance += ghostDistance  # Keep a running total
        averageGhostDistance /= len(newGhostStates)  # Take the actual average

        score = successorGameState.getScore()

        scoreWt = 1/10
        avgGhostDistWt = 1/100
        rval = (1/closestFoodDistance) + (score * scoreWt) + (averageGhostDistance * avgGhostDistWt)
        return rval


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


def printDataStructure(dataStructure):
    """
    Print a Stack, Queue, or PriorityQueue from util.
    """
    for item in dataStructure.list:
        print(item)


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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def value(state, agentIndex, firstAction, depth=0):
            # If state is terminal state, return state's utility
            if state.isWin() or state.isLose() or depth >= (k * self.depth):
                return self.evaluationFunction(state), firstAction

            # Maximize for agent 0 (pacman), minimize for all others (ghosts)
            if agentIndex == 0:
                return maximize(state, agentIndex, firstAction, depth)
            else:
                return minimize(state, agentIndex, firstAction, depth)

        def maximize(state, agentIndex, firstAction, depth):
            maxi = -999999
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                (utility, firstAction) = value(nextState, (agentIndex+1)%k, firstAction, depth+1)
                if utility > maxi:
                    maxi = utility
                    if depth == 0:
                        firstAction = action
            return maxi, firstAction

        def minimize(state, agentIndex, firstAction, depth):
            mini = 999999
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                (utility, firstAction) = value(nextState, (agentIndex+1)%k, firstAction, depth+1)
                if utility < mini:
                    mini = utility
            return mini, firstAction

        k = gameState.getNumAgents()
        (utilityValue, act) = value(gameState, 0, None, 0)
        return act


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def value(state, agentIndex, firstAction, alphaBeta, depth=0):
            """
            @summary: Returns the utility value of a node, while implementing alpha-beta pruning

            @param state:       the GameState, or node, for which to derive its utility value
            @param agentIndex:  an integer representing which agent is determining value; 0 is pacman, 1+ are ghosts
            @param firstAction: a string representing the single action that provides the greatest utility
            @param alphaBeta:   a tuple containing the values of alpha and beta; used for pruning
            @param depth:       an integer representing the current tree depth of the search

            @return: a tuple containing an integer utility value of GameState
                     and a string initial action to obtain that utility
            """
            # If state is terminal state, return state's utility
            #    Here, self.depth is actually the ply and not the tree depth.
            #    In order to compare the two, we multiply the ply by the number of agents. (k * self.depth)
            if state.isWin() or state.isLose() or depth >= (k * self.depth):
                # We return not only the utility value, but also the initial action at root that gets pacman there.
                return self.evaluationFunction(state), firstAction

            # Maximize for agent 0 (pacman), minimize for all others (ghosts)
            #    All parameters are propagated downward without change.
            if agentIndex == 0:
                return maximize(state, agentIndex, firstAction, alphaBeta, depth)
            else:
                return minimize(state, agentIndex, firstAction, alphaBeta, depth)

        def maximize(state, agentIndex, firstAction, alphaBeta, depth):
            """
            @summary: Maximize the utility value from all successor states

            @param state:       the GameState, or node, for which to derive its utility value
            @param agentIndex:  an integer representing which agent is determining value; 0 is pacman, 1+ are ghosts
            @param firstAction: a string representing the single action that provides the greatest utility
            @param alphaBeta:   a tuple containing the values of alpha and beta; used for pruning
            @param depth:       an integer representing the current tree depth of the search

            @return: a tuple containing an integer utility value of GameState
                     and a string initial action to obtain that utility
            """
            (alpha, beta) = alphaBeta
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                # Get utility of next GameState for a legal action
                (utility, firstAction) = value(nextState, (agentIndex+1)%k, firstAction, (alpha, beta), depth+1)
                if utility > alpha:
                    alpha = utility
                    if depth == 0:
                        firstAction = action
                if alpha >= beta and depth != 0:
                    return alpha, firstAction
            return alpha, firstAction

        def minimize(state, agentIndex, firstAction, alphaBeta, depth):
            """
            @summary: Minimize the utility value from all successor states

            @param state:       the GameState, or node, for which to derive its utility value
            @param agentIndex:  an integer representing which agent is determining value; 0 is pacman, 1+ are ghosts
            @param firstAction: a string representing the single action that provides the greatest utility
            @param alphaBeta:   a tuple containing the values of alpha and beta; used for pruning
            @param depth:       an integer representing the current tree depth of the search

            @return: a tuple containing an integer utility value of GameState
                     and a string initial action to obtain that utility
            """
            (alpha, beta) = alphaBeta
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                # Get utility of next GameState for a legal action
                (utility, firstAction) = value(nextState, (agentIndex+1)%k, firstAction, (alpha, beta), depth+1)
                if utility < beta:
                    beta = utility
                    if depth == 0:
                        firstAction = action
                if alpha >= beta and depth != 0:
                    return beta, firstAction
            return beta, firstAction

        k = gameState.getNumAgents()
        (utilityValue, act) = value(gameState, 0, None, (-999999,999999), 0)
        return act


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
        def value(state, agentIndex, firstAction, depth=0):
            if state.isWin() or state.isLose() or depth >= (k * self.depth):
                return self.evaluationFunction(state), firstAction

            if agentIndex == 0:
                return maximize(state, agentIndex, firstAction, depth)
            else:
                return chance(state, agentIndex, firstAction, depth)

        def maximize(state, agentIndex, firstAction, depth):
            maxi = -999999
            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                (utility, firstAction) = value(nextState, (agentIndex+1)%k, firstAction, depth+1)
                if utility > maxi:
                    maxi = utility
                    if depth == 0:
                        firstAction = action
            return maxi, firstAction

        def chance(state, agentIndex, firstAction, depth):
            average = 0
            utilities = []

            for action in state.getLegalActions(agentIndex):
                nextState = state.generateSuccessor(agentIndex, action)
                (utility, firstAction) = value(nextState, (agentIndex+1)%k, firstAction, depth+1)
                utilities.append(utility)

            # Ghosts choose uniformly random, so we average the utility values
            for utility in utilities:
                average += utility
            average /= len(utilities)

            return average, firstAction

        k = gameState.getNumAgents()
        (utilityValue, act) = value(gameState, 0, None)
        return act


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Weighted sum of distance to closest food pellet, current score, and average distance to ghosts.

    Closest food is the most important, followed by the score, then the ghost's distances. Whether or not a ghost is
    scared is considered, and the average distance favors parts of the grid with less ghosts.
    """
    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()

    closestFoodDistance = 999999
    for x in range(0, food.width):
        for y in range(0, food.height):
            if food[x][y] and pos != (x, y):
                foodDistance = manhattanDistance(pos, (x, y))
                if foodDistance < closestFoodDistance:
                    closestFoodDistance = foodDistance

    score = currentGameState.getScore()

    scoreWt = 1/10
    return (1 / closestFoodDistance) + (score * scoreWt)


# Abbreviation
better = betterEvaluationFunction
