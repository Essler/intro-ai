# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

from game import Directions

n = Directions.NORTH
w = Directions.WEST
s = Directions.SOUTH
e = Directions.EAST


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    return [s, s, w, s, w, w, s, w]


def printPriorityQueue(priority_queue):
    for element in priority_queue.heap:
        print(element, end=", ")
    print()


def graphSearch(problem, strategy, heuristic=None):
    # closed = set()
    closed = []

    fringe = None
    if strategy == dfs:
        fringe = util.PriorityQueueWithFunction(lambda x: -len(x[direction]))
    elif strategy == bfs:
        fringe = util.PriorityQueueWithFunction(lambda x: 0)
    elif strategy == ucs:
        fringe = util.PriorityQueueWithFunction(lambda x: x[cost])
    elif strategy == astar:
        fringe = util.PriorityQueueWithFunction(lambda x: x[cost] + heuristic(x[state], problem))

    fringe.push((problem.getStartState(), [], 0))

    while len(fringe.heap) > 0:
        node = fringe.pop()

        if problem.isGoalState(node[state]):
            return node[direction]

        if node[state] not in closed:
            closed.append(node[state])
            for child in problem.getSuccessors(node[state]):
                fringe.push((child[state], node[direction]+[child[direction]], node[cost]+child[cost]))


def graphSearchPQ(problem, strategy):
    closed = []
    fringe = util.PriorityQueue()
    fringe.push((problem.getStartState(), [], 0), 0)
    decr_prio = 0

    while len(fringe.heap) > 0:
        node = fringe.pop()

        # todo don't necessarily want to return the first goal state? dfs/bfs wouldn't care... but ucs/a* would
        # or does the prio queue handle that?
        if problem.isGoalState(node[state]):
            # print('Found the goal!', node[state])
            # print('Solution:', node[direction])
            return node[direction]

        if node[state] not in closed:
            closed.append(node[state])
            # todo check costs for UCS, instead of looping through all successors
            # cost is the 3rd value of a node, may need to add up costs as we push nodes onto fringe...
            # todo check costs + some heuristic for A*, isntead of looping through all
            for child in problem.getSuccessors(node[state]):
                priority = 0
                if strategy == dfs:
                    priority = decr_prio
                    decr_prio -= 1
                elif strategy == bfs:
                    priority = 0
                elif strategy == astar:
                    priority = child[cost] + len(node[direction])
                elif strategy == ucs:
                    priority = child[cost]
                fringe.push((child[state], node[direction] + [child[direction]], node[cost]+child[cost]), priority)


def gSearch(problem, strategy):
    solution = []

    # Represent search tree as priority queue
    priorityQueue = util.PriorityQueue()

    # Initialize search tree with start state of problem
    state = problem.getStartState()
    priorityQueue.push(state, 0)

    # Change fringe handling based on strategy
    # strategies = {
    #     dfs: print(dfs),
    #     bfs: print(bfs),
    #     ucs: print(ucs),
    #     astar: print(astar)
    # }
    # strategies.get(strategy)

    while not problem.isGoalState(state):
        try:
            # Choose a leaf to expand according to strategy
            print('get successors')
            successors = problem.getSuccessors(state, 0)

            print('choose strategy')
            if strategy == dfs:
                print('do dfs, yo')
                for successor in successors:
                    print('successor: ', successor)
                    child_state = successor[0]
                    child_solution = successor[1]
                    print('child_state: ', child_state)
                    print('child_solution: ', child_solution)
                    if problem.isGoalState(child_state):
                        solution.insert(child_solution)
                        return solution

            elif strategy == bfs:
                print('do bfs, yo')
                solution = [s]

            # If node contains goal state, return corresponding solution
            # return solution

            # Otherwise, expand the node and add all to search tree
            priorityQueue.push(successors)

        except:
            # Fail if unable to expand
            print('FAIL: Unable to expand nodes.')
            quit(1)

    return solution


def depthFirstSearch(problem):
    """Search the deepest nodes in the search tree first."""
    return graphSearch(problem, dfs)


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    return graphSearch(problem, bfs)


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    return graphSearch(problem, ucs)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    return graphSearch(problem, astar, heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

# Node Indexes
state = 0
direction = 1
cost = 2
