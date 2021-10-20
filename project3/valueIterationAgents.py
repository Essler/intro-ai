# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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
import math

import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        """*** YOUR CODE HERE ***"""

        # Get all states from MDP
        states = self.mdp.getStates()

        # For the specified number of iterations...
        for k in range(self.iterations):
            iterationValues = util.Counter()

            # And for each state in the MDP...
            for s in states:
                maxValue = -math.inf
                # And for each possible action from a state...
                actions = self.mdp.getPossibleActions(s)
                for a in actions:
                    # Get the Q-Value
                    qValue = self.computeQValueFromValues(s, a)
                    # Maximize the value
                    if qValue > maxValue:
                        maxValue = qValue
                    iterationValues[s] = maxValue

            self.values = iterationValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Initialize summation to zero
        qValue = 0

        # For each possible transition from current state using specified action...
        transitionList = self.mdp.getTransitionStatesAndProbs(state, action)
        for transition in transitionList:
            # a transition is a pair containing the next state and the probability to reach it
            (nextState, prob) = transition

            # get immediate reward from taking specified action
            # (Note, the next state doesn't matter, since reward depends only on current state.)
            reward = self.mdp.getReward(state, action, nextState)

            # add next value, weighted by its probability, to the summation
            qValue += prob * (reward + self.discount * self.getValue(nextState))

        return qValue

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        return self.getMaxAction(state)

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        """
          Returns the policy at the state (no exploration).
        """
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

    def getMax(self, s):
        maxValue = -math.inf
        maxAction = None

        # For each possible action from a state...
        actions = self.mdp.getPossibleActions(s)
        for a in actions:
            # Get the Q-value
            qValue = self.computeQValueFromValues(s, a)

            # Keep track of the highest Q-value across all possible actions from s
            if qValue > maxValue:
                maxValue = qValue
                maxAction = a

        return maxValue, maxAction

    def getMaxQValue(self, s):
        return self.getMax(s)[0]

    def getMaxAction(self, s):
        return self.getMax(s)[1]


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """*** YOUR CODE HERE ***"""
        # Get all states from MDP
        states = self.mdp.getStates()

        # For the specified number of iterations...
        for k in range(self.iterations):
            # Get the next state in the cycle
            s = states[k % len(states)]
            maxValue = -math.inf

            # And for each legal action from that state...
            actions = self.mdp.getPossibleActions(s)
            for a in actions:
                # Get the Q-Value
                qValue = self.computeQValueFromValues(s, a)
                # Maximize the value
                if qValue > maxValue:
                    maxValue = qValue

            # Only if the state has available actions do we update the value
            if len(actions) > 0:
                self.values[s] = maxValue


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        """*** YOUR CODE HERE ***"""

        # Compute predecessors of all states.
        predecessors = self.computeAllPredecessors()

        # Initialize an empty priority queue.
        q = util.PriorityQueue()

        # For each non-terminal state s, do:
        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                # Find the absolute value of the difference between the current value of s in self.values
                # and the highest Q-value across all possible actions from s.
                diff = abs(self.getValue(s) - self.getMaxQValue(s))

                # Push s into the priority queue with priority -diff.
                # We use a negative because the priority queue is a min heap,
                # but we want to prioritize updating states that have a higher error.
                q.push(s, -diff)

        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for k in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if q.isEmpty():
                break

            # Pop a state s off the priority queue.
            s = q.pop()

            # Update the value of s (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(s):
                self.values[s] = self.getMaxQValue(s)

            # For each predecessor p of s, do:
            for p in predecessors[s]:
                # Find the absolute value of the difference between the current value of s in self.values
                # and the highest Q-value across all possible actions from p
                diff = abs(self.getValue(p) - self.getMaxQValue(p))

                # If diff > theta, push p into the priority queue with priority -diff.
                if diff > self.theta:
                    # as long as it does not already exist in the priority queue with equal or lower priority.
                    q.update(p, -diff)

    def computeAllPredecessors(self):
        predecessors = {}
        for s in self.mdp.getStates():
            for a in self.mdp.getPossibleActions(s):
                t = self.mdp.getTransitionStatesAndProbs(s, a)
                for (nextState, probability) in t:
                    if nextState not in predecessors:
                        # Make sure to store predecessors in a set, not a list, to avoid duplicates.
                        predecessors[nextState] = set()
                    predecessors[nextState].add(s)
        return predecessors
