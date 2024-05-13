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


import util
from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        values_nextIter = self.values.copy()
        for _ in range(self.iterations):
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue
                q_values = []
                for action in self.mdp.getPossibleActions(state):
                    q_values.append(self.getQValue(state, action))
                if len(q_values)!=0:
                    values_nextIter[state] = max(q_values)
            self.values = values_nextIter.copy()

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
        successor_states = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0
        for successor_state, p in successor_states:
            q_value = q_value + p*(self.mdp.getReward(state, action, successor_state)+ self.discount*self.getValue(successor_state))
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        q_values = util.Counter()
        possible_actions = self.mdp.getPossibleActions(state)
        for action in possible_actions:
            q_values[action]=self.getQValue(state,action)
        return q_values.argMax()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
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
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        numStates = len(states)
        for i in range(self.iterations):
            state = states[i % numStates]
            if self.mdp.isTerminal(state):
                continue
            q_values = []
            for action in self.mdp.getPossibleActions(state):
                q_values.append(self.computeQValueFromValues(state, action))    
            self.values[state] = max(q_values)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        parent_states = {}

        non_terminal_states = []
        for s in states: 
            if not self.mdp.isTerminal(s):
                non_terminal_states.append(s)

        for state in non_terminal_states:
            for action in self.mdp.getPossibleActions(state):
                for next_state,prob in self.mdp.getTransitionStatesAndProbs(state, action):
                    if next_state not in parent_states:
                        parent_states[next_state] = set()
                    parent_states[next_state].add(state)

        priorities  = util.PriorityQueue()
        for state in states:
            if not self.mdp.isTerminal(state):
                q_values =[]
                for a in self.mdp.getPossibleActions(state):
                    q_values.append(self.computeQValueFromValues(state, a))
                difference = abs(max(q_values) - self.getValue(state))
                priorities.update(state, -difference)

        for _ in range(self.iterations):
            if priorities.isEmpty():
                break

            state = priorities.pop()
            if not self.mdp.isTerminal(state):
                q_values =[]
                for a in self.mdp.getPossibleActions(state):
                    q_values.append(self.computeQValueFromValues(state, a))
                self.values[state] = max(q_values)
                for parent_state in parent_states[state]:
                    q_values_parent_state = []
                    for a in self.mdp.getPossibleActions(parent_state):
                        q_values_parent_state.append(self.computeQValueFromValues(parent_state, a))
                    difference = abs(max(q_values_parent_state) - self.getValue(parent_state))
                    if difference >= self.theta:
                        priorities.update(parent_state, -difference)

