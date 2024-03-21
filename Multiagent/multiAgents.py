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


import random
import util

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

        Print out these variables to see what you're getting, then combine the
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newGhostPositions = [ghostState.getPosition() for ghostState in newGhostStates]
        def ghostPacmanDistance():
            totalDistance = 0
            for ghostPos in newGhostPositions:
                distance  = ((newPos[0]-ghostPos[0])**2 + (newPos[1]-ghostPos[1])**2)**0.5    
                totalDistance = totalDistance +distance
            return totalDistance
        
        def foodPacmanDistance():
            food_positions = []
            for x,bool_list in enumerate(newFood):
                for y, bool in enumerate(bool_list):
                    if newFood[x][y] == True:
                        food_positions.append((x,y))
            totalDistance = []
            for foodPos in food_positions:
                distance  = ((newPos[0]-foodPos[0])**2 + (newPos[1]-foodPos[1])**2)**0.5    
                totalDistance.append(distance)
            food_score = 0
            if len(totalDistance)>0:
                food_score = min(totalDistance)
                return food_score
        
        print(foodPacmanDistance())
        total_scared_times = 0
        for t in newScaredTimes:
            total_scared_times = total_scared_times +t 
        
        foodDistanceScore = 0
        if foodPacmanDistance() != None:
            if foodPacmanDistance() > 0:
                foodDistanceScore =  1/(foodPacmanDistance())

        overall_score = successorGameState.getScore() +  total_scared_times  + foodDistanceScore

        if ghostPacmanDistance() <= 1:   
            overall_score = overall_score-50
        return overall_score

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """

        "*** YOUR CODE HERE ***"

        #util.raiseNotDefined()
        import numpy as np
        pacmanLegalActions  = gameState.getLegalActions(0)
        minimax_scores = []
        depth = 0
        
        for action in pacmanLegalActions:
            pacman_successor_state = gameState.generateSuccessor(0, action)
            minimax_scores.append(self.min_value(pacman_successor_state,1,depth))
        return pacmanLegalActions[np.argmax(minimax_scores)]
        
    def min_value(self,state,agentIndex,depth):
        if state.isWin() or state.isLose() :
            return self.evaluationFunction(state)
        else:
            u = 99999
            legalActions = state.getLegalActions(agentIndex)
            for action in legalActions:
                successor_state = state.generateSuccessor(agentIndex,action)
                if agentIndex < state.getNumAgents()-1:
                    u = min(u,self.min_value(successor_state,agentIndex+1,depth))
                elif agentIndex == state.getNumAgents()-1:
                    u = min(u,self.max_value(successor_state,depth+1))
            return u
        
    def max_value(self,state,depth):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        else:
            u = -99999
            legalActions = state.getLegalActions(0)
            for action in legalActions:
                successor_state = state.generateSuccessor(0,action)
                u = max(u,self.min_value(successor_state,1,depth))
            return u


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        "*** YOUR CODE HERE ***"
        import numpy as np
        pacmanLegalActions  = gameState.getLegalActions(0)
        minimax_scores = []
        depth = 0
        alpha = -99999
        beta = 99999
        print("-----max layer-----")
        print("depth : " + str(depth)) 
        for action in pacmanLegalActions:
            pacman_successor_state = gameState.generateSuccessor(0, action)
            u = self.min_value(pacman_successor_state,1,depth,alpha,beta)
            minimax_scores.append(u)
            alpha = max(u,alpha)
        return pacmanLegalActions[np.argmax(minimax_scores)]
        
    def min_value(self,state,agentIndex,depth,alpha,beta):
        print("-----min layer-----")
        print("depth : " + str(depth))
        print("alpha : " + str(alpha)) 
        print("beta : " + str(beta)) 
        if state.isWin() or state.isLose() :
            print("terminal state")
            print("u : " + str( self.evaluationFunction(state)))
            return self.evaluationFunction(state)
        else:
            u = 99999
            legalActions = state.getLegalActions(agentIndex)
            for action in legalActions:
                successor_state = state.generateSuccessor(agentIndex,action)
                if agentIndex < state.getNumAgents()-1:
                    print("intermediate ghost")
                    u = min(u,self.min_value(successor_state,agentIndex+1,depth,alpha,beta))
                elif agentIndex == state.getNumAgents()-1:
                    print("last ghost")
                    u = min(u,self.max_value(successor_state,depth+1,alpha,beta))
                if u< alpha:
                    print("u : "+str(u))
                    return u
                beta = min(u,beta)
            print("u : "+str(u))
            return u
        
    def max_value(self,state,depth,alpha,beta):
        print("-----max layer-----")
        print("depth : " + str(depth)) 
        print("alpha : " + str(alpha)) 
        print("beta : " + str(beta)) 
        if state.isWin() or state.isLose() or depth == self.depth:
            print("terminal state")
            print("u : "+str(self.evaluationFunction(state)))
            return self.evaluationFunction(state)
           
        else:
            u = -99999
            legalActions = state.getLegalActions(0)
            for action in legalActions:
                successor_state = state.generateSuccessor(0,action)
                u = max(u,self.min_value(successor_state,1,depth,alpha,beta))
                if u > beta:
                    print("u : "+str(u))
                    return u
                alpha = max(alpha,u)
            print("u : "+str(u))
            return u


        #util.raiseNotDefined()
        


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


        import numpy as np
        pacmanLegalActions  = gameState.getLegalActions(0)
        minimax_scores = []
        depth = 0
        
        for action in pacmanLegalActions:
            pacman_successor_state = gameState.generateSuccessor(0, action)
            minimax_scores.append(self.exp_value(pacman_successor_state,1,depth))
        return pacmanLegalActions[np.argmax(minimax_scores)]
        
    def exp_value(self,state,agentIndex,depth):
        if state.isWin() or state.isLose() :
            return self.evaluationFunction(state)
        else:
            u = 0
            legalActions = state.getLegalActions(agentIndex)
            for action in legalActions:
                successor_state = state.generateSuccessor(agentIndex,action)
                if agentIndex < state.getNumAgents()-1:
                    print("intermediate ghost")
                    u = self.exp_value(successor_state,agentIndex+1,depth)
                if agentIndex == state.getNumAgents()-1:
                    print("last ghost")
                    u = u + self.max_value(successor_state,depth+1)
            return u
    
    def max_value(self,state,depth):
        if state.isWin() or state.isLose() or depth == self.depth:
            return self.evaluationFunction(state)
        else:
            u = -99999
            legalActions = state.getLegalActions(0)
            for action in legalActions:
                successor_state = state.generateSuccessor(0,action)
                u = max(u,self.exp_value(successor_state,1,depth))
            return u

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Added Food related information
    1. NumberofFoodScore = 1/(Number of food grains)*20 
    When there is more food the NumberofFoodScore goes down
    2. minfoodDistanceScore = 1 /(Distance from nearest food)
    Makes pacman go near the food and eat it
    3. totalFoodDistanceScore = 1/ (Sum of Distances from all food granins)
    Makes pacman go near the bulk of food and eat it
    4. Penalise the score by -50 if distnace from nearest ghost is less than 3 and scaredtime is zero
    5. When scared time is more than zero we penalize for the distance between pacman and 
        nearest ghost so it moves towards the gost to eat it
    """

    "*** YOUR CODE HERE ***"

    #util.raiseNotDefined()
    #successorGameState = currentGameState.generateSuccessor(action)
    PacPos = currentGameState.getPacmanPosition()
    FoodPos = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    GhostPos = [ghostState.getPosition() for ghostState in GhostStates]

    def ghostPacmanDistance():
        totalDistance = []
        for ghostPos in GhostPos:
            distance  = ((PacPos[0]-ghostPos[0])**2 + (PacPos[1]-ghostPos[1])**2)**0.5    
            totalDistance.append(distance)
        return totalDistance
    
    def foodPacmanDistance():
            food_positions = []
            for x,bool_list in enumerate(FoodPos):
                for y, bool in enumerate(bool_list):
                    if FoodPos[x][y] == True:
                        food_positions.append((x,y))
            totalDistance = []
            for foodPos in food_positions:
                distance  = ((PacPos[0]-foodPos[0])**2 + (PacPos[1]-foodPos[1])**2)**0.5    
                totalDistance.append(distance)
            return totalDistance
    
    total_scared_times = 0
    for t in ScaredTimes:
        total_scared_times = total_scared_times +t 
    

    minfoodDistanceScore = 1
    totalFoodDistanceScore = 1
    NumberOfFoodScore = 1
    if foodPacmanDistance() != None:
        if len(foodPacmanDistance()) > 0:
            NumberOfFoodScore = 1/len(foodPacmanDistance())
            if  min(foodPacmanDistance()) > 0:
                minfoodDistanceScore = 1/ min(foodPacmanDistance())
            if sum(foodPacmanDistance()) > 0: 
                totalFoodDistanceScore = 1/sum(foodPacmanDistance())  
        

    overall_score = currentGameState.getScore() +  total_scared_times + 10*minfoodDistanceScore  +10*totalFoodDistanceScore + 20*NumberOfFoodScore 

    if min(ghostPacmanDistance()) <= 3 and total_scared_times ==0:   
        overall_score = overall_score-50

    if total_scared_times > 0:
        overall_score = overall_score + 10/min(ghostPacmanDistance()) 
    return overall_score

# Abbreviation
better = betterEvaluationFunction
