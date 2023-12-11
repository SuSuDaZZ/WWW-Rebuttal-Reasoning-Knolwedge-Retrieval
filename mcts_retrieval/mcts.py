import time
import math
import random
import numpy as np

K = 5
C = 3

class treeNode():
    def __init__(self, state, parent):
        self.state = state
        self.parent = parent
        self.numVisits = 0
        self.totalReward = 0
        self.isTerminal = state.isTerminal()
        self.isFullyExpanded = self.isTerminal
        self.children = {}
        self.actions = []
        self.ancestor_prior = 0
        self.node_uct_value = 0
        



class mcts():
    # Implementation of value-based MCTS

    def __init__(self, iterationLimit=None, explorationConstant=1/math.sqrt(2)):
        self.search_level = 0
        self.terminalFlag = False
        self.searchLimit = iterationLimit
        self.explorationConstant = explorationConstant
        


    def search(self, initialState):
        self.root = treeNode(initialState, None)
        self.current_node = self.root
        if self.root.isTerminal:
            return None
        # loop for executing
        while self.search_level < self.searchLimit and not self.terminalFlag:
            terminalNode = self.executeRound()

        # output the state_action chain
        ActionChain = terminalNode.state.state
        uct_value = terminalNode.node_uct_value
        stateActionChain = {"ActionChain": ActionChain, "UCTValue": uct_value}        
        return stateActionChain




    def executeRound(self):
        '''
        Selection - Expansion - Simulation - Backpropagation
        '''
        node = self.current_node
        node = self.selectNode(node)
        if not node.isTerminal and self.search_level<self.searchLimit:
            node = self.expand(node)
            reward = node.state.getReward()
            self.backpropogate(node, reward)
        else:
            self.terminalFlag = True
        return node


    
    def selectNode(self, node):
        if not node.isFullyExpanded:
            return node
        else:
            node = self.getBestChild(node, self.explorationConstant)
            self.current_node = node
            self.search_level += 1
            self.terminalFlag = node.isTerminal
            return node

    

    def expand(self, node):
        if len(node.children)==0:
            node.actions = node.state.getPossibleActions()
        for action in node.actions:
            if action not in node.children:
                new_state = node.state.takeAction(action)
                new_node = treeNode(state=new_state, parent=node)
                node.children[action] = new_node
                if len(node.actions) == len(node.children):
                    node.isFullyExpanded = True
                sent, score, current_central, next_central, sent_id = action.split(" [SEP] ")
                new_node.ancestor_prior = float(score)
                return new_node
        raise Exception("Should never reach here")


    def backpropogate(self, node, reward):
        while node is not None:
            node.numVisits += 1
            node.totalReward += reward
            node = node.parent

    
    def getBestChild(self, node, explorationValue):
        bestValue = float("-inf")
        bestNodes = []
        for child in node.children.values():
            node_Q_value = child.totalReward / child.numVisits
            node_policy_score = explorationValue * child.ancestor_prior * np.sqrt(node.numVisits) / child.numVisits
            node_uct_value = node_Q_value + node_policy_score
            child.node_uct_value = node_uct_value
            if node_uct_value > bestValue:
                bestValue = node_uct_value
                bestNodes = [child]
            elif node_uct_value == bestValue:
                bestNodes.append(child)
        return random.choice(bestNodes)




            


























