'''
*  Author : Tej Patel
*  Version : 1.0.1
*
*  ID3 algorithm implementaion
*  Features : 1) Decision Tree constructed with information gain measure without pruning
*             2) Decision Tree constructed with variance gain measure without pruning
*             3) Decision Tree constructed with information gain measure with pruning
*             4) Decision Tree constructed with variance gain measure with pruning
'''

import sys
import copy
from math import log
from random import randint
import pandas as pd


class TreeNode():
    def __init__(self, attributeName, dataset):
        self.attributeName = attributeName
        self.leftNode = None
        self.rightNode = None
        self.dataset = dataset


class DecisionTree():
    '''
        Here is the algorith to construct decision tree :-
        1)If all values of target in the dataset is True or False then return the corresponding node.
        2)If attributes are empty then return values that are more in the target class, False or True
        3)Select the best attribute from the gain measure and save it as root node also remove this attribute from set of attributes.
        4)Split dataset based on different values taken by the best attribute.
        5)For each new dataset, add a branch below the root node and call recursively the create tree method.
    '''
    def createDTEntropyPrepruning(self, dataset, attributes):
        attCopy = attributes[:]
        if(self.entropy(dataset) == 0):
            return TreeNode(self.majorityClass(dataset), [])
        elif(len(attributes) == 0):
            return TreeNode(self.majorityClass(dataset), [])
        else:
            bestAttribute = None
            bestGain = 0
            for attribute in attributes:
                g = self.gain(dataset, attribute)
                if(g >= bestGain):
                    bestGain = g
                    bestAttribute = attribute
            newDatasets = self.splitDataset(dataset, bestAttribute)
            root = TreeNode(bestAttribute, dataset)
            attCopy.remove(bestAttribute)
            if(len(newDatasets[0]) == 0):
                root.leftNode = TreeNode(self.majorityClass(dataset), [])
            else:
                root.leftNode = self.createDTEntropyPrepruning(newDatasets[0], attCopy)
            if(len(newDatasets[1]) == 0):
                root.rightNode = TreeNode(self.majorityClass(dataset), [])
            else:
                root.rightNode = self.createDTEntropyPrepruning(newDatasets[1], attCopy)
            return root
    
    def createDTVariancePrepruning(self, dataset, attributes):
        attCopy = attributes[:]
        if(self.variance(dataset) == 0):
            return TreeNode(self.majorityClass(dataset), [])
        elif(len(attributes) == 0):
            return TreeNode(self.majorityClass(dataset), [])
        else:
            bestAttribute = None
            bestGain = 0
            for attribute in attributes:
                g = self.varianceImpurity(dataset, attribute)
                if(g >= bestGain):
                    bestGain = g
                    bestAttribute = attribute
            newDatasets = self.splitDataset(dataset, bestAttribute)
            root = TreeNode(bestAttribute, dataset)
            attCopy.remove(bestAttribute)
            if(len(newDatasets[0]) == 0):
                root.leftNode = TreeNode(self.majorityClass(dataset), [])
            else:
                root.leftNode = self.createDTVariancePrepruning(newDatasets[0], attCopy)
            if(len(newDatasets[1]) == 0):
                root.rightNode = TreeNode(self.majorityClass(dataset), [])
            else:
                root.rightNode = self.createDTVariancePrepruning(newDatasets[1], attCopy)
            return root
    
    def createDTEntropyPostpruning(self, trainingset, validationset, attributes, L, K):
        root = self.createDTEntropyPrepruning(trainingset, attributes)
        return self.pruning(validationset, root, L, K)
    
    def createDTVariancePostpruning(self, trainingset, validationset, attributes, L, K):
        root = self.createDTVariancePrepruning(trainingset, attributes)
        return self.pruning(validationset, root, L, K)
        
    def pruning(self, validationset, root, L, K):
        DBest = copy.deepcopy(root)
        bestAccuracy = self.getPredictions(validationset, DBest)[2]
        for i in range(L):
            DTemp = copy.deepcopy(root)
            M = randint(0, K)
            for j in range(M):
                nodesList = self.levelOrder(DTemp)
                N = 0
                
                for level in nodesList:
                    for node in level:
                        if((node is not False) and (node is not True)):
                            N += 1
                if(N <= 1):
                    break
                
                P = randint(2, N)
                DTemp = self.modifyTree(DTemp, P)

            accuracyOfDTemp = self.getPredictions(validationset, DTemp)[2]
            if(accuracyOfDTemp > bestAccuracy):
                DBest = copy.deepcopy(DTemp)
                bestAccuracy = accuracyOfDTemp
                
        return DBest
    
    def entropy(self, dataset):
        classValues = list(dataset['Class'])
        totalLength = len(classValues)
        positiveValues = 0
        log2 = lambda x:log(x)/log(2)

        for values in classValues:
            if(values == 1):
                positiveValues += 1
        positiveFraction = positiveValues/totalLength

        if(positiveFraction == 1 or positiveFraction == 0):
            return 0
        else:
            return ((-positiveFraction*(log2(positiveFraction)))+(-(1-positiveFraction)*log2(1-positiveFraction)))

    def gain(self, dataset, attribute):
        entropyOfDataset = self.entropy(dataset)
        intermediateSum = 0
        sets = self.splitDataset(dataset, attribute)
        for aset in sets:
            if(len(aset) is not 0):
                entropyOfNewDataset = self.entropy(aset)
                intermediateSum += (len(aset)/len(dataset))*(entropyOfNewDataset)
        return entropyOfDataset - intermediateSum
    
    def variance(self, dataset):
        k = len(dataset)
        newDatasets = self.splitDataset(dataset, 'Class')
        k0 = len(newDatasets[0])
        k1 = len(newDatasets[1])
        return (k0*k1)/(k*k)
    
    def varianceImpurity(self, dataset, attribute):
        varianceOfDataset = self.variance(dataset)
        intermediateSum = 0
        sets = self.splitDataset(dataset, attribute)
        for aset in sets:
            if(len(aset) is not 0):
                varianceOfNewDataset = self.variance(aset)
                intermediateSum += (len(aset)/len(dataset))*(varianceOfNewDataset)
        return varianceOfDataset - intermediateSum
    
    def modifyTree(self, treeRoot, P):
        root = treeRoot
        if(P == 1):
            return TreeNode(self.majorityClass(root.dataset), [])
        else:
            que = []
            count = 1
            breakLoop = False
            que.append(root)
            length = len(que)
            while(length != 0):
                temp = []
                for i in range(length):
                    temp.append(que[i].attributeName)
                    if((que[i].leftNode != None) and (que[i].leftNode.attributeName != False) and (que[i].leftNode.attributeName != True)):
                        count += 1
                        if(count == P):
                            que[i].leftNode = TreeNode(self.majorityClass(que[i].leftNode.dataset), [])
                            breakLoop = True
                            break
                        que.append(que[i].leftNode)
                    if((que[i].rightNode != None) and (que[i].rightNode.attributeName != False) and (que[i].rightNode.attributeName != True)):
                        count += 1
                        if(count == P):
                            que[i].rightNode = TreeNode(self.majorityClass(que[i].rightNode.dataset), [])
                            breakLoop = True
                            break
                        que.append(que[i].rightNode)
                if(breakLoop):
                    break
                for i in range(length):
                    que.pop(0)
                length = len(que)
    
            return treeRoot
    
    def splitDataset(self, dataset, attribute):
        distinctValues = [0,1]
        attributeValues = list(dataset[attribute])
        sets = []
        for dvalue in distinctValues:
            newDataset = pd.DataFrame(columns=['Class'])
            for index in range(0, len(dataset)+1):
                if(attributeValues[index-1] == dvalue):
                    newDataset = newDataset.append(dataset[index-1:index])
            sets.append(newDataset)
        return sets
    
    def majorityClass(self, dataset):
        positiveCount = 0
        negativeCount = 0
        targetValues = list(dataset['Class'])
        for value in targetValues:
            if(value == 1):
                positiveCount += 1
            else:
                negativeCount += 1
        if(positiveCount >= negativeCount):
            return True
        else:
            return False
    
    def levelOrder(self, root):
        if(root == None):
            return []
        ans = []
        que = []
        que.append(root)
        length = len(que)
        while(length != 0):
            temp = []
            for i in range(length):
                temp.append(que[i].attributeName)
                if(que[i].leftNode != None):
                    que.append(que[i].leftNode)
                if(que[i].rightNode != None):
                    que.append(que[i].rightNode)
            for i in range(length):
                que.pop(0)
            length = len(que)
            ans.append(temp)
        return ans
    
    def getPredictions(self, testset, root):
        trueClassValues = list(testset['Class'])
        trueCount = 0
        for index in range(1,len(testset)+1):
            temp = root
            for i in range(len(testset[index-1:index].T)):
                pred = 0
                if(testset[temp.attributeName][index-1] == 0):
                    temp = temp.leftNode
                    if(temp.attributeName == True):
                        pred = 1
                        break
                    if(temp.attributeName == False):
                        pred = 0
                        break
                else:
                    temp = temp.rightNode
                    if(temp.attributeName == True):
                        pred = 1
                        break
                    if(temp.attributeName == False):
                        pred = 0
                        break
            if(pred == trueClassValues[index-1]):
                trueCount += 1
        return (trueCount, len(trueClassValues), (trueCount/len(trueClassValues))*100)
    
    def printTreeHelper(self, root, count, s):
        if(root != None):
            if(root.leftNode.attributeName == True):
                print(s*count+root.attributeName+' = 0 : '+str(1))
            elif root.leftNode.attributeName == False:
                print(s*count+root.attributeName+' = 0 : '+str(0))
            else:
                print(s*count+root.attributeName+' = 0')
                count += 1
                self.printTreeHelper(root.leftNode, count, s)
                count -=1
            if(root.rightNode.attributeName == True):
                print(s*count+root.attributeName+' = 1 : '+str(1))
            elif root.rightNode.attributeName == False:
                print(s*count+root.attributeName+' = 1 : '+str(0))
            else:
                print(s*count+root.attributeName+' = 1')
                count += 1
                self.printTreeHelper(root.rightNode, count, s)
                
    def printTree(self, root):
        self.printTreeHelper(root, 0, '|  ')

L = int(sys.argv[1])
K = int(sys.argv[2])
trainingset = pd.read_csv(sys.argv[3])
validationset = pd.read_csv(sys.argv[4])
testset = pd.read_csv(sys.argv[5])
toPrint = sys.argv[6]

dt = DecisionTree()
attributes = list(trainingset)
attributes.remove('Class')

rootEntPrepruning = dt.createDTEntropyPrepruning(trainingset, attributes)
rootVarPrepruning = dt.createDTVariancePrepruning(trainingset, attributes)
rootEntPostpruning = dt.pruning(validationset, rootEntPrepruning, L, K)
rootVarPostpruning = dt.pruning(validationset, rootVarPrepruning, L, K)

print('-'*100)
print()
print("Accuracy for info gain without pruning : "+str(dt.getPredictions(testset, rootEntPrepruning)))
print("Accuracy for variance impurity without pruning : "+str(dt.getPredictions(testset, rootVarPrepruning)))
print("Accuracy for info gain with pruning : "+str(dt.getPredictions(testset, rootEntPostpruning)))
print("Accuracy for variance impurity with pruning : "+str(dt.getPredictions(testset, rootVarPostpruning)))

print()
print('-'*100)
print()

if(toPrint == "yes"):
    print("Decision Tree for info gain without pruning")
    dt.printTree(rootEntPrepruning)
    print()
    print('-'*100)

    print("Decision Tree for variance impurity without pruning")
    dt.printTree(rootVarPrepruning)
    print()
    print('-'*100)

    print("Decision Tree for info gain with pruning")
    dt.printTree(rootEntPostpruning)
    print()
    print('-'*100)

    print("Decision Tree for variance impurity with pruning")
    dt.printTree(rootVarPostpruning)
    print()
    print('-'*100)