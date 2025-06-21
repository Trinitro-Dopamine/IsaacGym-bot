import os
import numpy as np
import numpy.matlib
import math
import copy
#https://oi-wiki.org/graph/tree-centroid/
def tree_normalize(original_string_file,normalized_string_file):
    original_string=open(original_string_file)
    normalized_string=open(original_string_file)
    datas=original_string.readlines()
    (body_num,joint_num)=datas[0].split(' ')
    body_num=int(body_num)
    joint_num=int(joint_num)
    joint_matrix = [[0 for i in range(5)] for j in range(joint_num)]

    tree=Tree(body_num)
    for m in range(0,joint_num):
        joint_matrix[m]=datas[m+1].strip().split(' ')
        start=int(joint_matrix[m][0])
        end=(int(joint_matrix[m][2]))
        link_info=[joint_matrix[m][1],joint_matrix[m][3],joint_matrix[m][4],joint_matrix[m][5]]
        tree.insert(start,end,link_info)
    centroids=tree.findCentroid(0,body_num)

    print(centroids,"\n")

    # for centroid in centroids:
    #     centralizedTree=tree.treeCentralize(centroid)
    #     printTreeReindex(centroid,centralizedTree)


def printTreeReindex(self, index):
    #fill in here
    pass


class Node:
    def __init__(self, index):
        self.index = index
        self.father=[]
        self.sons = []
        self.weight=-1

class Tree:
    def __init__(self,num):
        self.nodes=[Node(i) for i in range(0,num)]

    def getWeight(self,index):
        if (self.nodes[index].weight!=-1):
            return self.nodes[index].weight
        weight=0
        for son in self.nodes[index].sons:
            weight =max(weight,self.getWeight(son[0]))
        weight+=1
        self.nodes[index].weight=weight
        return weight

    def insert(self, start, end, link_info):
        self.nodes[start].sons.append([end,link_info])

        #link_info: start start_face end end_face
        self.nodes[end].father=[start,self.reverseLink(link_info)]

    def printTree(self, index):
        print(index,":",end="")
        for son in self.nodes[index].sons:
            print(son[0],end="")
        print("\n")
        for son in self.nodes[index].sons:
            self.printTree(son[0])


    def findCentroid(self, index,num): #gravity , root only , ###############
        #num=self.getWeight(index)
        max=0
        mark=[]
        for i in range(0,num):
            ca_weight=min(self.getWeight(i)-1,num-self.getWeight(i))

            print(ca_weight,i)

            if(ca_weight>max):
                max=ca_weight
                mark=[i]
            elif(ca_weight==max):
                mark.append(i)

        print(max)
        # if max<num/2:
        #     mark=[index]
        # elif (max==num/2)&(len(mark)==1):
        #     mark.apeend(index)
        return mark

    def reverseLink(self,link_info):
        if((link_info[0]=='00')|(link_info[0]=='11')):
            face1=1
        elif((link_info[0]=='10')|(link_info[0]=='01')):
            face1=-1

        if((link_info[1]=='00')|(link_info[1]=='11')):
            face2=1
        elif((link_info[1]=='10')|(link_info[1]=='01')):
            face2=-1


        if(face1*face2==1):
            return [link_info[1],link_info[0],str(int (link_info[2])%4),str(-int(link_info[3])%4)]
        elif(face1*face2==-1):
            return [link_info[1],link_info[0],str(-int(link_info[2])%4),str(-int(link_info[3])%4)]

    def reverseRelation(self, index,f):
        of = self.centralizedTree[index].father
        print(of)
        link_info=of[1]
        ofr=[index,self.reverseLink(link_info)]
        self.centralizedTree[index].sons.append(of)
        self.centralizedTree[index].father=f
        if(f!=[]):
            self.centralizedTree[index].sons.remove(f)

        return of[0],ofr

    def treeCentralize(self,newCenter):
        self.centralizedTree=copy.deepcopy(self.nodes)
        index=newCenter
        #tempGP=centralizedTree[centralizedTree[index].father[0]].father[0]
        f= []
        while (index!=[]):
            index,f=self.reverseRelation(index,f)
        return self.centralizedTree


def tree_hash(tree):
   return tree


if __name__ == '__main__' :
    original_string_file='input.txt'
    normalized_string_file='normalized_input.txt'
    tree_normalize(original_string_file,normalized_string_file)
