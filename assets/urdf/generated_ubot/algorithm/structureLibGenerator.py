#      ______ 
#     /       \    ____________________________
#    |  o  o   |  |                            |
#    |         |  |code.replace('while', 'if') |
#    |   O     |  | ___________________________| 
#    |         |   V
#    | A A A A |
#     V V V V V

import os
import numpy as np
import numpy.matlib
import math
import copy
import tree_match
import queue
import pandas as pd
import hashlib 
import shutil
from urdf_generator_gym import computeURDF
def find_child64(description_path):
    description_file=open(description_path)
    description=description_file.readlines()
    (body_num,joint_num)=description[0].split(' ')
    body_num=int(body_num)
    joint_num=int(joint_num)

    #ubot index,face index,rotate_xy, rotate_z
    #0 00 1 00 0 0
    #0 00 1 10 0 2 
    surfaces=np.ones((body_num,4,4,4),dtype=bool)
    joint_matrix = [[0 for i in range(5)] for j in range(joint_num)]

    for m in range(0,joint_num):
        joint_matrix[m]=description[m+1].strip().split(' ')
        surfaces[int(joint_matrix[m][0]),int(joint_matrix[m][1],2),int(joint_matrix[m][4]),int(joint_matrix[m][5])]=False
        surfaces[int(joint_matrix[m][0]),int(joint_matrix[m][1],2),int(joint_matrix[m][4]),int(joint_matrix[m][5])]=False
        return surfaces

#while 2 module connect, only 16 combinations
def find_child16(description_path):
    description_file=open(description_path)
    description=description_file.readlines()
    (body_num,joint_num)=description[0].split(' ')
    body_num=int(body_num)
    joint_num=int(joint_num)

    #ubot index,face index,rotate_xy, rotate_z
    #0 00 1 00 0 0
    #0 00 1 10 0 2 
    surfaces=np.ones((body_num,4,4),dtype=bool)
    joint_matrix = [[0 for i in range(5)] for j in range(joint_num)]

    for m in range(0,joint_num):
        joint_matrix[m]=description[m+1].strip().split(' ')
        surfaces[int(joint_matrix[m][0]),int(joint_matrix[m][1],2),int(joint_matrix[m][4]),int(joint_matrix[m][5])]=False
    return surfaces

def expandDescription64(from_seed=False,top=100):
    file_path = os.path.abspath(__file__)
    directory = os.path.dirname(file_path)
    structure_path=directory+"/../structure_library"
    if(from_seed==True):  
        shutil.copy(structure_path+"/reward_table_seed.csv", structure_path+"/reward_table.csv")

    reward_table=pd.read_csv(structure_path+"/reward_table.csv")
    current_index=0
    while (current_index<top):
        print(current_index,'/',top)
        if(reward_table.iloc[current_index]['child']=='tbs'):
            current_description_file=structure_path+"/library/"+str(current_index)+"/description.txt"
            current_description_content=open(current_description_file,'r')
            previous_structure=current_description_content.readlines()[1:]

            current_size=reward_table.iloc[current_index]["size"]
            if(current_size>5):
                break
            surfaces=find_child64(current_description_file)
            #here, make it iterate every element but not next subset
            #for surface in surfaces:
            for i in range(0,current_size):
                for j in range(0,4):
                    for k in range(0,4):
                        for l in range(0,4):
                            if (surfaces[i,j,k,l]==False):
                                continue

                            new_index=int(reward_table.iloc[-1]['index']+1)
                            new_desription_path=structure_path+'/library/'+str(new_index)
                            #modify new description (debugging)
                            try:
                                os.mkdir(new_desription_path)
                                #print(f"Directory '{new_desription_path}' created successfully.")
                            except FileExistsError:
                                #print(f"Directory '{new_desription_path}' already exists.")
                                pass
                            except OSError as error:
                                #print(f"Error: {error}")
                                pass
                            new_description_file=new_desription_path+'/description.txt'
                            with open(new_description_file,'w') as new_description_content:
                                new_description_content.writelines(str(current_size+1)+' '+str(current_size)+'\n')
                                new_description_content.writelines(previous_structure)
                                new_line=str(i)+' '+str(bin(j)[2:].zfill(2))+' '+str(current_size)+' '+str(bin((j+l)%4)[2:].zfill(2))+' '+str(k)+' '+str(l)+'\n'
                                #print(current_index,':',new_line)
                                new_description_content.write(new_line)
                            #append the new description into reward table (on going,todo: modify parent's child)
                            hash_obj = hashlib.sha256()
                            new_description_content =open(new_description_file, 'rb')
                            chunk = new_description_content.read()
                            hash_obj.update(chunk)
                            description_hash=hash_obj.hexdigest()
                            child=reward_table.iloc[current_index]['child']
                            if child=='tbs':
                                child=str(new_index)
                            else:
                                child+=(' '+str(new_index))
                            reward_table.loc[current_index,'child']=child
                            reward_table.loc[len(reward_table)] = {
                                "index":new_index,
                                "size":reward_table.iloc[current_index]['size']+1,
                                "isomorphic":'tbs',
                                "URDF":"none",
                                "USD":"none",
                                "reward":"none",
                                "parent":current_index,
                                "child":'tbs',
                                "hash":description_hash}
                            reward_table.to_csv(structure_path+"/reward_table.csv",index=False)
            current_index=current_index+1
    return

def expandDescription16(from_seed=False,top=100):
    file_path = os.path.abspath(__file__)
    directory = os.path.dirname(file_path)
    structure_path=directory+"/../structure_library"
    if(from_seed==True):  
        shutil.copy(structure_path+"/reward_table_seed.csv", structure_path+"/reward_table.csv")

    reward_table=pd.read_csv(structure_path+"/reward_table.csv")
    current_index=0
    while (current_index<top):
        print(current_index,'/',top)
        if(reward_table.iloc[current_index]['child']=='tbs'):
            current_description_file=structure_path+"/library/"+str(current_index)+"/description.txt"
            current_description_content=open(current_description_file,'r')
            previous_structure=current_description_content.readlines()[1:]

            current_size=reward_table.iloc[current_index]["size"]
            if(current_size>5):
                break
            surfaces=find_child16(current_description_file)
            print(surfaces)
            input()
            #here, make it iterate every element but not next subset
            #for surface in surfaces:
            for i in range(0,current_size):
                for j in range(0,4):
                    for k in range(0,4):
                        for l in range(0,4):
                            if (surfaces[i,j,k,l]==False):
                                continue

                            new_index=int(reward_table.iloc[-1]['index']+1)
                            new_desription_path=structure_path+'/library/'+str(new_index)
                            #modify new description (debugging)
                            try:
                                os.mkdir(new_desription_path)
                                #print(f"Directory '{new_desription_path}' created successfully.")
                            except FileExistsError:
                                #print(f"Directory '{new_desription_path}' already exists.")
                                pass
                            except OSError as error:
                                #print(f"Error: {error}")
                                pass
                            new_description_file=new_desription_path+'/description.txt'
                            with open(new_description_file,'w') as new_description_content:
                                new_description_content.writelines(str(current_size+1)+' '+str(current_size)+'\n')
                                new_description_content.writelines(previous_structure)
                                new_line=str(i)+' '+str(bin(j)[2:].zfill(2))+' '+str(current_size)+' '+str(bin((j+l)%4)[2:].zfill(2))+' '+str(k)+' '+str(l)+'\n'
                                #print(current_index,':',new_line)
                                new_description_content.write(new_line)
                            #append the new description into reward table (on going,todo: modify parent's child)
                            hash_obj = hashlib.sha256()
                            new_description_content =open(new_description_file, 'rb')
                            chunk = new_description_content.read()
                            hash_obj.update(chunk)
                            description_hash=hash_obj.hexdigest()
                            child=reward_table.iloc[current_index]['child']
                            if child=='tbs':
                                child=str(new_index)
                            else:
                                child+=(' '+str(new_index))
                            reward_table.loc[current_index,'child']=child
                            reward_table.loc[len(reward_table)] = {
                                "index":new_index,
                                "size":reward_table.iloc[current_index]['size']+1,
                                "isomorphic":'tbs',
                                "URDF":"none",
                                "USD":"none",
                                "reward":"none",
                                "parent":current_index,
                                "child":'tbs',
                                "hash":description_hash}
                            reward_table.to_csv(structure_path+"/reward_table.csv",index=False)
            current_index=current_index+1
    return

def convertDesc2URDF():
    file_path = os.path.abspath(__file__)
    directory = os.path.dirname(file_path)
    structure_path=directory+"/../structure_library"

    reward_table=pd.read_csv(structure_path+"/reward_table.csv")
    print(reward_table)
    input()
    current_index=1
    while (current_index<reward_table.shape[0]):
        print(current_index,"/",reward_table.shape[0])
        if(reward_table.iloc[current_index]['URDF']=='none'):
            current_description_file=structure_path+"/library/"+str(current_index)+"/description.txt"
            target_file = structure_path+"/library/"+str(current_index)+"/ubot.urdf"
            computeURDF(current_description_file,target_file)
            reward_table.loc[current_index,'URDF']="yes"
            print(reward_table.loc[current_index]["URDF"])
            reward_table.to_csv(structure_path+"/reward_table.csv",index=False)     
        current_index=current_index+1    
    return


if __name__ == '__main__' :
    
    # original_string_file='input.txt'
    # normalized_string_file='normalized_input.txt'
    #expandDescription16(from_seed=True,top=10)
    expandDescription64(from_seed=True,top=10)
    #convertDesc2URDF()
    #trainManager()

    