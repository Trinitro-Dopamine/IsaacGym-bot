import os
import numpy as np
import numpy.matlib
import math

def trans(vec):
    ans=np.matlib.zeros((4,4))
    for i in range(4):
        ans[i,i]=1
    ans[0,3]=vec[0]
    ans[1,3]=vec[1]
    ans[2,3]=vec[2]
    return ans

def rotate(axis,arc):
    ans=np.matlib.zeros((4,4))
    ans[3,3]=1
    if axis=='x':
        ans[0,0]=1
        ans[1,1]=round(np.cos(arc))
        ans[1,2]=round(-np.sin(arc))
        ans[2,1]=round(np.sin(arc))
        ans[2,2]=round(np.cos(arc))
    if axis=='y':
        ans[1,1]=1
        ans[0,0]=round(np.cos(arc))
        ans[0,2]=round(np.sin(arc))
        ans[2,0]=round(-np.sin(arc))
        ans[2,2]=round(np.cos(arc))
    if axis=='z':
        ans[2,2]=1
        ans[0,0]=round(np.cos(arc))
        ans[0,1]=round(-np.sin(arc))
        ans[1,0]=round(np.sin(arc))
        ans[1,1]=round(np.cos(arc))
    return ans

def computeURDF(source_file, target_file):

    output=open(target_file,"w")
    output.write('<robot\n\tname="Ubot">\n')
    input_string=open(source_file)

    datas=input_string.readlines()
    (body_num,joint_num)=datas[0].split(' ')
    body_num=int(body_num)
    joint_num=int(joint_num)
    #x0->c0->y0->
    XBasedBodyTemp=open(r'linkTemplateXBased.urdf').read(10000)
    YBasedBodyTemp=open(r'linkTemplateYBased.urdf').read(10000)
    joint_matrix = [[0 for i in range(4)] for j in range(joint_num)]

    joint_temp=open(r'jointTemplate.urdf').read(10000)

    for m in range(0,joint_num):
        joint_matrix[m]=datas[m+1].strip().split(' ')

    if  joint_matrix[0][1][0]=='0':
        body_edit=YBasedBodyTemp.replace('Index','0')
    elif joint_matrix[0][1][0]=='1':
        body_edit=XBasedBodyTemp.replace('Index','0')
    output.write(body_edit+'\n\n')

    for m in range(0,joint_num):
        #generate body
        if(joint_matrix[m][3][0]=='0'):
            body_edit=XBasedBodyTemp.replace('Index',joint_matrix[m][2])
        elif(joint_matrix[m][3][0]=='1'):
            body_edit=YBasedBodyTemp.replace('Index',joint_matrix[m][2])
        output.write(body_edit+'\n\n')

        joint_edit=joint_temp.replace('StartTag',joint_matrix[m][0])
        joint_edit=joint_edit.replace('EndTag',joint_matrix[m][2])
        if(joint_matrix[m][1][0]=='0'):
            joint_edit=joint_edit.replace('ParentFace','X')
        elif(joint_matrix[m][1][0]=='1'):
            joint_edit=joint_edit.replace('ParentFace','Y')

        if(joint_matrix[m][3][0]=='0'):
            joint_edit=joint_edit.replace('ChildFace','X')
        elif(joint_matrix[m][3][0]=='1'):
            joint_edit=joint_edit.replace('ChildFace','Y')
        
        #from contact face coordinate system to center system
        R1Dic={'0':[-0.0218,0,0],'1':[0,-0.0218,0]}
        R1=trans(R1Dic[joint_matrix[m][1][0]])
        #from original center to target center
        R2Dic={'00':[0.0916,0,0],'01':[0,-0.0916,0],'10':[-0.0916,0,0],'11':[0,0.0916,0]}
        R2=trans(R2Dic[joint_matrix[m][1]])

        arcxy=int(joint_matrix[m][4])*math.pi/2
        if(joint_matrix[m][1][1]=='0'):
            R3=rotate('x',arcxy)
            # ROvalue=str(arcxy)
            # PIvalue=str(0)
        elif(joint_matrix[m][1][1]=='1'):
            R3=rotate('y',arcxy)
            # ROvalue=str(0)
            # PIvalue=str(arcxy)

        #arcz=int(joint_matrix[m][5])*math.pi/2
        arcz=int(0)*math.pi/2
        R4=rotate('z',arcz)
        YAvalue=str(arcz)

        R5Dic={'0':[0.0218,0,0],'1':[0,0.0218,0]}
        R5=trans(R5Dic[joint_matrix[m][3][0]])
        
        #R=R1@R2@R3@R4@R5
        R=R1@R2@R3@R5

        print(joint_matrix[m][0],joint_matrix[m][2])
        print(R)

        Xvalue=str(R[0,3])
        Yvalue=str(R[1,3])
        Zvalue=str(R[2,3])    
        
        joint_edit=joint_edit.replace("Xvalue",Xvalue)
        joint_edit=joint_edit.replace('Yvalue',Yvalue)
        joint_edit=joint_edit.replace('Zvalue',Zvalue)

        PIvalue=np.arcsin(-R[2,0])
        if(np.round(PIvalue)==0):
            YAvalue=np.arcsin(R[1,0]/np.cos(PIvalue))
            ROvalue=np.arcsin(R[2,1]/np.cos(PIvalue))
        elif(np.round(PIvalue)!=0):
            # if(np.arcsin(-R[0,1])!=0):
            #     ROvalue=0
            #     YAvalue=np.arcsin(-R[0,1])
            # elif(np.arccos(R[1,1])!=0):
            #     ROvalue=0
            #     YAvalue=np.arccos(R[1,1])
            if(np.round(np.sin(PIvalue))==1):
                if(np.round(R[0,1])==0):
                    ROvalue=0
                    YAvalue=-np.arccos(R[1,1])
                elif(np.round(R[1,1])==0):
                    ROvalue=0
                    YAvalue=-np.arcsin(R[0,1])
            elif(np.round(np.sin(PIvalue))==-1):
                if(np.round(R[0,1])==0):
                    ROvalue=0
                    YAvalue=np.arccos(R[1,1])
                elif(np.round(R[1,1])==0):
                    ROvalue=0
                    YAvalue=-np.arcsin(R[0,1])
            


        print(ROvalue,PIvalue,YAvalue,'\n')
        joint_edit=joint_edit.replace('ROvalue',str(ROvalue))
        joint_edit=joint_edit.replace('PIvalue',str(PIvalue))
        joint_edit=joint_edit.replace('YAvalue',str(YAvalue))
        #
        output.write(joint_edit+'\n\n')


    output.write('</robot>')
    output.close()
    input_string.close()
    return

if __name__ == '__main__' :
    
    source_file="/home/lynx/Desktop/IsaacGymEnvs-main/assets/urdf/generated_ubot/algorithm/input.txt"
    target_path="/home/lynx/Desktop/IsaacGymEnvs-main/assets/urdf/generated_ubot/urdf/shell.urdf"
    computeURDF(source_file,target_path)
