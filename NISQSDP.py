#nt# -*- coding: utf-8 -*-
"""


@author: Tobias Haug, tobias.haug@u.nus.edu
Program for NISQ SDP
"Nisq algorithm for semidefinite programming"
K Bharti, T Haug, V Vedral, LC Kwek
arXiv:2106.03891


For finding ground state of Hamiltonians as well as finding optimal POVMs for state discrimination
Support various hamiltonians and initial states.
Runs either with generalized eigenvalue problem, or SDP implemented via matlab (use do_SDP=True)


Find ground state of Hamiltonian 
- transverse and longitudonal Ising for model=14 (either SDP or with generalized eigenvalue problem)
- to find ground state within symmetry sector, need SDP
Find optimal POVMs with model=30 (only works with SDP)

To run with SDP, need to install Matlab and CVXPY for Matlab, as well as matlab.engine for python


"""

import os

import qutip as qt


import operator
from functools import reduce
import numpy as np
from matplotlib.ticker import AutoMinorLocator


import scipy

import scipy.optimize

import time
#from helper_tools import *


import matplotlib.pyplot as plt

def plot1D(data,x,xlabelstring="",ylabelstring="",
           logx=False,logy=False,legend=[]):

    fig_size=(6,5)
    #self constructed from color brewer
    colormap=np.array([(56,108,176),(251,128,114),
                       (51,160,44),(253,191,111),(227,26,28),
                       (178,223,138),(166,206,227),(255,127,0),
                       (202,178,214),(106,61,154),(0,0,0)])/np.array([255.,255.,255.]) 
    
    elements=len(data)


    fsize=18
    fsizeLabel=fsize+12
    fsizeLegend=19
    
    plt.figure(figsize=fig_size)

    ax = plt.gca()
    

    plot1DLinestyle=flatten([["solid","dashed","dotted","dashdot"] for i in range(elements//4+1)])


    markerStyle=flatten([["o","X","v","^","s",">","<"] for i in range(elements//4+1)])

    markersize=3
    lwglobal=3
    tickerWidth=1.2
    minorLength=4
    majorLength=8
    linewidth=[lwglobal for k in range(elements)]

    dashdef=[]
    for i in range(elements):
        if(plot1DLinestyle[i]=="solid"):
            dashdef.append([1,1])
        elif(plot1DLinestyle[i]=="dotted"):
            dashdef.append([0.2,1.7])
            linewidth[i]*=1.7
        elif(plot1DLinestyle[i]=="dashed"):
            dashdef.append([3,2])
        elif(plot1DLinestyle[i]=="dashdot"):
            dashdef.append([5,2.5,1.5,2.5])

        
        else:
            dashdef.append([1,1])

            
    for i in range(elements):
        l,=plt.plot(x[i],data[i],color=colormap[i],linewidth=linewidth[i],
            linestyle=plot1DLinestyle[i],marker=markerStyle[i],ms=markersize,dash_capstyle = "round")
    
        
    if(logx==True):
        ax.set_xscale('log')

    if(logy==True):
        ax.set_yscale('log')
        

    plt.locator_params(axis = 'x',nbins=4)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    plt.locator_params(axis = 'y',nbins=4)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        
    plt.tick_params(axis ='both',which='both', width=tickerWidth)
    plt.tick_params(axis ='both',which='minor', length=minorLength)
    plt.tick_params(axis ='both', which='major', length=majorLength)



    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
                
    plt.xlabel(xlabelstring)
    plt.ylabel(ylabelstring)
    
    if(len(legend)>0):
        plt.legend(legend, fontsize=fsizeLegend,columnspacing=0.5)
            


    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(fsizeLabel)
    for item in ([]+ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsize)
    

    

def prod(factors):
    return reduce(operator.mul, factors, 1)


def flatten(l):
    return [item for sublist in l for item in sublist]

def genFockOp(op,position,size,levels,opdim=0):
    opList=[qt.qeye(levels) for x in range(size-opdim)]
    opList[position]=op
    return qt.tensor(opList)




#multiplies paulis together
def multiply_paulis(curr_paulis,to_mult_paulis,curr_val_expansion=[],to_val_expansion=[]):
    new_paulis=[]
    new_val=[]
    for i in range(len(curr_paulis)):
        for j in range(len(to_mult_paulis)):
            add_pauli=np.zeros(len(curr_paulis[i]),dtype=int)

            for k in range(len(curr_paulis[i])):
                if(curr_paulis[i][k]==1 and to_mult_paulis[j][k]==2):
                    add_pauli[k]=3
                elif(curr_paulis[i][k]==2 and to_mult_paulis[j][k]==1):
                    add_pauli[k]=3
                else:
                    add_pauli[k]=np.abs(curr_paulis[i][k]-to_mult_paulis[j][k])

            new_paulis.append(add_pauli)
            if(len(curr_val_expansion)>0):
                new_val.append(curr_val_expansion[i]*to_val_expansion[j])

    #new_paulis=list(np.unique(new_paulis,axis=0))
    new_paulis,inverse_array=list(np.unique(new_paulis,axis=0,return_inverse=True))
    new_val=np.array(new_val)
    new_val_unique=np.zeros(len(new_paulis))
    
    #reconstruct weight values for each pauli, for paulis which occur multiple times are added up
    for i in range(len(new_val)):
        new_val_unique[inverse_array[i]]+=np.abs(new_val[i])
    

    return new_paulis,new_val_unique






def get_ini_state(ini_state_type):
    global anneal_time_opt
    #get initial state

    if(ini_state_type==0):#product state plust state
        initial_state=qt.tensor([qt.basis(levels,1)+qt.basis(levels,0) for i in range(n_qubits)])
        
        #initial_state=qt.tensor([qt.basis(levels,1)]+[qt.basis(levels,0) for i in range(L-1)]) #tjis was used for paper to compare against imag time evolution
        


    elif(ini_state_type==1):#all 0
        initial_state=qt.tensor([qt.basis(levels,0) for i in range(n_qubits)])
        
    elif(ini_state_type==2): #random state
    
        rand_angles=np.random.rand(depth,n_qubits)*2*np.pi
        rand_pauli=np.random.randint(1,4,[depth,n_qubits])

        entangling_layer=prod([opcsign[j] for j in range(n_qubits-1)])
        initial_state=qt.tensor([qt.basis(levels,0) for i in range(n_qubits)])
        initial_state=qt.tensor([qt.qip.operations.ry(np.pi/4) for i in range(n_qubits)])*initial_state

        for j in range(depth):

            rot_op=[]
            for k in range(n_qubits):
                angle=rand_angles[j][k]
                if(rand_pauli[j][k]==1):
                    rot_op.append(qt.qip.operations.rx(angle))
                elif(rand_pauli[j][k]==2):
                    rot_op.append(qt.qip.operations.ry(angle))
                elif(rand_pauli[j][k]==3):
                    rot_op.append(qt.qip.operations.rz(angle))
                    

            initial_state=qt.tensor(rot_op)*initial_state

            initial_state=entangling_layer*initial_state

            


    elif(ini_state_type==15): #time annealing ansatz
        if(model==0):
            # H1=0 ##ZZ terms
            # for i in range(len(Hstrings)//2):
            #     H1+=Hvalues[i]*get_pauli_op(Hstrings[i])
            # H2=0 #X terms
            # for i in range(len(Hstrings)//2):
            #     H2+=Hvalues[i+len(Hstrings)//2]*get_pauli_op(Hstrings[i+len(Hstrings)//2])
                
                
            initial_state=qt.tensor([qt.basis(levels,1)-qt.basis(levels,0) for i in range(n_qubits)])
            initial_state/=initial_state.norm()
            if(annealtime!=0):
                ZZ=qt.tensor([qt.sigmaz(),qt.sigmaz()])
                for i in range(depth):
                    ZZrot=(-1j*Hvalues[0]*annealtime/depth*ZZ*(i+1)).expm()
                    for j in range(n_qubits):
                        initial_state=qt.qip.operations.gate_expand_2toN(ZZrot,n_qubits,j,(j+1)%n_qubits)*initial_state
                    for j in range(n_qubits):
                        initial_state=qt.qip.operations.gate_expand_1toN(qt.qip.operations.rx(2*Hvalues[-1]*annealtime),n_qubits,j)*initial_state
        elif(model==14): ##transverse ising with longitonidal field
            initial_state=qt.tensor([qt.basis(levels,1)-qt.basis(levels,0) for i in range(n_qubits)])
            initial_state/=initial_state.norm()
            if(annealtime!=0):
                ZZ=qt.tensor([qt.sigmaz(),qt.sigmaz()])
                for i in range(depth):
                    for j in range(n_qubits):
                        ZZrot=(-1j*J*Hvalues[j]*annealtime/depth*ZZ*(i+1)).expm()
                        initial_state=qt.qip.operations.gate_expand_2toN(ZZrot,n_qubits,j,(j+1)%n_qubits)*initial_state
                    for j in range(n_qubits):
                        initial_state=qt.qip.operations.gate_expand_1toN(qt.qip.operations.rz(2*Hvalues[j+2*n_qubits]*annealtime/depth*(i+1)),n_qubits,j)*initial_state
                    for j in range(n_qubits):
                        initial_state=qt.qip.operations.gate_expand_1toN(qt.qip.operations.rx(2*Hvalues[j+n_qubits]*annealtime),n_qubits,j)*initial_state
                     
    elif(ini_state_type==16): ##optimize annealing time
        
        def get_energy_ramp(x):
            annealtime=x[0]
            initial_state=qt.tensor([qt.basis(levels,1)-qt.basis(levels,0) for i in range(n_qubits)])
            initial_state/=initial_state.norm()
            if(model==0):
                if(annealtime!=0):
                    ZZ=qt.tensor([qt.sigmaz(),qt.sigmaz()])
                    for i in range(depth):
                        ZZrot=(-1j*Hvalues[0]*annealtime/depth*ZZ*(i+1)).expm()
                        for j in range(n_qubits):
                            initial_state=qt.qip.operations.gate_expand_2toN(ZZrot,n_qubits,j,(j+1)%n_qubits)*initial_state
                        for j in range(n_qubits):
                            initial_state=qt.qip.operations.gate_expand_1toN(qt.qip.operations.rx(2*Hvalues[-1]*annealtime),n_qubits,j)*initial_state
                            
            elif(model==14): ##transverse ising  
                if(annealtime!=0):
                    ZZ=qt.tensor([qt.sigmaz(),qt.sigmaz()])
                    for i in range(depth):
                        for j in range(n_qubits):
                            ZZrot=(-1j*J*Hvalues[j]*annealtime/depth*ZZ*(i+1)).expm()
                            initial_state=qt.qip.operations.gate_expand_2toN(ZZrot,n_qubits,j,(j+1)%n_qubits)*initial_state
                        for j in range(n_qubits):
                            initial_state=qt.qip.operations.gate_expand_1toN(qt.qip.operations.rz(2*Hvalues[j+2*n_qubits]*annealtime/depth*(i+1)),n_qubits,j)*initial_state
                        for j in range(n_qubits):
                            initial_state=qt.qip.operations.gate_expand_1toN(qt.qip.operations.rx(2*Hvalues[j+n_qubits]*annealtime),n_qubits,j)*initial_state
                         
            
            circuit_energy=qt.expect(H,initial_state)
            return circuit_energy
        
        annealtime_ini=annealtime
        ramp_method="Nelder-Mead"
        options={"maxiter":20}
        res=scipy.optimize.minimize(get_energy_ramp,[annealtime_ini],method=ramp_method,options=options)
        anneal_time_opt=res["x"][0]
        print("anneal time found",anneal_time_opt,res["fun"],res["nit"])

        initial_state=qt.tensor([qt.basis(levels,1)-qt.basis(levels,0) for i in range(n_qubits)])
        initial_state/=initial_state.norm()
        annealtime_run=anneal_time_opt
        if(model==0):
            if(annealtime_run!=0):
                ZZ=qt.tensor([qt.sigmaz(),qt.sigmaz()])
                for i in range(depth):
                    ZZrot=(-1j*Hvalues[0]*annealtime_run/depth*ZZ*(i+1)).expm()
                    for j in range(n_qubits):
                        initial_state=qt.qip.operations.gate_expand_2toN(ZZrot,n_qubits,j,(j+1)%n_qubits)*initial_state
                    for j in range(n_qubits):
                        initial_state=qt.qip.operations.gate_expand_1toN(qt.qip.operations.rx(2*Hvalues[-1]*annealtime_run),n_qubits,j)*initial_state
            
        elif(model==14): ##transverse ising  with longitudinal field
            if(annealtime_run!=0):
                ZZ=qt.tensor([qt.sigmaz(),qt.sigmaz()])
                for i in range(depth):
                    for j in range(n_qubits):
                        ZZrot=(-1j*J*Hvalues[j]*annealtime_run/depth*ZZ*(i+1)).expm()
                        initial_state=qt.qip.operations.gate_expand_2toN(ZZrot,n_qubits,j,(j+1)%n_qubits)*initial_state
                    for j in range(n_qubits):
                        initial_state=qt.qip.operations.gate_expand_1toN(qt.qip.operations.rz(2*Hvalues[j+2*n_qubits]*annealtime_run/depth*(i+1)),n_qubits,j)*initial_state
                    for j in range(n_qubits):
                        initial_state=qt.qip.operations.gate_expand_1toN(qt.qip.operations.rx(2*Hvalues[j+n_qubits]*annealtime_run),n_qubits,j)*initial_state
                     


    initial_state/=initial_state.norm()
    
    return initial_state


#get pauli strings for models
def get_Hamiltonian_string(L,model,J,h):
    Hstrings=[]
    Hvalues=[]
    Hoffset=0
    
    if(model==0):#ising
        if(J!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=3
                paulistring[(i+1)%L]=3
                Hstrings.append(list(paulistring))
                Hvalues.append(0.5*J)
        if(h!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=1
                Hstrings.append(list(paulistring))
                Hvalues.append(0.5*h)
    
    elif(model==1):#heisenberg
    
        if(h!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=3
                paulistring[(i+1)%L]=3
                Hstrings.append(list(paulistring))
                Hvalues.append(h)
                
        if(J!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=2
                paulistring[(i+1)%L]=2
                Hstrings.append(list(paulistring))
                Hvalues.append(J)
        

            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=1
                paulistring[(i+1)%L]=1
                Hstrings.append(list(paulistring))
                Hvalues.append(J)
            

            

                
    elif(model==14):# transvesre ising with longitudonal field
        if(J!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=3
                paulistring[(i+1)%L]=3
                Hstrings.append(list(paulistring))
                Hvalues.append(J)
        if(h!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=1
                Hstrings.append(list(paulistring))
                Hvalues.append(h)     
        if(g!=0):
            for i in range(L):
                paulistring=np.zeros(L,dtype=int)
                paulistring[i]=3
                Hstrings.append(list(paulistring))
                Hvalues.append(g)    
                

            
        
        
    HpauliFactor=np.zeros([len(Hstrings),L,4])
    for i in range(len(Hstrings)):
        pauliFactor=np.zeros([L,4])
        for j in range(L):
            if(Hstrings[i][j]==0):
                pauliFactor[j]=[1,0,0,0]
            elif(Hstrings[i][j]==1):
                pauliFactor[j]=[0,1,0,0]
            elif(Hstrings[i][j]==2):
                pauliFactor[j]=[0,0,1,0]
            elif(Hstrings[i][j]==3):
                pauliFactor[j]=[0,0,0,1]
        HpauliFactor[i]=pauliFactor
    return Hstrings,HpauliFactor,Hvalues,Hoffset


#make operator from pauli string
def get_pauli_op(pauli_string):
    pauli_circuit=opId
    for i in range(len(pauli_string)):
        if(pauli_string[i]!=0):
            if(pauli_string[i]==1):
                pauli_circuit=pauli_circuit*opX[i]
            elif(pauli_string[i]==2):
                pauli_circuit=pauli_circuit*opY[i]
            elif(pauli_string[i]==3):
                pauli_circuit=pauli_circuit*opZ[i]

    return pauli_circuit





def numberToBase(n, b,n_qubits):
    if n == 0:
        return np.zeros(n_qubits,dtype=int)
    digits = np.zeros(n_qubits,dtype=int)
    counter=0
    while n:
        digits[counter]=int(n % b)
        n //= b
        counter+=1
    return digits[::-1]
                
def decomposePauli(H):
    """Decompose Hermitian matrix H into Pauli matrices"""
    sx = np.array([[0, 1],  [ 1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0],  [0, -1]], dtype=np.complex128)
    id = np.array([[1, 0],  [ 0, 1]], dtype=np.complex128)
    S = [id, sx, sy, sz]
    dim_matrix=np.shape(H)[0]
    n_qubits=int(np.log2(dim_matrix))
    if(dim_matrix!=2**n_qubits):
        raise NameError("matrix is not power of 2!")
    hilbertspace=2**n_qubits
    n_paulis=4**n_qubits
    pauli_list=np.zeros([n_paulis,n_qubits],dtype=int)
    for k in range(n_paulis):
        pauli_list[k,:]=numberToBase(k,4,n_qubits)
    weights=np.zeros(n_paulis,dtype=np.complex128)
    for k in range(n_paulis):
        pauli=S[pauli_list[k][0]]

        for n in range(1,n_qubits):
            pauli=np.kron(pauli,S[pauli_list[k][n]])

        #weights[k] = 1/hilbertspace* (np.dot(H.conjugate().transpose(), pauli)).trace()
        weights[k] = 1/hilbertspace* np.dot(pauli,H).trace()

    return pauli_list,weights


def reconstructMatrix(pauli_list,weights):
    sx = np.array([[0, 1],  [ 1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0],  [0, -1]], dtype=np.complex128)
    id = np.array([[1, 0],  [ 0, 1]], dtype=np.complex128)
    S = [id, sx, sy, sz]
    
    n_paulis=len(weights)
    n_qubits=np.shape(pauli_list)[1]
    H=np.zeros([2**n_qubits,2**n_qubits],dtype=np.complex128)
    for k in range(n_paulis):
        if(weights[k]!=0):
            pauli=S[pauli_list[k][0]]
    
            for n in range(1,n_qubits):
                pauli=np.kron(pauli,S[pauli_list[k][n]])
            H+=weights[k]*pauli
    return H



starttime=time.time()




seed=1
np.random.seed(seed)

n_qubits=8#number of qubits


#state to use as ansatz for k-moment expansion 
ini_type=16 #initial state for ansatz state via K moment, 0: + product state, 1: zero state, 2: random circuit, 15: circuit annealing with fixed anneal time, 16: circuit annealing where annealing time is optimized
annealtime=0.3#0.66 #annealing time for ini_type=15

depth=10#layers of circuit


do_SDP=False #for model=0/1/14, do SDP or solve generalized eigenvalue problem, model=30 only works with SDP




model=14 #model to be optimized: 0: transverse ising, 1: Heisenberg ,  14: transverse ising model with longitudinal field, 30: find optimal POVMs for state discrimination
h=1 #for models the h parameter
J=1 #for models the J parameter
g=J #g parameter of Ising model


##Find POVMS by choosing model=30, works only for do_SDP=True
##parameters for model=30 POVM
slack_overlap=0 ##slack overlap for POVM 
fix_overlap=-1 ## set to values between 0 to 1 to fix overlap between states to be discriminated, set to -1 to simply choose random states for state discriminiation
n_params=1 ##number of parameters for scanning over overlap between states to be discriminated
parameterlist=np.linspace(0,1,num=n_params) ##parameters for fix_overlap, only used when n_params>1
n_POVM=2# number of POVMS for model=30 to discriminate, so far only =2 implemented



##do conserved number calculation for Hamiltonians, only works for SDP
get_conserved_number=False #whether to try to get particular conservation number
conserved_number=0 # get lowest energy eigenstate with conservation of number of particles in SDP


hilbertspace=2**n_qubits
##range of expansion


max_expansion=2## how many orders of K-moment expansion should be performed
max_states=0 #maximal number of ansatz states, set to zero to get maximal possible number given max_expansion
expand_elements=np.arange(1,max_states)

if(model==30): #only for POVM, number of random Paulis for expansion
    n_pauli_sample=100 ##number of pauli to sample for POVM basis, needs to be larger than max(expand_elements)
    expand_elements=[6]




inv_cond=1e-10 #conditioning factor for inversion and QAE, increase if norm is diverging 


levels=2




if(do_SDP==True or model==30):
    do_SDP=True
    import matlab.engine
    matlab_engine = matlab.engine.start_matlab()


qt_dims=[[2 for i in range(n_qubits)],[2 for i in range(n_qubits)]]

anneal_time_opt=0 #to initiliase variable for annealing ansatz
    

#define operators
opZ=[genFockOp(qt.sigmaz(),i,n_qubits,levels) for i in range(n_qubits)]
opX=[genFockOp(qt.sigmax(),i,n_qubits,levels) for i in range(n_qubits)]
opY=[genFockOp(qt.sigmay(),i,n_qubits,levels) for i in range(n_qubits)]

opId=genFockOp(qt.qeye(levels),0,n_qubits,levels)


if(n_qubits>1):
    opcsign=[qt.qip.operations.csign(n_qubits,i,(i+1)%n_qubits) for i in range(n_qubits)]

##whether to use sparse solver to get ground state as reference
if(n_qubits>10):
    sparse_gs=True
else:
    sparse_gs=False


target_op=[]



if(model==0 or model==1 or model==14): #get Hamiltonian 
    Hstrings,HpauliFactor,Hvalues,Hoffset=get_Hamiltonian_string(n_qubits,model,J,h)
    H=0
    for i in range(len(Hvalues)):
        H+=Hvalues[i]*get_pauli_op(Hstrings[i])
    H+=Hoffset
    target_op=[H]

    groundstate_energy,states_pure=H.groundstate(sparse=sparse_gs)
    print("Ground state energy",groundstate_energy)
    


elif(model==30):##for POVM discrimination get operators
    # ##target_op## plays role of density matrix to be measured here
    # for k in range(n_POVM): #generate states for POVM 
    #     target_dens_matrix=1/hilbertspace*opId+opX[k]
    
    #     target_op.append(target_dens_matrix)
    
    ##make sure dens matrix well defined by being trace=1 and trace(rho^2)<=1 (e.g. 1/hilbertspace before opId and )
    target_dens_matrix=1/hilbertspace*(opId+opX[0])
    target_op.append(target_dens_matrix)
    
    target_dens_matrix=1/hilbertspace*(opId+opZ[0])
    target_op.append(target_dens_matrix)
    
n_target_op=len(target_op)



##calculates ground state energy for models except POVM
if(model==30):
    pass
else:
    #only when there are conserved numbers, 
    if(model==1 and get_conserved_number==True): #excitation number
        H_conserved=sum([opZ[i] for i in range(n_qubits)])
        H_sq_conserved=H_conserved*H_conserved
        n_target_op=3
        target_op=[H,H_conserved,H_sq_conserved]
        
        
        ##get ground state of symmetry sector by shifting all eigenvalues in other symmetry sectors by a large margin
        conserved_eigenvalues,conserved_eigenstates=np.linalg.eigh(H_conserved.data.toarray())
        
        modified_eigvals=np.abs(conserved_eigenvalues-conserved_number)*10**4
        

        #Hamiltonian where unfitting symmetries are shifted by a large margin
        shifted_conserved_Hamiltonian=H+qt.Qobj(np.dot(conserved_eigenstates,np.dot(np.diag(modified_eigvals),np.transpose(np.conjugate(conserved_eigenstates)))),dims=qt_dims)
        

        
        groundstate_energy,states_pure=shifted_conserved_Hamiltonian.groundstate()
        print("Ground state energy in symmetry sector",groundstate_energy)


    elif(model==0 and get_conserved_number==True): #parity
        H_conserved=prod([opX[i] for i in range(n_qubits)])
        H_sq_conserved=H_conserved*H_conserved
        n_target_op=3
        target_op=[H,H_conserved,H_sq_conserved]
        
        conserved_eigenvalues,conserved_eigenstates=np.linalg.eigh(H_conserved.data.toarray())
        
        modified_eigvals=np.abs(conserved_eigenvalues-conserved_number)*10**4
        

        #Hamiltonian where unfitting symmetries are shifted by a large margin
        shifted_conserved_Hamiltonian=H+qt.Qobj(np.dot(conserved_eigenstates,np.dot(np.diag(modified_eigvals),np.transpose(np.conjugate(conserved_eigenstates)))),dims=qt_dims)
        

        
        groundstate_energy,states_pure=shifted_conserved_Hamiltonian.groundstate()
        print("Ground state energy in symmetry sector",groundstate_energy)
        
        
    else:
        n_target_op=1
        get_conserved_number=False

initial_state=get_ini_state(ini_type) #get quantum state used for expansion




opt=qt.Options()#options for solver taken from qutip


#define state to do moment expansion with


##generate all possible pauli strings
n_paulis=2**n_qubits
all_pauli_list=np.zeros([n_paulis,n_qubits],dtype=int)
for k in range(n_paulis):
    all_pauli_list[k,:]=numberToBase(k,2,n_qubits)


##generate paulis for state spaces used for finding POVMs
if(model==30):
    ##randomly sample vvarious pauli operators for expansion of state

    ##all zeros is added first to make ensure base state \psi is part of expansionö
    all_pauli_list=np.concatenate((np.zeros([1,n_qubits],dtype=int),np.random.randint(0,4,size=[n_pauli_sample-1,n_qubits])))
    all_pauli_list=np.unique(all_pauli_list,axis=0) ##only take unique paulis for expansion
    np.random.shuffle(all_pauli_list[1:]) ##shuffle all except identity pauli
    


 
#define which pauli operator one wants to measure of resulting state
target_paulistrings=[]
n_target_paulistrings=0
#target_paulistrings=[np.zeros(n_qubits,dtype=int) for i in range(n_target_paulistrings)]
#target_paulistrings[0][0]=1 #0: Identity, 1: X ,2: Y, 3:Z

fidelity_list=[]
evolved_QAE_state_list=[]



E_matrix_list=[]
D_matrix_list=[]


beta_solution_list=[]

prob_meas_POVM_list=[]
overlap_states_list=[]




qae_result_list=[] ##store alpha or beta of qae

##generate K-moment expansion that serves as linear combination basis of the subspace
base_expand_strings=[np.zeros(n_qubits,dtype=int)]
if(model==0 or model==1 or model==14):


    for k in range(max_expansion):

        #do moment expansion with Hamiltonian terms.
        base_expand_strings+=list(multiply_paulis(base_expand_strings,Hstrings)[0])
        new_strings,string_index=list(np.unique(base_expand_strings,axis=0, return_index=True))
        sorted_index=np.sort(string_index)
        base_expand_strings=[base_expand_strings[k] for k in sorted_index]
        

        ####base_expand_strings=list(new_strings)
        

        if(max_states>0):
            base_expand_strings=base_expand_strings[:max_states]

    # split=2*L+1
    # split_base=list(base_expand_strings[:split])
    # base_expand_strings=base_expand_strings[split:]
    # np.random.shuffle(base_expand_strings)
    # base_expand_strings=np.array(split_base+list(base_expand_strings))
    
    # if(max_states>0):
    #     base_expand_strings=base_expand_strings[:max_states]
    
    all_pauli_list=base_expand_strings
    
    if(max_states==0):
        expand_elements=np.unique(list(np.arange(1,len(all_pauli_list)+1,2))+[len(all_pauli_list)])




expand_order=len(expand_elements)



#expand to expand_order moments
for pp in range(len(parameterlist)): #this loop is to vary overlap for POVM, does only one iteration for hamiltonian stuff
    if(len(parameterlist)>1):
        fix_overlap=parameterlist[pp]
        
    #get maximal subspace 
    expand_strings=all_pauli_list[:max(expand_elements)]
    expand_states=[]
    n_expand_states=max(expand_elements)
    for i in range(n_expand_states):
        expand_states.append(get_pauli_op(expand_strings[i])*initial_state)
    
    
        
    E_matrix_all=np.zeros([n_expand_states,n_expand_states],dtype=np.complex128)

    D_matrix_all=np.zeros([n_target_op,n_expand_states,n_expand_states],dtype=np.complex128)
    
    for m in range(n_expand_states):
        for n in range(n_expand_states):
            E_matrix_all[m,n]=expand_states[m].overlap(expand_states[n])
            for k in range(n_target_op):
                D_matrix_all[k][m,n]=(expand_states[m].overlap(target_op[k]*expand_states[n]))
    
    

        
    ##now go through subspace for a smaller amount of basis states to see convergence with increasing number of subspace states
    for p_counter,p in enumerate(expand_elements):
        expand_strings=all_pauli_list[:p]
    

        n_expand_states=len(expand_strings)
            
    
        print("Order",p, "n states",n_expand_states)
    

        E_matrix=E_matrix_all[:n_expand_states,:][:,:n_expand_states]
        D_matrix=[D_matrix_all[k][:n_expand_states,:][:,:n_expand_states] for k in range(n_target_op)]
                
                
        E_matrix_list.append(E_matrix)
        D_matrix_list.append(D_matrix)
    
        
        #adjust E_matrix to make it numerical stable for positive definite
        
        #E_matrix can be not positive definite due to numerical issues. Diagonalize matrix, and set negative eigenvalues to positiv value
        #this is only needed for QAE, not for QAS
        
        
        e_vals,e_vecs=scipy.linalg.eigh(E_matrix)
    #    e_vals_adjusted=np.array(e_vals)
    #    
    #
    #    for k in range(len(e_vals_adjusted)):
    #        if(e_vals_adjusted[k]<epsilon):
    #            e_vals_adjusted[k]=epsilon
    #    E_matrix_corrected=np.dot(e_vecs,np.dot(np.diag(e_vals_adjusted),np.transpose(np.conjugate(e_vecs))))
    #    
    #    
        
        ##calculate inverse of E
        ##inv_cond determines threshold for SVD values: below inv_cond, value is set to zero. This is to avoid problems with small eigenvalues, which can blow up with inversion
        ##E_inv=np.linalg.pinv(E_matrix,hermitian=True, rcond=inv_cond)
    
    
        
        #choose initial alpha. Here, we choose initial_state as beginning state. In principle, any alpha can be chosen. One could also use QAE to find some state
        ini_alpha=np.zeros(n_expand_states,dtype=np.complex128)
    
    
        #get closest projection of initial evolution state via QAE
        ini_matrix=D_matrix
        
    
    
        print("Start SDP")
        e_vals_cond=np.array(e_vals)
        for k in range(len(e_vals_cond)):
            if(e_vals_cond[k]<inv_cond):
                e_vals_cond[k]=0
    
        E_matrix_corrected=np.dot(e_vecs,np.dot(np.diag(e_vals_cond),np.transpose(np.conjugate(e_vecs))))
    
        E_matrix_cp=E_matrix#E_matrix_corrected
        
    
    
        if(model==30):# POVM
    
            curent_dir=os.getcwd()
            matlab_engine.cd(curent_dir)
            

            
            ##get pure states 
            states_vector=[2*np.random.rand(n_expand_states)-1 for k in range(n_POVM)]
            
            ##normalise state
            for k in range(n_POVM):
                states_vector[k]=states_vector[k]/np.sqrt(np.dot(np.conjugate(states_vector[k]),np.dot(E_matrix_cp,states_vector[k])))
            
            if(fix_overlap>=0):##adjust overlap between states to be discriminated
                if(n_POVM==2):
                    ##get orthogonal states
                    states_vector[1]=states_vector[1]-np.dot(np.conjugate(states_vector[0]),np.dot(E_matrix_cp,states_vector[1]))*states_vector[0]
                
                    states_vector[1]=states_vector[1]/np.sqrt(np.dot(np.conjugate(states_vector[1]),np.dot(E_matrix_cp,states_vector[1])))
                
                states_vector[1]=(1-fix_overlap)*states_vector[1]+fix_overlap*states_vector[0] #interpolate overlaps
                states_vector[1]=states_vector[1]/np.sqrt(np.dot(np.conjugate(states_vector[1]),np.dot(E_matrix_cp,states_vector[1])))
                
                
            ##construct matrix description
            states_matrix=[np.outer(np.conjugate(states_vector[k]),states_vector[k]) for k in range(n_POVM)]
            
            overlap_states=[[np.trace(np.dot(states_matrix[q],np.dot(E_matrix_cp,np.dot(states_matrix[k],E_matrix_cp)))) for k in range(n_POVM)] for q in range(n_POVM)]
            
            if(n_expand_states>1): #run SDP
        
                slack_overlap=float(slack_overlap)
                mat_E_matrix=matlab_engine.zeros(n_expand_states,n_expand_states,"like",1j)
                #mat_D_matrix=matlab_engine.zeros(n_POVM,n_expand_states,n_expand_states,"like",1j)
                mat_states_matrix=matlab_engine.zeros(n_expand_states,n_expand_states,n_POVM,"like",1j)
    
                for i in range(n_expand_states):
                    for j in range(n_expand_states):
                        mat_E_matrix[i][j]=E_matrix_cp[i,j]
                        for k in range(n_POVM):
                            #mat_D_matrix[k][i][j]=D_matrix[k,i,j]
                            mat_states_matrix[i][j][k]=states_matrix[k][i,j]
                        
                #a=matlab_engine.matlabSDP(1,nargout=1)
                print("Run Matlab")
                #beta_solution_matlab = matlab_engine.matlabPOVMSDP(mat_E_matrix,mat_D_matrix,slack_overlap)
                beta_solution_matlab = matlab_engine.matlabPOVM_EMatrix_SDP(mat_E_matrix,mat_states_matrix,slack_overlap)
                print("Finished Matlab")
                beta_solution=np.array(beta_solution_matlab)
                beta_solution=np.swapaxes(beta_solution,0,2)
                beta_solution=np.swapaxes(beta_solution,1,2)
                
                
                #prob_correct_meas=[np.trace(np.dot(beta_solution[:,:,k],np.dot(mat_E_matrix,np.dot(mat_states[k],mat_E_matrix)))) for k in range(n_POVM)]
                
    
                if(np.isnan(beta_solution).any()==True):
                    print("WARN: Solution has nan, replace with default")
                    beta_solution=np.zeros([n_POVM,n_expand_states,n_expand_states],dtype=np.complex128)
    
    
                ##first index is state, second index is POVM
                prob_meas_POVM=[[np.real(np.trace(np.dot(beta_solution[k,:,:],np.dot(E_matrix_cp,np.dot(states_matrix[q],E_matrix_cp))))) for k in range(n_POVM)] for q in range(n_POVM)]
     
            else:
                beta_solution=np.array([np.array([[1]]) for k in range(n_POVM)])
                prob_meas_POVM=[[1]]
                
                
            objective_val=np.sum(np.diag(prob_meas_POVM)) #sum over probabilities of POVM to find right states
            
            meas_error=np.sum(prob_meas_POVM)-objective_val
            
            print("Meas correct",objective_val,"Meas wrong",meas_error)
            
            if(n_POVM==2):
                ##exact theory for 2 pure states and unambigious discrimination (slack_overlap=0)
                theta=np.arccos(np.sqrt(np.real(overlap_states[0][1]))) ##angle between two states
                theory_obj=2*(1-np.cos(theta))
                print("Theory",theory_obj,"actual",objective_val)
            
                
            beta_solution_list.append(beta_solution)
            prob_meas_POVM_list.append(prob_meas_POVM)
            overlap_states_list.append(overlap_states)
            
            beta_qae=beta_solution
    

        if(model==0 or model==1 or model==14): #find Hamiltonian ground state
            
            ini_matrix=D_matrix[0]
            if(do_SDP==True): #do SDP to find Hamiltonian ground state
    

                shift_value=-10##shift lowest soluton to negative so we do not get zero solution by minimization
                
                
                curent_dir=os.getcwd()
                matlab_engine.cd(curent_dir)
                
                if(n_expand_states>1):
            
                    mat_E_matrix=matlab_engine.zeros(n_expand_states,n_expand_states,"like",1j)
                    mat_D_matrix=matlab_engine.zeros(n_expand_states,n_expand_states,"like",1j)
                    mat_C_matrix=matlab_engine.zeros(n_expand_states,n_expand_states,"like",1j)
                    mat_C2_matrix=matlab_engine.zeros(n_expand_states,n_expand_states,"like",1j)
                    
                    if(get_conserved_number==True):
                        for i in range(n_expand_states):
                            for j in range(n_expand_states):
                                mat_C_matrix[i][j]=D_matrix[1][i,j]
                                mat_C2_matrix[i][j]=D_matrix[2][i,j]
                        
                        
        
                    
                    for i in range(n_expand_states):
                        for j in range(n_expand_states):
                            mat_E_matrix[i][j]=E_matrix_cp[i,j]
                            mat_D_matrix[i][j]=ini_matrix[i,j]
                            
                    #a=matlab_engine.matlabSDP(1,nargout=1)
                    print("Run Matlab")
                    beta_solution_matlab = matlab_engine.matlabSDP(mat_E_matrix,mat_D_matrix,float(shift_value),get_conserved_number,float(conserved_number),mat_C_matrix,mat_C2_matrix)
                    print("Finished Matlab")
                    beta_solution=np.array(beta_solution_matlab)
                    
                    if(np.isnan(beta_solution).any()==True):
                        print("WARN: Solution has nan, replace with default")
                        beta_solution=np.zeros([n_expand_states,n_expand_states],dtype=np.complex128)
                        beta_solution[0,0]=1
                    
                else:
                    beta_solution=np.array([[1]])
                
                    
                    
                val_solution=np.real(np.trace(np.dot(beta_solution,ini_matrix))) ##remove the shift
                
                
                trace_E=np.real(np.trace(np.dot(beta_solution,E_matrix_cp)))
                purity_E=np.real(np.trace(np.dot(np.dot(beta_solution,E_matrix_cp),np.dot(beta_solution,E_matrix_cp))))
                
                print("Trace Beta*E",trace_E, "purity",purity_E)
                
                print("SDP value is", val_solution)
                
                
                beta_solution_list.append(beta_solution)
        
                beta_qae=beta_solution
                
                # eigval,eigvec=np.linalg.eigh(beta_solution)
                
                
                # print("Beta eigval",eigval)
                # ini_alpha_vec=eigvec[:,-1] #take largest eigensvalue of beta
                
                U_matrix=np.hstack([expand_states[i].data.toarray() for i in range(n_expand_states)])
                
                ini_reconstructed_density_matrix=qt.Qobj(np.dot(np.dot(U_matrix,beta_qae),np.transpose(np.conjugate(U_matrix))),dims=qt_dims)
                    
                dm_eigvals,dm_eigstates=ini_reconstructed_density_matrix.eigenstates()
                
                ##largest eigenstate of density matrix is reconstructed state
                ini_reconstructed_state=dm_eigstates[-1]
                
                
                
            else: #do generalized eigenvalue problem
                            
                #get e_matrix eigenvalues inverted, cutoff with inv_cond
                e_vals_inverted=np.array(e_vals)
            
                for k in range(len(e_vals_inverted)):
                    if(e_vals_inverted[k]<inv_cond):
                        e_vals_inverted[k]=0
                    else:
                        e_vals_inverted[k]=1/e_vals_inverted[k]
                        

                #calculate ground state via generalized eigenvalue problem D\alpha=\lambda E\alpha
                #qae_energy,qae_vectors=scipy.linalg.eigh(ini_matrix,E_matrix_corrected)
                
                ###calculate eigenvalues for E^-1*H\alpha=\lambda \alpha. DOesnt seem to work for some resason though....
                ##E_inv=np.linalg.pinv(E_matrix,hermitian=True, rcond=inv_cond)
                #qae_energy,qae_vectors=scipy.linalg.eigh(np.dot(E_inv,ini_matrix))
            
            
                #E_inv=np.linalg.pinv(E_matrix,hermitian=True, rcond=inv_cond)
            
                #convert generalized eigenvalue problem with a regular eigenvalue problem using "EIGENVALUE PROBLEMS IN STRUCTURAL MECHANICS"
                #we want to solve D\alpha=\lambda E\alpha
                #turns out this does not work well if E_matrix has near zero eigenvalues
                #instead, we turn this into regular eigenvalue problem which is more behaved
                #we diagonalize E_matrix=U*F*F*U^\dag with diagonal F
                #Then, define S=U*F, and S^-1=F^-1*U^\dag. Use conditioned eigenvalues F for this such that no negative eigenvalues appear, and for inverse large eigenvalues set to zero
                #solve S^-1*D*S^-1^\dag*a=\lambda a
                #convert \alpha=S^-1^\dag*a. This is the solution to original problem.
                #this procedure ensures that converted eigenvalue problem remains hermitian, and no other funny business happens
                s_matrix=np.dot(e_vecs,np.diag(np.sqrt(e_vals_cond)))
                s_matrix_inv=np.dot(np.diag(np.sqrt(e_vals_inverted)),np.transpose(np.conjugate(e_vecs)))
                toeigmat=np.dot(s_matrix_inv,np.dot(ini_matrix,np.transpose(np.conjugate(s_matrix_inv))))
            
                qae_energy,qae_vectors=scipy.linalg.eigh(toeigmat)
                #print(qae_energy)
                
                for k in range(len(e_vals_cond)): #go through eigenvectors, take the lowest one that has non-zero norm
                    ini_alpha_vec=qae_vectors[:,k]
                    ini_alpha_vec=np.dot(np.transpose(np.conjugate(s_matrix_inv)),ini_alpha_vec)
                
                    norm_ini_alpha=np.sqrt(np.abs(np.dot(np.transpose(np.conjugate(ini_alpha_vec)),np.dot(E_matrix,ini_alpha_vec))))
                    if(norm_ini_alpha!=0):
                        break
                    
                print("norm",norm_ini_alpha)
                ini_alpha_vec=ini_alpha_vec/norm_ini_alpha
                        
            
                qae_result_list.append(ini_alpha_vec)
                
                ini_reconstructed_state=sum([expand_states[i]*ini_alpha_vec[i] for i in range(n_expand_states)])

            
        
        #get Fidelity for finding ground state
        if(model==30):
            pass
        else:
            fidIni=np.abs(ini_reconstructed_state.overlap(states_pure))**2
            fidelity_list.append(fidIni)
            evolved_QAE_state_list.append(ini_reconstructed_state)
            print("Initial state fidelity",fidIni)
        
    
            
        
    


            
##get expectation values
if(model==30): #for POVM no expectation values 
    expectZ=[]
    expectX=[]
    expectH=[]
    expectD=[]
    expectZpure=[]
    expectXpure=[]
    expectHpure=[]
    expectDpure=[]
    norm_list=[]
else:
    #if(noisy_ini_state==0):        
    #get various expectation values
    expectZ=[[np.real(qt.expect(opZ[k],evolved_QAE_state_list[i]))  for k in range(n_qubits)] for i in range(expand_order)]
    
    expectX=[[np.real(qt.expect(opX[k],evolved_QAE_state_list[i]))  for k in range(n_qubits)]for i in range(expand_order)]
                
    
    #energy for static hamiltonian
    expectH=[np.real(qt.expect(target_op[0],evolved_QAE_state_list[i])) for i in range(expand_order)]
    
    
    #energy for static hamiltonian
    expectD=[[np.real(qt.expect(target_op[k],evolved_QAE_state_list[i])) for k in range(n_target_op)] for i in range(expand_order)]
    
    
    
    
    expectZpure=[np.real(qt.expect(opZ[k],states_pure)) for k in range(n_qubits)]
        
    expectXpure=[np.real(qt.expect(opX[k],states_pure))  for k in range(n_qubits)]
    expectHpure=np.real(qt.expect(target_op[0],states_pure))
    expectDpure=[np.real(qt.expect(target_op[k],states_pure)) for k in range(n_target_op)] 
    
    ##
    #gibbs=(-H*times[10]*2).expm()
    #gibbs=gibbs/gibbs.tr()
    #qt.fidelity(states_pure[10],gibbs)
    
    legendOrder=[]
    xlabelstring="$M$"
    
    #legendOrder=["$K="+str(i)+"$" for i in range(0,expand_order+1)]
    
    #fidleity with exact evolution
    plot1D([fidelity_list],[expand_elements],xlabelstring=xlabelstring,ylabelstring="$F$",legend=legendOrder)
    
    
    #norm of evolved state. Should remain one. If not, try increasing inv_cond 
    #norm_list=[evolved_QAE_state_list[i].norm() for i in range(expand_order+1)]

    norm_list=[]
    
    xrange=np.arange(n_qubits)

    
    
    #static energy over time
    customMarkerStyle=["o"]*2
    customMarkerStyle[-1]=""
    
    plot1D(np.transpose(expectD),[expand_elements]*n_target_op,xlabelstring=xlabelstring,ylabelstring="$\\langle D\\rangle$")
        
    
    plot1D([[np.sum([np.abs(expectD[i][k]-expectDpure[k]) for k in range(0,n_target_op)]) for i in range(expand_order)]],[expand_elements],xlabelstring=xlabelstring,ylabelstring="$\\vert\\langle C\\rangle-C_\mathrm{exact}\\vert$")
        
    
    
    legendOrder=["NISQ SDP","exact"]
    plot1D([expectH]+[[expectHpure]*(expand_order)],[expand_elements]*2,xlabelstring=xlabelstring,ylabelstring="$\\langle H\\rangle$",legend=legendOrder)
        
    


    #print("Energy NISQ SDP",expectH,"groundstate",expectHpure)


print("Total time taken",time.time()-starttime)


