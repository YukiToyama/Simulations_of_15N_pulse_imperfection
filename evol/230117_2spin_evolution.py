# -*- coding: utf-8 -*-

"""
Sometimes pulses just have to be perfect
 – an example based on the measurement of amide proton transverse relaxation rates in proteins

Atul Kaushik Rangadurai, Yuki Toyama, and Lewis E. Kay

Simulation of 1HN transverse evolution with different 15N inversion pulses. 

written by Y. Toyama

Trajectories of 1HN in-phase (Hx), 1HN-15N anti-phase (2HyNz), 
and 1HN-15N multiple quantum elements (2HyNx and 2HyNy) are simulated 
by calculating the time evolution of the density matrix of 
a 1HN-15N J-coupled two spin spin-system in Liouville space, as described by Allard et al. 
Ref: P. Allard, M. Helgstrand, T. Hard, J Magn Reson, 134 (1998) 7-16.

The relaxation matrix in this script contains the 1H CSA and 1H-15N DD/1H CSA CCR.

Set the "N_option" to simulate 4 different 15N refocusing pulses.
"hard" : rectangular 180 degree pulse
"adiabatic" : adiabatic pulse
"composite180y" : 90x-180y-90x composite pulse
"composite240y" : 90x-240y-90x composite pulse

The option "adiabatic" and "reburp" reads the shape file for Bruker spectrometers.
(The foldar contains Apod_Chirp_NHT2.BRF for adiabatic pulse and 1360us_reburp.BRF for 1H REBURP pulse.)

This script also allows you to use 1H REBURP pulse for refocusing.
Set "H_option" to "reburp".

numpy : 1.21.5
scipy: 1.7.3
matplotlib: 3.5.1

"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.linalg

###############################
## Input
###############################

setoffsetH = 0 # 1H offset in Hz
setoffsetN = 1000 # 15N offset in Hz
settmax = 27000 # Delay time T, in us
H_option = "hard" # "hard" or "reburp", otherwise pulse not applied
N_option = "hard"  # "hard", "adiabatic", "composite180y", "composite240y", otherwise pulse not applied

outname = "evol_15N_"+str(setoffsetN)+"_max_"+str(settmax)+"_"+H_option+"_"+N_option

###############################
## Define some useful functions
###############################

def Commute(A,B):
  return A@B-B@A

def func(x, R): 
    return np.exp(-1*R*x)
    

###############################
## Construction of spin system
###############################

JAB = 90 # Hz

B0  = 23.4 # Tesla

# Pauli matrices
Ix = 0.5*np.array([[0, 1],[1, 0]],dtype=complex)
Iy = 0.5*np.array([[0, -1j],[1j, 0]],dtype=complex)
Iz = 0.5*np.array([[1, 0],[0, -1]],dtype=complex)
E = np.eye(2)

# Cartesian Operators
Ax = np.kron(Ix,E)
Ay = np.kron(Iy,E)
Az = np.kron(Iz,E)

Bx = np.kron(E,Ix)
By = np.kron(E,Iy)
Bz = np.kron(E,Iz) 

AxBz = Ax@Bz
AyBz = Ay@Bz
AzBx = Az@Bx
AzBy = Az@By

AxBx = Ax@Bx
AxBy = Ax@By
AyBx = Ay@Bx
AyBy = Ay@By

AzBz = Az@Bz

E2 =  np.kron(E,E)


##############################
# Construction of Liouvillian
##############################

basis =  [Ax,Ay,Az,Bx,By,Bz,2*AxBz,2*AyBz,2*AzBx,2*AzBy,2*AxBx,2*AxBy,2*AyBx,2*AyBy,2*AzBz]


# Liouvillian for evolution, with arbitrary power, phase, and offset on B
# w1, phase, and offset are in Hz
# Note that the factor of (-i) is already included in the function.

def L(w1A,phaseA,offsetA,w1B,phaseB,offsetB):
    HRF = w1A*2*np.pi*Ax*np.cos(phaseA*np.pi/180) + w1A*2*np.pi*Ay*np.sin(phaseA*np.pi/180) + w1B*2*np.pi*Bx*np.cos(phaseB*np.pi/180) + w1B*2*np.pi*By*np.sin(phaseB*np.pi/180)
    HCS = offsetA*2*np.pi*Az + offsetB*2*np.pi*Bz 
    HJ =  2*np.pi*JAB*AzBz 
    
    L = np.zeros([len(basis),len(basis)],dtype=complex)
    for i in range(len(basis)):
      for k in range(len(basis)):
        L[i,k] = -1j*(basis[i]@Commute(HRF+HCS+HJ,basis[k])).trace()/(basis[i]@basis[i]).trace()
    return L

#####################
# Relaxation rate
#####################

## Constants, here I assume A is proton and B is nitrogen.
hbar= 6.626E-34/2/np.pi
gH = 2.67522E8 # rad s-1 T-1
gC = 6.728E7 # rad s-1 T-1
gN = -2.713E7 # rad s-1 T-1

## CSA assuming axial symmetric tensor
deltasigmaH = -9.9E-6
deltasigmaN = -161E-6

thetaA = 7*np.pi/180 # NH - 1H CSA PP axis in radian (7 deg)
thetaB = 19*np.pi/180 # NH - 15N CSA PP axis in radian (19 deg)

## Larmor frequencies
wA = B0*gH
wB = B0*gN

# Dipolar interaction
rHN = 1.04E-10 # m

# Rotational correlation time
tauc = 5E-9 # sec

## Interaction strength following Palmer's notation
Ad =  -1*np.sqrt(6)*1E-7*(hbar*gH*gN/rHN**3) #N-H dipolar interaction
Ac_A =  np.sqrt(2/3)*deltasigmaH*gH*B0 # 1H CSA interaction
Ac_B =  np.sqrt(2/3)*deltasigmaN*gN*B0 # 15N CSA interaction

## Spectral density assuming isotropic rotational motion
def J(tauc,w):
    return 2/5*tauc/(1+w**2*tauc**2)

## Spectral density function for DD or CSA auto-relaxation
J_0 =  J(tauc,0)
J_wA =  J(tauc,wA)
J_wB =  J(tauc,wB)
J_wApwB =  J(tauc,wA+wB)
J_wAmwB =  J(tauc,wA-wB)

## Spectral density function for AB DD- A CSA cross-correlated relaxation
KABA_0 =  J(tauc,0)*(3*np.cos(thetaA)**2-1)/2
KABA_wA =  J(tauc,wA)*(3*np.cos(thetaA)**2-1)/2

## Spectral density function for AB DD- B CSA cross-correlated relaxation
KABB_0 =  J(tauc,0)*(3*np.cos(thetaB)**2-1)/2
KABB_wB =  J(tauc,wB)*(3*np.cos(thetaB)**2-1)/2

## Spectral density function for A CSA- B CSA cross-correlated relaxation
## Not considered in the calculation though CSA-CSA cross-correlation does exist. 
KAB_0 = 0 

## Relaxation matrix
Gamma = np.zeros([15,15])
Gamma[0, 0] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ad**2*J_0/12 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/8
Gamma[0, 6] = Ac_A*Ad*KABA_0/3 + Ac_A*Ad*KABA_wA/4
Gamma[1, 1] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ad**2*J_0/12 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/8 
Gamma[1, 7] = Ac_A*Ad*KABA_0/3 + Ac_A*Ad*KABA_wA/4
Gamma[2, 2] = Ac_A**2*J_wA/2 + Ad**2*J_wA/8 + Ad**2*J_wAmwB/24 + Ad**2*J_wApwB/4
Gamma[2, 5] = -Ad**2*J_wAmwB/24 + Ad**2*J_wApwB/4
Gamma[2, 14] = Ac_A*Ad*KABA_wA/2
Gamma[3, 3] = Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_0/12 + Ad**2*J_wA/8 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16
Gamma[3, 8] = Ac_B*Ad*KABB_0/3 + Ac_B*Ad*KABB_wB/4
Gamma[4, 4] = Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_0/12 + Ad**2*J_wA/8 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16
Gamma[4, 9] = Ac_B*Ad*KABB_0/3 + Ac_B*Ad*KABB_wB/4
Gamma[5, 2] = -Ad**2*J_wAmwB/24 + Ad**2*J_wApwB/4
Gamma[5, 5] = Ac_B**2*J_wB/2 + Ad**2*J_wAmwB/24 + Ad**2*J_wApwB/4 + Ad**2*J_wB/8
Gamma[5, 14] = Ac_B*Ad*KABB_wB/2
Gamma[6, 0] = Ac_A*Ad*KABA_0/3 + Ac_A*Ad*KABA_wA/4
Gamma[6, 6] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ac_B**2*J_wB/2 + Ad**2*J_0/12 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 
Gamma[7, 1] = Ac_A*Ad*KABA_0/3 + Ac_A*Ad*KABA_wA/4
Gamma[7, 7] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ac_B**2*J_wB/2 + Ad**2*J_0/12 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 
Gamma[8, 3] = Ac_B*Ad*KABB_0/3 + Ac_B*Ad*KABB_wB/4
Gamma[8, 8] = Ac_A**2*J_wA/2 + Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_0/12 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16
Gamma[9, 4] = Ac_B*Ad*KABB_0/3 + Ac_B*Ad*KABB_wB/4
Gamma[9, 9] = Ac_A**2*J_wA/2 + Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_0/12 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16
Gamma[10, 10] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16 
Gamma[10, 13] = -2*Ac_A*Ac_B*KAB_0/3 + Ad**2*J_wAmwB/48 - Ad**2*J_wApwB/8
Gamma[11, 11] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16
Gamma[11, 12] = 2*Ac_A*Ac_B*KAB_0/3 - Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8
Gamma[12, 11] = 2*Ac_A*Ac_B*KAB_0/3 - Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8
Gamma[12, 12] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16 
Gamma[13, 10] = -2*Ac_A*Ac_B*KAB_0/3 + Ad**2*J_wAmwB/48 - Ad**2*J_wApwB/8
Gamma[13, 13] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16
Gamma[14, 2] = Ac_A*Ad*KABA_wA/2
Gamma[14, 5] = Ac_B*Ad*KABB_wB/2
Gamma[14, 14] = Ac_A**2*J_wA/2 + Ac_B**2*J_wB/2 + Ad**2*J_wA/8 + Ad**2*J_wB/8


#################
## reburp  ####
#################

reburp = np.loadtxt("1360us_reburp.BRF",delimiter=",",dtype=float)
reburp_ss = 1 # stepsize in us
reburp_tp = 1360 # total length of the pulse in us
reburp_w1 = 4600 # amplitude in Hz

#################
## Adiabatic  ####
#################

chirp = np.loadtxt("Apod_Chirp_NHT2.BRF",delimiter=",",dtype=float)
chirp_ss = 1 # stepsize in us
chirp_tp = 1000 # total length of the pulse in us
chirp_w1 = 5000 # amplitude in Hz


#####################
# Spin evolution
#####################

## Starting fom x magnetization of A (Ax)
initial = np.zeros(15)
initial[0] = 1

w1A = 25000
p90_A = 0.25/w1A
offsetA = setoffsetH

w1B = 6250
p90_B = 0.25/w1B
offsetB = setoffsetN

## Calculate the evolution
timestep = 1 # in us
tmax = settmax # in us
tp_A = p90_A*1E6
tp_B = p90_B*1E6

# in-phase, anti-phase, and MQ terms are monitored.
t = np.zeros(0)
Hx = np.zeros(0)
HyNz = np.zeros(0)
HxNx = np.zeros(0)
HxNy = np.zeros(0)
HyNx = np.zeros(0)
HyNy = np.zeros(0)

# Initial
count = 0
rho = initial

Hx = np.append(Hx,rho[0])
HyNz = np.append(HyNz,rho[7])
HxNx = np.append(HxNx,rho[10])
HxNy = np.append(HxNy,rho[11])
HyNx = np.append(HyNx,rho[12])
HyNy = np.append(HyNy,rho[13])
t = np.append(t,count)

# Trelax/4
Ltemp = L(0,0,offsetA,0,0,offsetB)-Gamma
for i in range(int(0.25*tmax/timestep)):
    rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
    Hx = np.append(Hx,rho[0])
    HyNz = np.append(HyNz,rho[7])
    HxNx = np.append(HxNx,rho[10])
    HxNy = np.append(HxNy,rho[11])
    HyNx = np.append(HyNx,rho[12])
    HyNy = np.append(HyNy,rho[13])
    count += timestep
    t = np.append(t,count)
    

# 15N refocus
if N_option == "hard":
    Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
    for i in range(int(2*tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)

elif N_option == "adiabatic":
    for i in range(int(chirp_tp/chirp_ss)):
        Ltemp = L(0,0,offsetA,chirp_w1*chirp[i,0]/100,chirp[i,1],offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*chirp_ss*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += chirp_ss
        t = np.append(t,count)

elif N_option == "composite180y":
    Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
    for i in range(int(tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)
    Ltemp = L(0,0,offsetA,w1B,90,offsetB)-Gamma
    for i in range(int(2*tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)
    Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
    for i in range(int(tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)

elif N_option == "composite240y":
    Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
    for i in range(int(tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)
    Ltemp = L(0,0,offsetA,w1B,90,offsetB)-Gamma
    for i in range(int(24/9*tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)
    Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
    for i in range(int(tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)
        
     
# Trelax/4 
Ltemp = L(0,0,offsetA,0,0,offsetB)-Gamma
for i in range(int(0.25*tmax/timestep)):
    rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
    Hx = np.append(Hx,rho[0])
    HyNz = np.append(HyNz,rho[7])
    HxNx = np.append(HxNx,rho[10])
    HxNy = np.append(HxNy,rho[11])
    HyNx = np.append(HyNx,rho[12])
    HyNy = np.append(HyNy,rho[13])
    count += timestep
    t = np.append(t,count)
    

# 1H refocus
if H_option == "hard":
    Ltemp = L(w1A,0,offsetA,0,0,offsetB)-Gamma
    for i in range(int(2*tp_A/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)

elif H_option == "reburp":
    for i in range(int(reburp_tp/reburp_ss)):
        Ltemp = L(reburp_w1*reburp[i,0]/100,reburp[i,1],offsetA,0,0,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*reburp_ss*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += reburp_ss
        t = np.append(t,count)
 
# Trelax/4 
Ltemp = L(0,0,offsetA,0,0,offsetB)-Gamma
for i in range(int(0.25*tmax/timestep)):
    rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
    Hx = np.append(Hx,rho[0])
    HyNz = np.append(HyNz,rho[7])
    HxNx = np.append(HxNx,rho[10])
    HxNy = np.append(HxNy,rho[11])
    HyNx = np.append(HyNx,rho[12])
    HyNy = np.append(HyNy,rho[13])
    count += timestep
    t = np.append(t,count)
    
# 15N refocus
if N_option == "hard":
    Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
    for i in range(int(2*tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)

elif N_option == "adiabatic":
    for i in range(int(chirp_tp/chirp_ss)):
        Ltemp = L(0,0,offsetA,chirp_w1*chirp[i,0]/100,chirp[i,1],offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*chirp_ss*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += chirp_ss
        t = np.append(t,count)

elif N_option == "composite180y":
    Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
    for i in range(int(tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)
    Ltemp = L(0,0,offsetA,w1B,90,offsetB)-Gamma
    for i in range(int(2*tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)
    Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
    for i in range(int(tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)

elif N_option == "composite240y":
    Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
    for i in range(int(tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)
    Ltemp = L(0,0,offsetA,w1B,90,offsetB)-Gamma
    for i in range(int(24/9*tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)
    Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
    for i in range(int(tp_B/timestep)):
        rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
        Hx = np.append(Hx,rho[0])
        HyNz = np.append(HyNz,rho[7])
        HxNx = np.append(HxNx,rho[10])
        HxNy = np.append(HxNy,rho[11])
        HyNx = np.append(HyNx,rho[12])
        HyNy = np.append(HyNy,rho[13])
        count += timestep
        t = np.append(t,count)
        
        
# Trelax/4
Ltemp = L(0,0,offsetA,0,0,offsetB)-Gamma
for i in range(int(0.25*tmax/timestep)):
    rho = sp.linalg.expm(1E-6*timestep*Ltemp) @ rho
    Hx = np.append(Hx,rho[0])
    HyNz = np.append(HyNz,rho[7])
    HxNx = np.append(HxNx,rho[10])
    HxNy = np.append(HxNy,rho[11])
    HyNx = np.append(HyNx,rho[12])
    HyNy = np.append(HyNy,rho[13])
    count += timestep
    t = np.append(t,count)
    

#####################
# Output
#####################

fig1 = plt.figure(figsize=(4,3),dpi=300)
ax1 = fig1.add_subplot(111)

ax1.plot(1e-3*t,HyNy,linewidth=1, color = 'dodgerblue',label='2HyNy',alpha=1)
ax1.plot(1e-3*t,HyNx,linewidth=1, color = 'palegreen',label='2HyNx',alpha=1)
ax1.plot(1e-3*t,Hx,linewidth=1, color = 'black',label='Hx')
ax1.plot(1e-3*t,HyNz,linewidth=1, color = 'tomato',label='2HyNz')

ax1.set_ylabel('Intensity',fontsize=5)
ax1.set_xlabel('Relaxation delay (ms)',fontsize=5)
ax1.tick_params(direction='out',axis='both',length=1.5,width=0.5,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
ax1.grid(linestyle='dashed',linewidth=1,dash_capstyle='round',dashes=(1,3))
ax1.set_ylim(-1.05,1.05)
ax1.set_xlim(0,1.1*settmax/1000)

if N_option == "adiabatic":
    Npow = chirp_w1
else:
    Npow = w1B

ax1.set_title("$^1H$ refocus = "+str(H_option)+", $^{15}N$ refocus = " +str(N_option)+
              "\n$\\Omega_{N}$ = " + str(offsetB) + " (Hz), $\omega_{N,RF}$ = " + str(Npow) + " (Hz)"
              ,fontsize=7)

ax1.spines['top'].set_linewidth(0.)
ax1.spines['right'].set_linewidth(0.)
ax1.spines['left'].set_linewidth(0.75)
ax1.spines['bottom'].set_linewidth(0.75)
ax1.legend(fontsize=6, loc='lower left')
plt.locator_params(axis='y',nbins=6)
plt.tight_layout()

plt.savefig(outname+".pdf")
