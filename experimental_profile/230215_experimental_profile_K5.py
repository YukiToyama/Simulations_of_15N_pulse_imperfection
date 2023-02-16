# -*- coding: utf-8 -*-

"""
Sometimes pulses just have to be perfect
 – an example based on the measurement of amide proton transverse relaxation rates in proteins

Atul Kaushik Rangadurai, Yuki Toyama, and Lewis E. Kay

Simulation of the decay curve of 1HN in-phase transverse magnetization
with rectangular 15N inversion pulses to compare with the experimental result. 
(Supplementary figure 2)

written by Y. Toyama

Decay curves of 1HN transverse magnetization are simulated 
by calculating the time evolution of the density matrix of 
a 1HN-15N J-coupled two spin spin-system in Liouville space, as described by Allard et al. 
Ref: P. Allard, M. Helgstrand, T. Hard, J Magn Reson, 134 (1998) 7-16.

In order to compare with the experient, the simulation starts from in-phase Hx 
and detect anti-phase 2HyNz, as the relaxation period incoporates the INEPT transfer
as shown in Fig. 1A.

The relaxation matrix in this script contains the 1H CSA and 1H-15N DD/1H CSA CCR.
To reproduce the experimental result, an addtional term that takes into account
the cntributions of DD interactions involving  external 1H spins and solvent exchange 
is added to the auto relaxation rate of the transverse 1H components.
This term is equivalent to lamdaH in Allard's paper.

As in the experiment, this script uses phase-shifted 1H REBURP pulse for inversion.
The carrier is assumed to be on resonance to water, and the REBURP pulse is applied at
the center of the amide region.
"H_option" is set to "reburp".


Set the "N_option" to simulate 4 different 15N refocusing pulses.
"hard" : rectangular 180 degree pulse
"adiabatic" : adiabatic pulse
"composite180y" : 90x-180y-90x composite pulse
"composite240y" : 90x-240y-90x composite pulse

The option "adiabatic" and "reburp" reads the shape file for Bruker spectrometers.
(The foldar contains Apod_Chirp_NHT2.BRF for adiabatic pulse and 1360us_reburp.BRF for 1H REBURP pulse.)


numpy : 1.21.5
scipy: 1.7.3
matplotlib: 3.5.1

"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.optimize import curve_fit  

###############################
## Input
###############################

setoffsetN = 820.545   # 15N offset in Hz
setoffsetH = -3859.110 # 1H offset in Hz (from water, -1 multiplied to be consistent with Bruker spectrometer)
H_option = "reburp" # "hard" or "reburp"
N_option = "hard"  # "hard", "adiabatic", "composite180y", "composite240y"

# Delay time points need to be set in the separate text file (in sec).
vd = np.loadtxt("vdlist")

taua = 2400 # in us, 1/4J
outname = "K5_decay_15N_"+str(setoffsetN)+"_1H_"+str(setoffsetH)+"_"+H_option+"_"+N_option


###############################
## Define some useful functions
###############################

def Commute(A,B):
  return A@B-B@A

def func(x, I0, R):
    return I0*np.exp(-1*R*x)
      

###############################
## Construction of spin system
###############################

JAB = 90 #Hz

B0  = 23.4

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
Ac_A =  np.sqrt(2/3)*deltasigmaH*gH*B0 # 1H CSA
Ac_B =  np.sqrt(2/3)*deltasigmaN*gN*B0 # 15N CSA

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
## Not included in the calculation though CSA-CSA cross-correlation does exist. 
KAB_0 = 0 

## Contributions of DD interactions involving external 1H spins
## and solvent exchange on transverse relaxation rates (in s-1).
## This term is equivalent to lamdaH in Allard's paper.
lambda_H = 2.4

## Relaxation matrix

Gamma = np.zeros([15,15])
Gamma[0, 0] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ad**2*J_0/12 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/8 + lambda_H
Gamma[0, 6] = Ac_A*Ad*KABA_0/3 + Ac_A*Ad*KABA_wA/4
Gamma[1, 1] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ad**2*J_0/12 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/8 + lambda_H
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
Gamma[6, 6] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ac_B**2*J_wB/2 + Ad**2*J_0/12 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + lambda_H
Gamma[7, 1] = Ac_A*Ad*KABA_0/3 + Ac_A*Ad*KABA_wA/4
Gamma[7, 7] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ac_B**2*J_wB/2 + Ad**2*J_0/12 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + lambda_H 
Gamma[8, 3] = Ac_B*Ad*KABB_0/3 + Ac_B*Ad*KABB_wB/4
Gamma[8, 8] = Ac_A**2*J_wA/2 + Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_0/12 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16
Gamma[9, 4] = Ac_B*Ad*KABB_0/3 + Ac_B*Ad*KABB_wB/4
Gamma[9, 9] = Ac_A**2*J_wA/2 + Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_0/12 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16
Gamma[10, 10] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16 + lambda_H 
Gamma[10, 13] = -2*Ac_A*Ac_B*KAB_0/3 + Ad**2*J_wAmwB/48 - Ad**2*J_wApwB/8
Gamma[11, 11] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16 + lambda_H
Gamma[11, 12] = 2*Ac_A*Ac_B*KAB_0/3 - Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8
Gamma[12, 11] = 2*Ac_A*Ac_B*KAB_0/3 - Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8
Gamma[12, 12] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16 + lambda_H 
Gamma[13, 10] = -2*Ac_A*Ac_B*KAB_0/3 + Ad**2*J_wAmwB/48 - Ad**2*J_wApwB/8
Gamma[13, 13] = Ac_A**2*J_0/3 + Ac_A**2*J_wA/4 + Ac_B**2*J_0/3 + Ac_B**2*J_wB/4 + Ad**2*J_wA/16 + Ad**2*J_wAmwB/48 + Ad**2*J_wApwB/8 + Ad**2*J_wB/16 + lambda_H
Gamma[14, 2] = Ac_A*Ad*KABA_wA/2
Gamma[14, 5] = Ac_B*Ad*KABB_wB/2
Gamma[14, 14] = Ac_A**2*J_wA/2 + Ac_B**2*J_wB/2 + Ad**2*J_wA/8 + Ad**2*J_wB/8


#################
## reburp  ####
#################

reburp = np.loadtxt("reb_7p95.BRF",delimiter=",",dtype=float)
reburp_ss = 1 # in us
reburp_tp = 1360 # in us, This is the total length of the pulse
reburp_w1 = 4600 # in Hz

#################
## Adiabatic  ####
#################

chirp = np.loadtxt("Apod_Chirp_NHT2.BRF",delimiter=",",dtype=float)
chirp_ss = 1 # in us
chirp_tp = 1000 # in us, This is the total length of the pulse
chirp_w1 = 5000 # in Hz


#####################
# Spin evolution
#####################

t = vd # in sec
HyNz = np.zeros(len(t))

## Starting fom x magnetization of A (Ax)
initial = np.zeros(15)
initial[0] = 1

w1A = 25000
p90_A = 0.25/w1A
offsetA = setoffsetH

w1B = 5952
p90_B = 0.25/w1B
offsetB = setoffsetN

## Calculate the evolution
tp_A = p90_A*1E6
tp_B = p90_B*1E6


for s in range(len(t)):
    # Initial
    count = 0
    rho = initial
    tmax = t[s]*1E6 # in us

    # Trelax/4 + tau/2
    Ltemp = L(0,0,offsetA,0,0,offsetB)-Gamma
    rho = sp.linalg.expm(1E-6*(tmax/4+taua/2)*Ltemp) @ rho   
    
    # 15N refocus
    if N_option == "hard":
        Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*2*tp_B*Ltemp) @ rho
    
    elif N_option == "adiabatic":
        for i in range(int(chirp_tp/chirp_ss)):
            Ltemp = L(0,0,offsetA,chirp_w1*chirp[i,0]/100,chirp[i,1],offsetB)-Gamma
            rho = sp.linalg.expm(1E-6*chirp_ss*Ltemp) @ rho
    
    elif N_option == "composite180y":
        Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*tp_B*Ltemp) @ rho
        Ltemp = L(0,0,offsetA,w1B,90,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*2*tp_B*Ltemp) @ rho
        Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*tp_B*Ltemp) @ rho
    
    elif N_option == "composite240y":
        Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*tp_B*Ltemp) @ rho
        Ltemp = L(0,0,offsetA,w1B,90,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*24/9*tp_B*Ltemp) @ rho
        Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*tp_B*Ltemp) @ rho
            
         
    # Trelax/4 - taua/2
    Ltemp = L(0,0,offsetA,0,0,offsetB)-Gamma
    rho = sp.linalg.expm(1E-6*(tmax/4-taua/2)*Ltemp) @ rho   
        
    
    # 1H refocus
    if H_option == "hard":
        Ltemp = L(w1A,0,offsetA,0,0,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*2*tp_A*Ltemp) @ rho
        
    elif H_option == "reburp":
        for i in range(int(reburp_tp/reburp_ss)):
            Ltemp = L(reburp_w1*reburp[i,0]/100,reburp[i,1],offsetA,0,0,offsetB)-Gamma
            rho = sp.linalg.expm(1E-6*reburp_ss*Ltemp) @ rho
    
    # Trelax/4 + taua/2
    Ltemp = L(0,0,offsetA,0,0,offsetB)-Gamma
    rho = sp.linalg.expm(1E-6*(tmax/4+taua/2)*Ltemp) @ rho   
    
    # 15N refocus
    if N_option == "hard":
        Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*2*tp_B*Ltemp) @ rho
  
    elif N_option == "adiabatic":
        for i in range(int(chirp_tp/chirp_ss)):
            Ltemp = L(0,0,offsetA,chirp_w1*chirp[i,0]/100,chirp[i,1],offsetB)-Gamma
            rho = sp.linalg.expm(1E-6*chirp_ss*Ltemp) @ rho
    
    elif N_option == "composite180y":
        Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*tp_B*Ltemp) @ rho
        Ltemp = L(0,0,offsetA,w1B,90,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*2*tp_B*Ltemp) @ rho
        Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*tp_B*Ltemp) @ rho
    
    elif N_option == "composite240y":
        Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*tp_B*Ltemp) @ rho
        Ltemp = L(0,0,offsetA,w1B,90,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*24/9*tp_B*Ltemp) @ rho
        Ltemp = L(0,0,offsetA,w1B,0,offsetB)-Gamma
        rho = sp.linalg.expm(1E-6*tp_B*Ltemp) @ rho
    
    # Trelax/4 - taua/2
    Ltemp = L(0,0,offsetA,0,0,offsetB)-Gamma
    rho = sp.linalg.expm(1E-6*(tmax/4-taua/2)*Ltemp) @ rho   

    # Detection
    HyNz[s] = -1*rho[7]


fit, cov = curve_fit(func,t,HyNz)
tsim = np.linspace(0,np.max(t),100)

I0 = fit[0]
Rfit = fit[1]


#####################
# Output
#####################

fig1 = plt.figure(figsize=(2.4,2.4),dpi=300)
ax1 = fig1.add_subplot(111)

ax1.scatter(1000*t,HyNz/I0, s=4, marker='o',facecolor="black", color = 'black',label='$-2H_yN_z$')
ax1.plot(1000*tsim,func(tsim,I0,Rfit)/I0,linewidth=0.5, color = 'black',label='exp(-Rt)')

ax1.set_ylabel('Normalized $2H_yN_z$ ',fontsize=6)
ax1.set_xlabel('Relaxation time (ms)',fontsize=6)
ax1.tick_params(direction='out',axis='both',length=2,width=1,grid_alpha=0.3,bottom=True,top=False,left=True,right=False,labelsize=6)
ax1.grid(linestyle='dashed',linewidth=1,dash_capstyle='round',dashes=(1,3))

if N_option == "adiabatic":
    Npow = chirp_w1
else:
    Npow = w1B

ax1.set_title("$^1H$ refocus = "+str(H_option)+", $^{15}N$ refocus = " +str(N_option)+
              "\n$\\Omega_{N}$ = " + str(offsetB) +", $\\Omega_{H}$ = " + str(offsetA) + " (Hz)"+
              "\n$\omega_{N,RF}$ = " + str(Npow) + " (Hz)"
              ,fontsize=6)

ax1.text(0.03,0.15,"$R_{fit}$ = "+str(round(Rfit,2)),transform=ax1.transAxes,va="top",fontsize=8,color='black')

ax1.spines['top'].set_linewidth(0.)
ax1.spines['right'].set_linewidth(0.)
ax1.spines['left'].set_linewidth(0.75)
ax1.spines['bottom'].set_linewidth(0.75)
ax1.set_xlim(0,1.1*np.max(vd)*1000)
plt.tight_layout()

plt.tight_layout()


plt.savefig(outname+".pdf")
