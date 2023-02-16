# -*- coding: utf-8 -*-
"""
This script allows you to create an adiabatic inversion pulse.
Based on Lewis's ccode "BRF_apod_chirp.c" to generate an adiabatic pulse. 

In the Bruker format, the pulse phase is defined in the unit of degree.
The phase is incremented by 1/2*k*t**2 (Kupce and Freeman, JMR A 115 273-276, 1995)
The amplitude is 

w1(n) = w1max*sin{(n*pi)/(2f)}              1<n<f
      = w1max                               f<n<np-f
      = w1max*sin(pi/2+pi/2*[n-(np-f)]/f)   np-f<n<np

f = iedge 
np-f = idelta

(Zwahlen et al., J. Am. Chem. Soc., Vol. 119, No. 29, 1997)

written by Y. Toyama

"""

import math

### Parameters ####

outname = "Apod_Chirp_NHT2.BRF" # Output file name
Pulse = 1000E-6 # total pulse duration in sec
Stepsize = 1E-6 # time step in sec
Offset = -20000 # in Hz
Sweep = 40000 # frequency sweep in Hz
fract = 20 # Truncation level in percentage, typically 20
phase0 = 0 # initial phase

#####################

Sweep = -1*Sweep # for Bruker
Offset = -1*Offset # for Bruker

Number_Steps = Pulse/Stepsize # the number of steps
phase_Step = 360*Stepsize*Offset

iedge = Number_Steps*fract/100 # the first edge to ramp up the amplitude
idelta = Number_Steps - iedge # the last edge to ramp down the amplitude

k = Sweep/Pulse # This is the phase sweep rate.

pi = 3.141592

#######################
# Write headers for Bruker shape file
#######################

f = open(outname,'w')
f.write("##TITLE= Apod_Chirp\n")
f.write("##JCAMP-DX= 5.00 Bruker JCAMP library\n")
f.write("##DATA TYPE= Shape Data\n")
f.write("##ORIGIN= NMR Centre, UofT\n")
f.write("##OWNER= <nmrsu>\n")
f.write("##DATE= 2013/10/02\n")
f.write("##TIME= 12:00:00\n")
f.write("##$SHAPE_PARAMETERS= Type: null\n")
f.write("##MINX= 0.000000E00\n")
f.write("##MAXX= 1.000000E02\n")
f.write("##MINY= 0.000000E00\n")
f.write("##MAXY= 1.800000E02\n")
f.write("##$SHAPE_EXMODE= None\n")
f.write("##$SHAPE_TOTROT= 1.800000E02\n")
f.write("##$SHAPE_TYPE= Inversion\n")
f.write("##$SHAPE_USER_DEF=\n")
f.write("##$SHAPE_REPHFAC= \n")
f.write("##$SHAPE_BWFAC= 0.000000E00\n")
f.write("##$SHAPE_BWFAC50= \n")
f.write("##$SHAPE_INTEGFAC= 2.097446E-04\n")
f.write("##$SHAPE_MODE= 0\n")
f.write("##NPOINTS= 1000\n")
f.write("##XYPOINTS= (XY..XY)\n")

#######################
# Calculate and add the amplitude and phase
#######################

for i in range(int(Number_Steps)):

  t = i*Stepsize
  
  ### Phase ###
  # The constant phase to shift the frequency. phase = 0 when on resonance.
  phase =  i*phase_Step
  
  # Add the phase shift 
  phase += 360*0.5*k*t**2
  
  # Add the initial phase
  phase += phase0  
  
  phase_b = phase
  while phase_b > 360:
    phase_b -= 360

  while phase_b < 0:
    phase_b += 360
  
  ### Amplitude ###
  
  if i+1 <= iedge:
    amp = 100*math.sin((i+1)*pi/2/iedge)
  
  elif i+1 > idelta:
    amp = 100*math.sin(pi/2+pi/2*(i-idelta)/iedge) 
    
  else:
    amp = 100
  
  f.write('{:6e}'.format(amp) +", " +'{:6e}'.format(phase_b)+"\n")

f.write("##END\n")
f.close()
  