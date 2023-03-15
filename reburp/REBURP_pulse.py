# -*- coding: utf-8 -*-
"""
This script allows you to create a phase-shifted REBURP pulse.
Based on Lewis's ccode "BRF_reburp.c" to generate a reburp pulse. 

In the Bruker format, the pulse phase is defined in the unit of degree.
The amplitude is 

w1(t) = A0 + sigma[An*cos(nwt)]

where w = 2*pi/tau and tau is the pulse duration.
(see Eq 45 in the reference below)

A0/An is the Fourier coefficient. Since REBURP is symmetric, we do not need Bn terms.
(Geen and Freeman, J. Mag. Res., Vol. 93, 93-141, 1991)

The pulse phase is lineary incremented by offset*timestep*360 (deg),
where the offset defines the center of the pulse irradiation from the carrier.

written by Y. Toyama

numpy : 1.21.5

"""

import numpy as np

### Parameters ####

outname = "reb_7p95.BRF" # Output file name
Pulse = 1360E-6 # total pulse duration in sec
Stepsize = 1E-6 # time step in sec
Offset = 3254 # in Hz

#####################

Offset = -1*Offset # for Bruker

Number_Steps = Pulse/Stepsize # the number of steps

# Fourier coefficients taken from Table 8 in Geen and Freeman's paper.
An = [0.49,-1.02,1.11,-1.57,0.83,-0.42, 0.26,-0.16,0.10,-0.07,0.04,-0.03,0.01,-0.02,0.00,-0.01]


#######################
# Calculate and add the amplitude and phase
#######################

amplist = np.zeros(int(Number_Steps))
phaselist = np.zeros(int(Number_Steps))

for i in range(int(Number_Steps)):
 
  ### Amplitude ###
  
  omegat = 2*np.pi*(i+1)/(Number_Steps+1)
  amp_t = 0
  
  for k in range(16):
    amp_t += An[k]*np.cos(k*omegat)
  
  
  ### Phase ###
  # The constant phase to shift the frequency.

  phase_b =  Offset*Pulse*360*(i+1-(Number_Steps+1)/2)/(Number_Steps+1)
  
  while phase_b > 360:
    phase_b -= 360

  while phase_b < 0:
    phase_b += 360
  
  # Note when the amplitude is negative, the pulse is applied on -x,
  # so the phase need to be added/subtracted by 180 deg.
  # The amplitude in the Bruker's table needs to be an absolute value.
  
  if amp_t < 0:
      if phase_b < 180:
          phase_b += 180
      else:
          phase_b -= 180
      amp_t = -1*amp_t
  
    
  amplist[i] = amp_t
  phaselist[i] = phase_b

# Normalize amplist
# The values in the amplitude is normalized so that the maximum is 100 (otherwise ~ 6.14).
# The pulse power is defined in the experimental parameter such as spdb in Topspin.
 
amplist = 100*amplist/np.max(amplist)


#######################
# Write headers for Bruker shape file
#######################

f = open(outname,'w')
f.write("##TITLE= reburp\n")
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
f.write("##$SHAPE_TYPE= Refocusing\n")
f.write("##$SHAPE_USER_DEF=\n")
f.write("##$SHAPE_REPHFAC= \n")
f.write("##$SHAPE_BWFAC= 0.000000E00\n")
f.write("##$SHAPE_BWFAC50= \n")
f.write("##$SHAPE_INTEGFAC= 2.097446E-04\n")
f.write("##$SHAPE_MODE= 0\n")
f.write("##NPOINTS= "+str(int(Number_Steps))+"\n")
f.write("##XYPOINTS= (XY..XY)\n")

#######################
# Add amplitude and phase lines
#######################

for i in range(int(Number_Steps)):
    f.write('{:6e}'.format(amplist[i]) +", " +'{:6e}'.format(phaselist[i])+"\n")

f.write("##END\n")
f.close()
  