#%% Importing Modules
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.optimize import curve_fit
from scipy.stats import chisquare

# %% Computing Random Walk: x(t) & x_RMS(t)
emptyList = [0] #position list
emptyRmsList = [0] #xrms list

for majorIteration in range(10000): #computing 10000 steps

    for iteration in range(7): #computing 7 steps, with size +/- 1

        # Loop for x(t)
        possibleStepValues = [-1,1]
        stepChosen = np.random.choice(possibleStepValues)
        updatedDistance = emptyList[-1] + stepChosen
        emptyList = np.append(emptyList,updatedDistance)

        # Loop for x_rms(t)
        xrmsValue = 1
        updatedXrms = emptyRmsList[-1] + xrmsValue
        emptyRmsList = np.append(emptyRmsList,updatedXrms)


    for newIteration in range(1): #computing 1 step, with size +/- 3

        # Loop for x(t)
        possibleStepValues = [-3,3]
        stepChosen = np.random.choice(possibleStepValues)
        updatedDistance = emptyList[-1] + stepChosen
        emptyList = np.append(emptyList,updatedDistance)

        # Loop for x_rms(t)
        xrmsValue = 9
        updatedXrms = emptyRmsList[-1] + xrmsValue
        emptyRmsList = np.append(emptyRmsList,updatedXrms)

sizeOfTotalWalks = len(emptyList) #total amount of steps
horizontalValues = np.arange(sizeOfTotalWalks) 

# %% Computing Predictions
D = 1/2

#Defining fitting functions
def angela(t, epsilon):
    return np.sqrt(2*(D+epsilon)*t)

def mike(t, delta): 
    return np.sqrt(2*D) * t**(1/2+delta)

#Best fit parameters
poptAngela, pcovAngela = curve_fit(angela, horizontalValues, np.sqrt(emptyRmsList))
poptMike, pcovMike = curve_fit(mike, horizontalValues, np.sqrt(emptyRmsList))

# %% Testing Goodness of Fits: ChiSquare Test
RMS = np.delete(emptyRmsList,0) #removing the (0,0) value which blows up the test
HOR = np.delete(horizontalValues,0)
chisqAngela = chisquare(f_obs=np.sqrt(RMS), f_exp=angela(HOR, *poptAngela))
chisqMike = chisquare(f_obs=np.sqrt(RMS), f_exp=mike(HOR, *poptMike))

# %% Plotting Reults
plt.grid()
plt.plot(horizontalValues, emptyList, color='blue', label='x(t)')
plt.plot(horizontalValues, np.sqrt(emptyRmsList), color='springgreen', label= r'$x_{RMS}(t)$',linewidth=3)
plt.plot(horizontalValues, angela(horizontalValues, *poptAngela), color='fuchsia', label='Angela')
plt.plot(horizontalValues, mike(horizontalValues, *poptMike), color='black', label='Donald')
plt.title("Random Walk  in 1D For 80000 Steps")
plt.xlabel("Steps")
plt.ylabel("Position")
plt.legend()
plt.savefig('RandomWalk.jpg', bbox_inches='tight', dpi=400)
plt.show()

# %%
