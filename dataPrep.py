import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import math

class DataPreparation:
   m = 0
   lf = 0
   lr = 0
   Cf = 0
   Cr = 0
   Iz = 0
   steering_ratio = 0
   Ts = 0
   K = 0
   L = 0
   P = 0

   def __init__(self, drivingProfile) -> None:
    self.m = drivingProfile['m'][0]
    self.lf = drivingProfile['lf'][0]
    self.lr = drivingProfile['lr'][0]
    self.Cf = drivingProfile['Cf'][0]
    self.Cr = drivingProfile['Cr'][0]
    self.Iz = drivingProfile['Iz'][0]
    self.steering_ratio = drivingProfile['steering_ratio'][0]
    self.Ts = drivingProfile['Sampling Period'][0]
    self.K = np.array([[1.8639, -0.0279],[-542794.8896, 32604.6225]])
    self.L = np.array([[-0.000000000009229, -0.000000000131049],[0.000000000135782, 0.000000001407986]])
    self.P = np.array([[2, 1],[1,1]])
  
   def calculateParameters(self, v_x):
    A = np.array([
      np.array([(-2*self.Cf-2*self.Cr)/(self.m*v_x), -1-((2*self.Cf*self.lf - 2*self.Cr*self.lr)/(self.m*v_x*v_x))]), 
      np.array([((2*self.Cr*self.lr)-(2*self.Cf*self.lf))/self.Iz, (-2*self.Cf*self.lf*self.lf-2*self.Cr*self.lr*self.lr)/(self.Iz*v_x)])
      ])
    B = np.array([[(2*self.Cf)/(self.m*v_x), 0], [(2*self.Cf*self.lf)/self.Iz, 1/self.Iz]])
    C = np.array([[0, 1], [(-2*self.Cf-2*self.Cr)/self.m, -(2*self.Cf*self.lf-2*self.Cr*self.lr)/(self.m*v_x)]])
    D = np.array([[0, 0], [(2*self.Cf)/self.m, 0]])

    return [A, B, C, D]
   
   def computeData(self, Time, deltaSteer, Vx):
    n = math.floor(self.Ts/(Time[1]-Time[0]))

    time = [Time[i] for i in range(len(Time)) if i%n==0]
    steeringAngle = [deltaSteer[i] for i in range(len(deltaSteer)) if i%n==0]
    velocity = [Vx[i] for i in range(len(Vx)) if i%n==0]

    u = [np.transpose(np.array([np.array([deltaSteer[0]*math.pi/180, 0])]))]
    x = [np.dot(np.linalg.inv(-self.K), u[0])]
    x_cap = [np.dot(np.linalg.inv(-self.K), u[0])]
    A, B, C, D = self.calculateParameters(max(Vx))
    y = [np.dot(C,x[0])]
    y_cap = [np.dot(C,x_cap[0])]

    for i in range(1, len(velocity)):
        A, B, C, D = self.calculateParameters(velocity[i-1])

        out = signal.StateSpace(A, B, C, D)
        out = out.to_discrete(self.Ts)

        r = y[i-1] - y_cap[i-1]

        x.append(np.array(np.dot(out.A, x[i-1])+np.dot(out.B, u[i-1])))
        x_cap.append(np.array(np.dot(out.A, x[i-1])+np.dot(out.B, u[i-1]))+np.dot(self.L,r))

        y.append(np.dot(out.C, x[i-1])+np.dot(out.D, u[i-1]))
        y_cap.append(np.dot(out.C, x_cap[i-1])+np.dot(out.D, u[i-1]))

        utemp = np.dot(-self.K, x[i])
        utemp[0][0] += steeringAngle[i]*(math.pi/180)*self.steering_ratio
        u.append(utemp)

    Mz = []
    delta = []
    for i in range(len(u)):
        Mz.append(u[i][1][0])
        delta.append(u[i][0][0])
    
    r = []
    beta = []
    for i in range(len(x)):
        r.append(x[i][1][0])
        beta.append(x[i][0][0])
    
    r_cap = []
    beta_cap = []
    for i in range(len(x)):
        r_cap.append(x_cap[i][1][0])
        beta_cap.append(x_cap[i][0][0])
    
    r1 = []
    ay = []
    for i in range(len(y)):
        r1.append(y[i][0][0])
        ay.append(y[i][1][0])
    
    r1_cap = []
    ay_cap = []
    for i in range(len(y_cap)):
        r1_cap.append(y_cap[i][1][0])
        ay_cap.append(y_cap[i][0][0])
    
    return [time, Mz, delta, r, r_cap, beta, beta_cap, ay, ay_cap]
   
   def getOutput(self, Time, deltaSteer, Vx):
    [time, Mz, delta, r, r_cap, beta, beta_cap, ay, ay_cap] = self.computeData(Time, deltaSteer, Vx)
    return [time, [r, ay]]
   
   def getEstimatedOutput(self, Time, deltaSteer, Vx):
    [time, Mz, delta, r, r_cap, beta, beta_cap, ay, ay_cap] = self.computeData(Time, deltaSteer, Vx)
    return [time, [r_cap, ay_cap]]
   
   def getInput(self, Time, deltaSteer, Vx):
    [time, Mz, delta, r, r_cap, beta, beta_cap, ay, ay_cap] = self.computeData(Time, deltaSteer, Vx)
    return [time, [Mz, delta]]
   
   def getStates(self, Time, deltaSteer, Vx):
    [time, Mz, delta, r, r_cap, beta, beta_cap, ay, ay_cap] = self.computeData(Time, deltaSteer, Vx)
    return [time, [r, beta]]
   
   def getEstimatedStates(self, Time, deltaSteer, Vx):
    [time, Mz, delta, r, r_cap, beta, beta_cap, ay, ay_cap] = self.computeData(Time, deltaSteer, Vx)
    return [time, [r_cap, beta_cap]]

##################################################################################

'''
drivingProfile = pd.read_csv('./parameter.csv')
dp = DataPreparation(drivingProfile)

data = pd.read_csv('./SLC/SLC_input.csv')

Time = data['time'].to_numpy()
deltaSteer = data['Drv_DeltaSteer'].to_numpy()
Vx = data['Veh_Vx'].to_numpy()

output = dp.getOutput(Time, deltaSteer, Vx)
input = dp.getInput(Time, deltaSteer, Vx)
'''
