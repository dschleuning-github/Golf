import math
import matplotlib.pyplot as plt
import numpy as np


print('hello world')


DIAMETER_BALL =	0.04267 #m
AREA_BALL =	0.00143 #m^2
GRAVITY = 9.81 #m/s^2
MASS_BALL = 0.04568 #kg
DELTA_T = 0.03 #s   <- change in time
RHO = 1.2041 #kg/m^3
NU = 0.0000146  #kinematic viscosity of air -> m^2/s
SPIN_DECAY = 0.04  #spin decay per second
pi = 3.14159265359


V_BALL = input("What is the ball speed?(mph) ") #mph
BALL_VELOCITY = float(V_BALL) * 1609.34 / 60.0 / 60.0  #m/s
launchAngle = float(input("What is the launch angle?(deg) ")) # deg
LAUNCH_ANGLE = float(launchAngle) *pi/180 #radians
RPM = float(input("What is the spin rate?(rpm) ")) #rpm
OMEGA = float(RPM)*2*pi/60  # rad/sec



"""
this function calculates the Reynold's Number when
given the velocity of the ball
the equation also uses the diameter of the ball and the kinematic
viscosity of air
"""
def findRE(Velocity):
    RE = float(Velocity)*DIAMETER_BALL/NU
    return RE

"""
This function is used to calibrate later equations (that use the drag coeficient)
to empirical data to provide accurate results.
refrence: https://blog.trackmangolf.com/trackman-average-tour-stats/
"""
def dragMultiplier(RPM):
    result = -0.000000019*(float(RPM)-6750)*(float(RPM)-6750)+1.52
    return result

"""
Because balls traveling at different speeds have different drag coeficients,
this function uses equations based off of empirical data to return a drag
coeficient for a given Reynold's Number
refrence for empirical data: https://www.semanticscholar.org/paper/Aerodynamics-of-Golf-Balls-in-Still-Air-Lyu-Kensrud/595d3d1f91e240cb47e153b8d0ef586b8caa67c4/figure/3
"""
def dragCoeficient(RE, DRAG_MULTIPLIER):
  result = 0;
  if(float(RE)<81207.187):
      #low speed
      result = 0.000000000129*float(RE)*float(RE)-0.0000259*float(RE)+1.5
  elif(float(RE)<141361.257):
      #high speed
      result = (1.91)*(0.00000000001)*float(RE)*float(RE)-0.0000054*float(RE)+0.56
  else:
      #"higher" speed
      result = 0.178
  
  finalResult = result*DRAG_MULTIPLIER
  return finalResult

"""
This function finds the force of drag on the ball.
R=1/2 *Cd*rho*A*V^2
The unit of the result is Newtons.
For the parameters, v represents ball velocity and Cd represents the coeficient
of drag.
"""
def airResistance(V, Cd):
    result = 0.5*float(Cd)*RHO*AREA_BALL*float(V)*float(V)
    return result


"""
As a ball moves through the air, its spin rate changes over time
For the purposes of this project, the rate of decay is assumed to be 4%

reference: https://www.semanticscholar.org/paper/Aerodynamics-of-Golf-Balls-in-Still-Air-Lyu-Kensrud/595d3d1f91e240cb47e153b8d0ef586b8caa67c4
go to section "3.3. Golf Ball Trajectory Simulation" (of reference article)

This function returns the angular velocity of the ball for a given amount of time
the parameter t represents time
"""

def angularVelocity(t, OMEGA):
    result = float(OMEGA) *(1- SPIN_DECAY*t)
    return result


"""
To find the force of lift exerted on the ball, we first need to find the 
coeficient of lift. The coeficeint of lift can be determined based off of a 
spin factor (S) where S = (angular velocity)*(radius of ball)/(velocity of ball)

reference:https://www.semanticscholar.org/paper/Aerodynamics-of-Golf-Balls-in-Still-Air-Lyu-Kensrud/595d3d1f91e240cb47e153b8d0ef586b8caa67c4/figure/4
"""
def spinFactor(omega, velocity):
    S = float(omega) * DIAMETER_BALL /2 /float(velocity)
    return S

def liftMultiplier(rpm):
    result = -0.0000000047*float(rpm)*float(rpm)+1.05
    return result


# S represents the spin factor and liftMult represents the lift multiplier
def coeficientLift(S, liftMult):
    if(float(S)<0.2145):
        result = -2.4*float(S)*float(S)+1.8*float(S)
    else:
        result = 1/1.68 * math.sqrt(float(S))

    result = result * float(liftMult)
    return result


def findRPM(omega):
    result = float(omega)/2/pi * 60
    return result

"""
This function is used to calculate the force of lift exerted on the ball
Cl represents the coeficient of lift and V represents the velocity of the ball.
unit for result in Newtons
"""
def forceLift(Cl, V):
    result = 0.5*float(Cl)*RHO*AREA_BALL*float(V)*float(V)
    return result

#this function calculates the force of gravity exerted on the ball
#unit for result in Newtons
def forceGravity():
    result = MASS_BALL*GRAVITY
    return result
#this function returns the angle of the direction of the velocity for an inputed x velocity and y velocity
def findTheta(Vx, Vy):
    return (math.atan(Vy/Vx))


#This function calculates the acceleration of the ball in the y direction by suming the forces acting on the ball and dividing by the mass
def accelerationY(theta, lift, drag, gravity):
    a = (lift*math.cos(theta) - drag*math.sin(theta) - gravity)/MASS_BALL
    return a

#acceleration in the x direction
def accelerationX(theta, lift, drag):
    a = (-lift*math.sin(theta) - drag*math.cos(theta)) /MASS_BALL
    return a

"""
functions to find x velocity and y velocity at a given time
Vf=Vi +at
where Vi is initial velocity and a is acceleration
"""
def findVx(Vix, ax):
    result = Vix + ax*DELTA_T
    return result

def findVy(Viy, ay):
    result = Viy + ay*DELTA_T
    return result

"""
y1+ ay*t+0.5ay t^2
y1 is the previous y value
the first values of x and y are 0 because the ball starts at (0,0)
findY and findX return the ball's position relative to the orgin (0,0)
"""
def findY(y1, Vy, ay):
    result = y1 + Vy*DELTA_T + 0.5*ay*DELTA_T*DELTA_T
    return result

def findX(x1, Vx, ax):
    result = x1 + Vx*DELTA_T + 0.5*ax*DELTA_T*DELTA_T
    return result

#this function finds the total velocity of the ball for a given x velocity and a given y velocity
#the total velocity is needed to calculate the Reynolds number and to calculate forces
def findVelocity(Vx, Vy):
    result = math.sqrt(Vx*Vx + Vy*Vy)
    return result


"""
To calculate the trajectory of the ball, we must use a numerical solution.
To do this, we repeat steps (ex. find magnitude of each force, then combine forces, then find acceleration, then find velocity, then find distance)
This function will combine the previous functions used to carry out the smaller steps.
arrays will contain the x and y position of the ball for each value of time.

The inputs will be the user inputed values for ball speed, launch angle, and spin rate //the inputed values should be in the correct units
"""
def calculateDistance(ballSpeed, LAUNCH_ANGLE, RPM):
    #first, set x and y positions to zero because the ball starts at the position (0,0)
    x = 0
    y = 0
    theta = LAUNCH_ANGLE
    OMEGA = RPM*2*pi/60
    distanceX = [0]
    distanceY = [0]
    COUNT = range(0, 400,1) #count at 400 for high ball speeds and/or to ensure enough data points
    rpm = RPM
    Vball = ballSpeed
    dragMult = dragMultiplier(RPM)
    liftMult = liftMultiplier(RPM)
    time = 0
    RE = findRE(Vball)
    Cd = dragCoeficient(RE, dragMult)
    omega = angularVelocity(time, OMEGA)
    S = spinFactor(omega, Vball)
    Cl = coeficientLift(S, liftMult)
    R = airResistance(Vball, Cd)
    Fl = forceLift(Cl, Vball)
    Fg = forceGravity()
    ax = accelerationX(theta, Fl, R)
    ay = accelerationY(theta, Fl, R, Fg)
    vx = Vball*math.cos(theta)
    vy = Vball*math.sin(theta)
    x = findX(x, vx, ax)
    y = findY(y, vy, ay)

    #for loop to repeat steps(finding forces...) for numerical solution
    #the value of COUNT does not mater but it should be large enough so that the ball crosses the x axis again to calculate the carry distance
    #for(var i = 0; i < COUNT; i++){
    for i in COUNT:
        theta = findTheta(vx, vy)
        RE = findRE(Vball)
        Cd = dragCoeficient(RE, dragMult)
        omega = angularVelocity(time, OMEGA)
        rpm = findRPM(omega)
        S = spinFactor(omega, Vball)
        Cl = coeficientLift(S, liftMult)
        R = airResistance(Vball, Cd)
        Fl = forceLift(Cl, Vball)
        ax = accelerationX(theta, Fl, R)
        ay = accelerationY(theta, Fl, R, Fg)
        vx = findVx(vx, ax)
        vy = findVy(vy, ay)
        x = findX(x, vx, ax)
        y = findY(y, vy, ay)
        Vball = findVelocity(vx, vy)
        
        #push values of x and y into arrays
        distanceX.append(x)
        distanceY.append(y)
        #increase time for next loop
        time += DELTA_T
        
    #carry distance is how far in the x direction the ball traveled in the air (before hiting the ground)
    #use distanceY array to find the maximum height of the ball
    maxY = 0
    for i in range(0, len(distanceY), 1):
        if(distanceY[i] > maxY):
            maxY = distanceY[i]  

    maxX = 0    
    for i in range(0, len(distanceX), 1):
        if(distanceY[i] > 0):
            if(distanceX[i] > maxX):
                maxX = distanceX[i]

    #convert from meters to yards
    maxY = maxY*1.09361
    maxX = maxX*1.09361

    print('Carry Distance: {:.0f} yards'.format(maxX))
    print('Maximum Height: {:.0f} yards'.format(maxY))    

    

    #provide graph of data
    #first conform x values so that the graph only shows positive x and positive y values
#    for i in range(0, len(distanceY)-1):

    yardsX = []
    yardsY = []
    for i in range(0, len(distanceX), 1):
        if(distanceY[i] >= 0):
            x = distanceX[i]*1.09361
            y = distanceY[i]*1.09361
            yardsX.append(x)
            yardsY.append(y)

    #plot
    fig, ax = plt.subplots(figsize=(13,3))
    for i in range(0, len(distanceY), 1):
        ax.plot(yardsX, yardsY)
    ax.set_xlim(0,maxX+15)
    ax.set_ylim(0,maxY+5)
    plt.show()



calculateDistance(BALL_VELOCITY, LAUNCH_ANGLE, RPM)
