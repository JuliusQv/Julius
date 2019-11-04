import math
import numpy as np
# benchmark函数


def Brown(x):
    # 32维输入变量
    # 变量范围为 [-1,4]
    y = 0
    for n in range(0, 31):
        y = y + (x[n]**2)**(x[n+1]**2+1) + (x[n+1]**2)**(x[n]**2+1)
    return y


def Rosenbrock(x):
    #  8 维输入变量 [-5,10]   min = 0
    f = 0
    for k in range(1, 8):
        f = f + 100*(x[k-1]**2-x[k])**2+((x[k-1]-1)**2)
    return f


def Chichinadze(x):
    # 2 维输入变量 [-30 30]
    x1 = x[0]
    x2 = x[1]
    return x1**2-12*x1**2+11+10*math.cos(math.pi*x1/2)+8*math.sin(5*math.pi*x1/2)-(0.2)**0.5*math.exp(-0.5*(x2-0.5)**2)


def DropWave(x):
    x1 = x[0]
    x2 = x[1]
    #  2 维输入变量 [-5.12,5.12]   min = -1
    return -(1+math.cos(12*math.sqrt(x1**2+x2**2)))/(0.5*(x1**2+x2**2)+2)


def EggHolder(x):
    #  2 维输入变量 [-512,512]   min = -959.641
    x1 = x[0]
    x2 = x[1]
    return -(x2+47)*math.sin(math.sqrt(np.abs(x2+0.5*x1+47)))-x1*math.sin(math.sqrt(np.abs(x1-x2-47)))


def Schubert(x):
    #  2 维输入变量 [-10 10]   min = -186.730
    x1 = x[0]
    x2 = x[1]
    f = 0
    for j in range(1, 6):
        f = f + j*math.cos((j+1)*x1)+j + j*math.cos((j+1)*x2)+j
    return f

#def HolderTable(x1,x2):
 #   return -np.abs(math.sin(x1)*math.cos(x2)*math.exp(np.abs(1-math.sqrt(x1**2+x2**2)/math.pi)))
    
#def Schafferl(x):
 #   x1 = x[0]
#    x2 = x[1]
 ##   return x1**2+x2**2-10*math.cos(2*math.pi*x1)-10*math.cos(2*math.pi*x2)+20

#def Schaffern2(x1,x2):
 #   return 0.5+((math.sin(x1**2-x2**2))**2-0.5)/(1+0.001*(x1**2+x2**2))**2

# def Bukin(x1, x2):
    # 2 维输入变量
   # return 100*math.sqrt(np.abs(x2-0.01*x1**2))+0.01*np.abs(x1+10)

#def SphereModel(x1,x2):
    #  |x|<=100   min = 0
#    return 100*(x2-x1**2)**2+(x1-1)**2

#def StyblinskyTang(x):
#    f=0
#    for a in x:
#        f = f + a**4-16*a**2+5*a
#    return 0.5*f

#def Schwefel1(x1,x2,x3):
#    return np.abs(x1)+np.abs(x2)+np.abs(x3)+np.abs(x1)*np.abs(x2)*np.abs(x3)

#def GRF(x):
#    x1=x[0]
#   x2=x[1]
#    return x1**2-10*math.cos(2*math.pi*x1)+10+x2**2-10*math.cos(2*math.pi*x2)+10
