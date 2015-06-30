# This file is part of the exercise code repository accompanying 
# the book: Neuronal Dynamics (see <http://neuronaldynamics.epfl.ch>)
# located at <http://github.com/EPFL-LCN/neuronaldynamics-exercises>.
# 
# This free software: you can redistribute it and/or modify it under 
# the terms of the GNU General Public License 2.0 as published by the 
# Free Software Foundation. You should have received a copy of the 
# GNU General Public License along with the repository. If not, 
# see <http://www.gnu.org/licenses/>.
#
# Should you reuse and publish the code for your own purposes, 
# please cite the book or point to the webpage <http://neuronaldynamics.epfl.ch>.
# 
# Wulfram Gerstner, Werner M. Kistler, Richard Naud, and Liam Paninski. 
# Neuronal Dynamics: From Single Neurons to Networks and Models of Cognition. 
# Cambridge University Press, 2014.

from pylab import *

# ------------ OJA's LEARNING RULE -------------------


def make_cloud(size=10000,ratio=1,angle=0):
    """
    DEFINITION
    builds an oriented elliptic (or circular) gaussian cloud of 2D points
    
    INPUT
    size: the number of points
    ratio: (std along the short axis) / (std along the long axis)
    angle: the rotation angle in degrees
    """
    
    if ratio>1: ratio=1/ratio
    
    x = randn(size,1)
    y = ratio * randn(size,1)
    z = concatenate((x,y),1)
    radangle = (180 - angle) * pi / 180
    transfo = [[cos(radangle),sin(radangle)],[-sin(radangle),cos(radangle)]]
    data = dot(transfo,z.T).T
    return data


def learn(cloud,eta=0.001):
    """
    DEFINITION
    run Oja's learning rule on a cloud of datapoints
    
    INPUT
    eta: learning rate
    cloud: the cloud of datapoints
    
    OUTPUT
    the time course of the weight vector 
    """
    
    w = array([1/sqrt(2),1/sqrt(2)])
    wcourse = zeros((len(cloud),2),float)
    for i in range(0,len(cloud)):
        wcourse[i] = w
        y = dot(w,cloud[i]) # output
        w = w + eta*y*(cloud[i]-y*w) # learning rule        
    return wcourse




# ------------ BCM LEARNING RULE -------------------

# for internal use
def circ_dist(n,i,j):
  if i == j: 
      return 0.
  else:
      if j < i:
          if (i - j) > (n/2): return (i-n-j)
          else: return (i-j)
      else: return (-circ_dist(n,j,i))



def make_image(m,sigma):
    """
    DEFINITION
    builds an m-by-m matrix featuring a gaussian bump
    centered randomly
    NOTA: 
    - to make this matrix a vector (concatenating all rows),
    you can use m.flatten()
    - conversely, if you need to reshape a vector into an 
    m-by-m matrix, use v.reshape((m,m))
    
    INPUT
    m: the size of the image (m-by-m)
    sigma: the std of the gaussian bump
    """
    
    img = zeros((m,m),float)
    ci = int(m*rand())
    cj = int(m*rand())
    for i in range(0,m):
        di = circ_dist(m,ci,i)
        for j in range(0,m):
            dj = circ_dist(m,cj,j)
            img[i,j] = exp (-(di*di + dj*dj)/(2.0*sigma*sigma))
    return (1./norm(img))*img


string = "***********************************************************"
string += "\r\n"
string += "***** Ecole polytechnique federale de Lausanne (EPFL) *****"
string += "\r\n"
string += "***** Laboratory of computational neuroscience (LCN)  *****"
string += "\r\n"
string += "*****               Neuronal Dynamics                 *****"
string += "\r\n"
string += "***********************************************************"
string += "\r\n"
string += "This file implements Oja's hebbian learning rule."
string += "\r\n"
string += "In order to know parameters and default values for each method"
string += "\r\n"
string += "use symbol ? after the name of method. For example: learn?"
string += "\r\n\r\n"
string += "-------------2014 EPFL-LCN----------------"
string += "\r\n"

print string







