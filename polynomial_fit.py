from __future__ import division
from scitbx.array_family import flex
import scitbx.lbfgs

class lbfgs_fit:
  def __init__(self,data,design_matrix):
     self.data = data
     self.design_matrix = design_matrix
     self.nterms = self.design_matrix.shape[1]
     self.x = flex.double(self.nterms,10.)
     self.minimizer = scitbx.lbfgs.run(target_evaluator=self,
                                       termination_params=scitbx.lbfgs.termination_parameters(
                                       traditional_convergence_test=True,
                                         traditional_convergence_test_eps=1.e-8,
                                         max_iterations=100)

     )
     self.a = self.x

  def print_step(pfh,message,target):
    print "%s %10.4f"%(message,target)

  def compute_functional_and_gradients(self):
     self.a = self.x
     f,poly = self.functional(self.a)
     #self.print_step("LBFGS stp",f)
     g = self.gvec_callable(self.x,poly)
     return f, g

  def functional(self,a):
    import numpy as np
    chi2 = 0.0
    #print "Design matrix", self.design_matrix.shape
    a_np = a.as_numpy_array()
    a_t = a_np.reshape((-1,1))
    polynomial = self.design_matrix.dot(a_t)
    polynomial = polynomial.flatten()
    obs = self.data.as_numpy_array()
    obs = obs.flatten()
    #nan_mask = obs > 0
    #polynomial = polynomial * nan_mask
    #obs = obs * nan_mask
    dif = (obs - polynomial)**2
    chi2 = np.nansum(dif)
    return chi2,polynomial

  def gvec_callable(self,values,poly): #this routine is VERY FAST
    result = flex.double(self.nterms)
    obs = self.data.as_numpy_array()
    obs = obs.flatten()
    diff = obs - poly
    for n in xrange(self.nterms):
      result[n] = -2 * diff.dot(self.design_matrix[:,n])
    return result


def lbfgs_example(verbose,data,V,X_flat,Y_flat):
  fit = lbfgs_fit(data,V)
  coef = fit.a
  Coefficients = coef.as_numpy_array()
  #print list(coef)
  Best_fits = V.dot(Coefficients) #Best polynomial fits

  Z = data.as_numpy_array()
  Z = Z.flatten()

  Calc_GF = Z/Best_fits
  plot_2d(Calc_GF)
  plot_3d(X_flat,Y_flat,Z,Best_fits)

def plot_3d(X,Y,Z,M):
  from matplotlib import pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D # implicit import
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  nan_mask = Z > 0
  #Z = Z * nan_mask
  #M = M * nan_mask
  ax.scatter(X, Y, Z, marker='.', color='r', s=3)
  ax.scatter(X, Y, M, marker='.', color='b', s=3)
  plt.xlabel('X')
  plt.ylabel('Y')
  ax.set_zlabel('Z')
  ax.axis('equal')
  ax.axis('tight')
  plt.show()

def plot_2d(CGF):
  from cxi_xdr_xes.per_pixel_gain.photon_field import gain_factor_k
  from matplotlib import pyplot as plt
  import numpy as np
  init_gain = gain_factor_k()
  init_gain = init_gain.as_numpy_array()
  CGF = np.asarray(CGF,dtype=float)
  plt.plot(init_gain,CGF,"r.")
  plt.xlabel('Initial Gain')
  plt.ylabel('Calculated Gain')
  plt.show()


def get_sim_data():
  from dxtbx.format.Registry import Registry
  from scitbx.array_family import flex
  file_name ="avg_summed_1000.smv"
  format_class = Registry.find(file_name)
  i = format_class(file_name)
  raw_data = i.get_raw_data()
  return raw_data

def get_design_matrix(data):
  import numpy as np
  x_dim = data.focus()[0]
  y_dim = data.focus()[1]
  X = np.linspace(1,x_dim,x_dim)
  Y = np.linspace(1,y_dim,y_dim)

  XX,YY = np.meshgrid(X,Y)

  XT = X.reshape((-1, 1)) #get the transpose of X
  YT = Y.reshape((-1, 1))

  lx = len(XT)
  ly = len(YT)

  r1 = np.ones(lx)
  r1T = r1.reshape((-1,1))
  r2 = np.ones(ly)
  r2T = r2.reshape((-1,1))

  Y_mat = YT*r1
  X_mat = r2T*X

  X_flat = X_mat.flatten() #1D array of x_dim*y_dim points
  Y_flat = Y_mat.flatten()

  order = 4
  #Constructing V matrix
  #nterms = (order+1)*(order+2)/2 #constructing half-matrix
  nterms = (order+1)*(order+1) #constructing the full matrix
  s = (len(data),int(nterms))
  V = np.ones(s)
  n = 0
  for r in range(nterms-1):
      n += 1
      if n%(order+1) == 0:
          V[:,n] = X_flat*V[:,n-order-1]
          V[:,n] = V[:,n]*10**-3
      else:
          V[:,n] = Y_flat*V[:,n-1]
          V[:,n] = V[:,n]*10**-3
  return V,X_flat,Y_flat

if (__name__ == "__main__"):
  # start here. Read the average image (average over 1000 simulated images).
  data = get_sim_data()
  print data.focus()
  V,X_flat,Y_flat = get_design_matrix(data) # V - Design_metrix
  print "V rows = ",V.shape[0],"columns = ",V.shape[1]
  verbose=True
  print "Use LBFGS to obtain 4th degree polynomial fit:"
  lbfgs_example(verbose,data,V,X_flat,Y_flat)
