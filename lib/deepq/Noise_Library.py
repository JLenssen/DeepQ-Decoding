import numpy as np

class NoiseFactory():

  def __init__(self, error_model, d, p_phys, **kwargs):
    self.error_model = error_model
    self.d = d
    self.p_phys = p_phys
    self.kwargs = kwargs

  def generate(self):
    if self.error_model == "X":
      return XNoise(self.d, self.p_phys)
    elif self.error_model == "DP":
      return DPNoise(self.d, self.p_phys)
    elif self.error_model == "IIDXZ":
      return IIDXZNoise(self.d, self.p_phys)
    elif self.error_model == "HEAT":
      if "heat" in self.kwargs:
        return HeatmapNoise(self.d, self.p_phys, self.kwargs["heat"])
    else:
      raise ValueError(f"Error model: {self.error_model} doesn't exist!")


class DPNoise():
  def __init__(self, d, p_phys, stddev=0.0):
    self.d = d
    self.p_phys = p_phys
    self.stddev = stddev

    if not np.isclose(self.stddev, 0.0):
      self.p_phys = np.random.normal(self.p_phys, self.stddev, (self.d, self.d))
      self.p_phys = self.p_phys.clip(min=0)

    print(f"Using DP Noise (stddev {self.stddev}), with physical qubit error distribution: ")
    print(self.p_phys)

  def get_error_model(self):
    return "DP"

  def generate_error(self):
      """"
      This function generates an error configuration, via a single application of the depolarizing noise channel, on a square dxd lattice.
      
      :param: d: The code distance/lattice width and height (for surface/toric codes)
      :param: p_phys: The physical error rate.
      :return: error: The error configuration
      """

      error = np.random.binomial(1, self.p_phys, (self.d, self.d))
      error = error.astype(np.uint8)
      error *= np.random.randint(low=1, high=4, size=(self.d,self.d), dtype=np.uint8)

      return error


class XNoise():
  def __init__(self, d, p_phys, stddev=0.0, context="X"):
    self.d = d
    self.p_phys = p_phys
    self.stddev = stddev

    if not np.isclose(self.stddev, 0.0):
      self.p_phys = np.random.normal(self.p_phys, self.stddev, (self.d, self.d))
      self.p_phys = self.p_phys.clip(min=0)

    print(f"Using X Noise (in context {context}, stddev {self.stddev}), with physical qubit error distribution: ")
    print(self.p_phys)

    # agent selects number of actions based on noise model
    # if we want to test X noise for DP trained agent, the number of
    # actions won't match. Therefore we set a different context.
    self.context = context

  def get_error_model(self):
    return self.context

  def generate_error(self):
      """"
      This function generates an error configuration, via a single application of the bit-flip noise channel, on a square dxd lattice.
      
      :param: d: The code distance/lattice width and height (for surface/toric codes)
      :param: p_phys: The physical error rate.
      :return: error: The error configuration
      """

      error = np.random.binomial(1, self.p_phys, (self.d, self.d))
      return error.astype(np.uint8)

class ZNoise():
  def __init__(self, d, p_phys, stddev=0.0, context="Z"):
    self.d = d
    self.p_phys = p_phys
    self.stddev = stddev

    if not np.isclose(self.stddev, 0.0):
      self.p_phys = np.random.normal(self.p_phys, self.stddev, (self.d, self.d))
      self.p_phys = self.p_phys.clip(min=0)

    print(f"Using Z Noise (in context {context}, stddev {self.stddev}), with physical qubit error distribution: ")
    print(self.p_phys)

    # agent selects number of actions based on noise model
    # if we want to test Z noise for DP trained agent, the number of
    # actions won't match. Therefore we set a different context.
    self.context = context

  def get_error_model(self):
    return self.context

  def generate_error(self):
      """"
      This function generates an error configuration, via a single application of the phase-flip noise channel, on a square dxd lattice.
      
      :param: d: The code distance/lattice width and height (for surface/toric codes)
      :param: p_phys: The physical error rate.
      :return: error: The error configuration
      """

      error = np.random.binomial(1, self.p_phys, (self.d, self.d))*3
      return error.astype(np.uint8)


class CrossNoise():
  def __init__(self, d, p_phys, bias=0.0, context="DP"):
    self.d = d
    self.p_phys = p_phys
    self.bias = bias

    if not np.isclose(self.bias, 0.0):
      self.p_phys = np.ones((self.d, self.d)) * p_phys
      p = p_phys + self.bias
      middle = (d-1)//2
      self.p_phys[middle] = p
      self.p_phys[:,middle] = p

    print(f"Using Cross-Noise (in context {context}, bias {self.bias}), with physical qubit error distribution: ")
    print(self.p_phys)

    # agent selects number of actions based on noise model
    # if we want to test Cross Noise noise for DP trained agent, the number of
    # actions won't match. Therefore we set a different context.
    self.context = context

  def get_error_model(self):
    return self.context

  def generate_error(self):

      error = np.random.binomial(1, self.p_phys, (self.d, self.d))
      error = error.astype(np.uint8)
      error *= np.random.randint(low=1, high=4, size=(self.d,self.d), dtype=np.uint8)

      return error


class QuadrantNoise():
  def __init__(self, d, p_phys, quadrant=1,  bias=0.0, context="DP"):
    self.d = d
    self.p_phys = p_phys
    self.bias = bias
    self.quadrant = quadrant

    if not np.isclose(self.bias, 0.0):
      self.p_phys = np.ones((self.d, self.d)) * p_phys
      p = p_phys + self.bias
      m = (d-1)//2 # row-column number at the middle of the SC

      if self.quadrant == 1:
        self.p_phys[0:m+1,0:m+1] = p
      elif self.quadrant == 2:
        self.p_phys[0:m+1,m:d] = p 
      elif self.quadrant == 3:
        self.p_phys[m:d,0:m+1] = p
      elif self.quadrant == 4:
        self.p_phys[m:d,m:d] = p
      else:
        raise ValueError("Quadrant not existant!")

    print(f"Using Quadrant-Noise (in context {context}, bias {self.bias}), with physical qubit error distribution: ")
    print(self.p_phys)

    # agent selects number of actions based on noise model
    # if we want to test Quadrant Noise noise for DP trained agent, the number of
    # actions won't match. Therefore we set a different context.
    self.context = context

  def get_error_model(self):
    return self.context

  def generate_error(self):

      error = np.random.binomial(1, self.p_phys, (self.d, self.d))
      error = error.astype(np.uint8)
      error *= np.random.randint(low=1, high=4, size=(self.d,self.d), dtype=np.uint8)

      return error


class ConcentricNoise():
  def __init__(self, d, p_phys, bias=0.0, context="DP"):
    self.d = d
    self.p_phys = p_phys
    self.bias = bias

    self.p_phys = np.zeros((self.d, self.d)) * p_phys
    m = (d-1)//2 # row-column number at the middle of the SC
    
    for i in range(d):
      self.p_phys[0,i] = i if i < m+1 else 2*m-i 
    for j in range(1,d):
      self.p_phys[j] = self.p_phys[j-1] + 1 if j < m+1 else self.p_phys[j-1] - 1
    
    err_range = np.linspace(p_phys/2, p_phys+bias, num=d)
    for i in range(d):
      for j in range(d):
        self.p_phys[i,j] = err_range[int(self.p_phys[i,j])]

    print(f"Using Concentric-Noise (in context {context}), with physical qubit error distribution: ")
    print(self.p_phys)

    # agent selects number of actions based on noise model
    # if we want to test Concentric Noise noise for DP trained agent, the number of
    # actions won't match. Therefore we set a different context.
    self.context = context

  def get_error_model(self):
    return self.context

  def generate_error(self):

      error = np.random.binomial(1, self.p_phys, (self.d, self.d))
      error = error.astype(np.uint8)
      error *= np.random.randint(low=1, high=4, size=(self.d,self.d), dtype=np.uint8)

      return error


class IIDXZNoise():

  def __init__(self, d, p_phys):
    self.d = d
    self.p_phys = p_phys

  def get_error_model(self):
    return "IIDXZ"

  def generate_IIDXZ_error(self):
      """"
      This function generates an error configuration, via a single application of the IIDXZ noise channel, on a square dxd lattice.
      
      :param: d: The code distance/lattice width and height (for surface/toric codes)
      :param: p_phys: The physical error rate.
      :return: error: The error configuration
      """

      error = np.zeros((self.d, self.d), dtype=np.uint8)
      for i in range(self.d):
          for j in range(self.d):
              X_err = False
              Z_err = False
              p = 0
              if np.random.rand() < self.p_phys:
                  X_err = True
                  p = 1
              if np.random.rand() < self.p_phys:
                  Z_err = True
                  p = 3
              if X_err and Z_err:
                  p = 2

              error[i, j] = p

      return error


class HeatmapNoise():

  def __init__(self, d, p_phys, heat):
    self.d = d
    self.p_phys = p_phys
    self.heat = heat

  def get_error_model(self):
    # TODO environment actions require DP and don't recognize HEAT
    return "DP"

  def generate_error(self):
      """"
      This function generates an error configuration, using a dxd heatmap as input that servers as error probability distribution
      for every qubit.

      :param: d: The code distance/lattice width and height (for surface/toric codes)
      :param: heatmaps: List of Numpy arrays, shape dxd. [heatmap_X, heatmap_Z] is the default
      """

      error = np.zeros((self.d, self.d), dtype=np.uint8)
      for i in range(self.d):
          for j in range(self.d):
              p = 0
              if np.random.rand() < self.heat[i,j]:
                  p = np.random.randint(1, 4)
                  error[i, j] = p

      return error

  def set_physical_error_rate(self, p_phys):
    # find factor so that max value in gaussian corresponds to depolarizing p_phys
    self.heat = p_phys/np.max(self.heat)*self.heat

    assert np.isclose(np.max(self.heat), p_phys)
