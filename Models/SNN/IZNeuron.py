class IZNeuron(object): # description of the formulars fo IZ given the paper "Simple model of spiking neurons (see zotero)"
  def __init__(self, a=0.02, b=0.2, c=-65, d=2):
    self.v = c
    self.u = b * self.v
    self.a = a
    self.b = b
    self.c = c
    self.d = d

  def _state(self):
    return self.v, self.u

  def integrate(self, I, timesteps = 1): # simulates one ms 
    self.v = self.v + timesteps * (0.04 * (self.v**2) + 5 * self.v + 140 - self.u + I)
    self.u = self.u + timesteps * (self.a * (self.b * self.v - self.u))

    if self.v >= 30:
      self.v = self.c
      self.u = self.u + d
      return True
    return False