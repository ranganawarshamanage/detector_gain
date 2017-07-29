
from __future__ import division
from scitbx.array_family import flex
from scitbx.random import variate, normal_distribution, poisson_distribution, set_random_seed

#from IPython import embed;embed()
#flex.set_random_seed=100968 # same values each time
from math import atan,sin,pow,log,exp

import random

def Io_values():
  random.seed(574957)
  array = [random.uniform(1,25) for i in xrange(1000)]
  return flex.double(array)

def gain_factor_k():
  distribution = normal_distribution(mean=10.,sigma=2.)
  pv = variate(distribution)
  set_random_seed(525700)
  k_array = pv(10000)
  return k_array

def common_mode_t():
  distribution = normal_distribution(mean=0.,sigma=50.)
  pv = variate(distribution)
  set_random_seed(525700)
  cm = pv(1000)
  return cm

class simulated_image_manager(object):
 def __init__(self):
  """
  def getEndian(): return 0
  from iotbx.detectors.detectorbase import DetectorImageBase
  self.base = DetectorImageBase(filename = "dummy")
  self.base.getEndian = getEndian
  self.base.parameters = dict(BYTE_ORDER="little_endian",
    SIZE1=100,
    SIZE2=100,
    PIXEL_SIZE=0.11, #mm
    DISTANCE=100, #mm
    TWOTHETA=0.0,
    PHI=0.0,
    OSC_START=0.0,
    OSC_RANGE=0.1, #deg
    WAVELENGTH=1.2, # angstroms
    BEAM_CENTER_X=-32.59,
    BEAM_CENTER_Y=-32.59,
    DETECTOR_SN=17)
  self.base.linearintdata = flex.int([10]*10000)
  self.base.linearintdata.reshape(flex.grid(100,100))
  """
  from dxtbx.model.detector import DetectorFactory
  factory = DetectorFactory()
  sensor = factory.sensor("PAD")
  self.detector = factory.simple(sensor=sensor, distance=100, beam_centre=(-32.59,-32.59),
                            fast_direction="+x", slow_direction="-y",
                            pixel_size=(0.11,0.11), image_size=(100,100),
                            trusted_range=(-65535,65535))
  from dxtbx.model.beam import BeamFactory
  self.beam = BeamFactory().simple(wavelength=1.2)
  from dxtbx.model.goniometer import GoniometerFactory
  self.gonio = GoniometerFactory().single_axis()

 def write_image(self,integer_data_field,image_number,image_name="data_series"):
  #self.base.debug_write(fileout="%s_%04d.smv"%(image_name,image_number),
  #                 mod_data = integer_data_field)
  from dxtbx.model.scan import ScanFactory
  self.scan = ScanFactory.make_scan(image_range=(0,0), exposure_times=[1.0,],
                              oscillation = (0.0,0.1), epochs=(0,))
  from dxtbx.format.FormatCBFMini import FormatCBFMini as Mini
  integer_data_field.reshape(flex.grid(100,100))
  Mini.as_file(detector=self.detector,beam=self.beam,
               gonio=self.gonio,scan=self.scan,
               data=integer_data_field,path="%s_%04d.smv"%(image_name,image_number)
              )

def rupp(kscale,detector,beam):
  """From Rupp textbook:
   < Iabs > = Sum(atoms) (f0j)**2
   ln <iobs>/ <abs> = ln (k) - 2 Biso (sintheta/lambda)**2

  """
  Iabs = 64.0 # oxygen
  Biso = 20. # B-factor
  from scitbx.matrix import col
  beam_cen = detector[0].get_beam_centre(beam.get_s0())
  detector_corner = col((-beam_cen[0],-beam_cen[1]))
  photon_field = flex.double()
  sampled_variate = flex.int()
  for address in xrange(100*100): # hardcode dimensions
    x, y = address%100, address//100
    pixel_vector = col((detector[0].get_pixel_size()[0] * x,
                        detector[0].get_pixel_size()[1] *y))
    lab_vector = detector_corner + pixel_vector
    lab_vector_length = lab_vector.length()
    two_theta = atan(lab_vector_length/ detector[0].get_distance())
    wilson_x = pow(sin(two_theta/2.)/beam.get_wavelength(),2.)
    r_h_s  = log(kscale) - 2. * Biso * wilson_x
    intensity = Iabs * exp (r_h_s)
    photon_field.append( intensity )

    distribution = poisson_distribution(intensity)
    pv = variate(distribution)
    sampled_variate.append( pv.next() )

  return photon_field.iround(),sampled_variate

if __name__ == "__main__":
  Io_all = Io_values()
  gain = gain_factor_k()
  cm = common_mode_t()
  for t in xrange(1000):
      Io = Io_all[t]
      print t, Io
      manager = simulated_image_manager()
      sampled_photon_count = rupp(kscale = Io,
        detector = manager.detector, beam=manager.beam)[1]
      assert len(sampled_photon_count) == len(gain)
      cm_t = cm[t]
      observation_field = sampled_photon_count.as_double() * gain + cm_t
      integer_data_field = observation_field.iround()
      manager.write_image(integer_data_field,image_number = t)
