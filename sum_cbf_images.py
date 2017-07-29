from __future__ import division
from cxi_xdr_xes.per_pixel_gain.photon_field import simulated_image_manager
from scitbx.array_family import flex

x_dim = y_dim = 100
n_images = 1000

def get_all_data():
  all_data = []
  from dxtbx.format.Registry import Registry
  format_class = Registry.find("data/data_series_0000.cbf")
  for t in xrange(n_images):
    file_name ="data/data_series_%04d.cbf"%t
    i = format_class(file_name)
    raw_data = i.get_raw_data()
    selected_block = raw_data.matrix_copy_block(0,0,x_dim,y_dim).as_double()
    all_data.append(selected_block)
  return all_data

def sum_images():
  sum_image = flex.double(x_dim*y_dim,0)
  data = get_all_data()
  for t in xrange(n_images):
    sum_image += data[t]
  return sum_image

if __name__ == "__main__":
  sum_image = sum_images()
  avg_image = sum_image/n_images
  avg_image = avg_image.iround()
  manager = simulated_image_manager()
  manager.write_image(avg_image,image_number = 1000,image_name="avg_summed")
