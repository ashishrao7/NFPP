import matplotlib.pyplot as plt 
import numpy as np
import utils
import eventpy
def main():
  camera_dims = (180,240)  # Dimensions of a DAVIS240
  pixel_location = (6,8) # location of pixel whose firing rate is to be studied
  filename = 'events'
  directory = 'slider_depth/'
  events = utils.read_data('data/' + directory + filename + '.txt')

  single_pixel_events = eventpy.dict_single_pixel_events(events, pixel_location)
  single_pixel_intensity = utils.read_intensity_at_location(pixel_location, 'data/slider_depth/', 'data/slider_depth/images.txt', log='yes')
  delta_mod_single_pixel_events = eventpy.dict_delta_mod_single_pixel_events(events, pixel_location, 2)

  utils.plot_dictionary(data=single_pixel_events, title='Events of pixel ({},{}) for the file {}'.format(pixel_location[0], pixel_location[1], filename),
                        xlimits=(events[0][0], events[-1][0]), xlabel='Time in s', ylabel='Events', type='stem')

  utils.plot_dictionary(data=delta_mod_single_pixel_events, title='Delta Modulated Log Intensity of pixel ({},{}) for {}'.format(pixel_location[0], pixel_location[1], filename),
                        xlimits=(events[0][0], events[-1][0]), xlabel='Time in s', ylabel='Log Intensity', type='step')

  utils.plot_dictionary(data=single_pixel_intensity, title='Intensity of pixel ({},{}) for the file {}'.format(pixel_location[0], pixel_location[1], filename),
                        xlimits=(events[0][0], events[-1][0]), xlabel='Time in s', ylabel='Intensity')
  
  utils.make_video('data/slider_depth/images/', video_name='videos/slider_depth.avi')

  utils.plot_multiple_dictionaries(single_pixel_intensity, delta_mod_single_pixel_events, single_pixel_events)

  utils.compare_plots(delta_mod_single_pixel_events, single_pixel_events, 'Log Intensity', 'Event Data')

if __name__=='__main__':
    main()