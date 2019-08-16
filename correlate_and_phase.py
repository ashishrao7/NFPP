import time
import utils
import eventpy
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import ndimage, io
from skimage.restoration import unwrap_phase


def main():
  object_type = 'swan'
  spatial_freqency = 10              # in Hertz
  trial_num = 'trial_1/'
  
  '''
    # 1 -> righthand, trial_1, f = 10 baseline 48
    # 2 -> ball, trial_2 , f = 20, baseline 56
    # 3 -> swan, trial_1, f = 10 , cyclic shift - 20 , baseline 48
  '''
  camera_dims = (260,346)            # Dimensions of a DAVIS346
  scan_speed = 20                    # Fps of video    
  Period_signal = camera_dims[0]/(scan_speed*spatial_freqency) 
  
  # Trivia : adjacent_pixel_phase_t_difference = 2*np.pi*scan_speed
  object_name = object_type + '_' + str(spatial_freqency) + 'hz_'
  filename = 'events_' + object_name + '20fps'
  
  directory = 'patterns_on_objects/' + trial_num
  phase_map = eventpy.init_phase_map(camera_dims)
  events = utils.read_data('data/' + directory + filename + '.txt')
  ref_row = 10
  
  ############################### Fast #################################################
  event_matrix_of_dicts = eventpy.create_event_matrix(events, camera_dims)
  exp_time_lim = (events[0][0], events[-1][0])
  phase_map = eventpy.calc_wrapped_phase_list_in_matrix(event_matrix_of_dicts, phase_map, camera_dims, ref_row, Period_signal, exp_time_lim)
  ######################################################################################
  
  phase_map_new = [[sum(np.array(phase_map[i][j])) / len(phase_map[i][j]) if len(phase_map[i][j])!= 0 else 0 for j in range(camera_dims[1])] for i in range(camera_dims[0])]  # average all values in the list at each element of the matrix to get a single value(bad design due to initial function design)
  phase_map = np.array(phase_map_new)  
  io.savemat('results/' + trial_num + 'phase_matrix_' + object_name + '.mat', {'phase_matrix':phase_map}) 
  image_unwrapped = unwrap_phase(phase_map)

  ###### Plot Wrapped Phase and Unwrapped Phase ####### 
  fig, axs = plt.subplots(2, 2)
  im1 = axs[0, 0].imshow(phase_map, cmap='gray')
  axs[0, 0].set_title('Wrapped Phase Image')
  fig.colorbar(im1, ax=axs[0, 0])

  im2 = axs[0, 1].imshow(image_unwrapped, cmap='gray')
  axs[0, 1].set_title('Unwrapped  Image')
  fig.colorbar(im2, ax=axs[0, 1])

  plt.tight_layout()
  plt.show()
  
if __name__=='__main__':
    main()