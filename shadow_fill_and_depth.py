#############################################################################################################
# This file fills shadows in the phase map and calculates the depth map using triangulation principles
# Author: Ashish Rao M
# email: ashish.rao.m@gmail.com
##################################################################################################################
import cv2
import time
import utils
import pickle
import eventpy
import numpy as np 
import matplotlib.pyplot as plt
from scipy import ndimage, io
from matplotlib.image import imsave
from skimage.exposure import histogram
from skimage.restoration import unwrap_phase

object_type = 'righthand'
trial_num = 'trial_1/'
spatial_freqency = 10    

'''
# 1 -> righthand, trial_1, f = 10
# 2 -> ball, trial_2 , f = 20
# 3 -> swan, trial_1, f = 10 , cyclic shift -> 20 
'''

camera_dims = (260,346)                                        # Dimensions of a DAVIS346
scan_speed = 20                                                # fps of video          
Period_signal = camera_dims[0]/(scan_speed*spatial_freqency)   # The time period of the temporally changing signal

# Trivia : adjacent_pixel_phase_t_difference = 2*np.pi*scan_speed

object_name = object_type + '_' + str(spatial_freqency) + 'hz_'
filename = 'events_' + object_name + '20fps'
directory = 'patterns_on_objects/' + trial_num
phase_map = eventpy.init_phase_map(camera_dims)

# comment out the below line if the shadow calculation has already been done, uncomment to do the calculations
eventpy.calculate_shadows(directory, filename, trial_num, object_type, camera_dims)


with open('shadows/' + object_type + '/' + trial_num + 'events_' + object_type + '_'+ str(spatial_freqency)+ 'hz_' + '20fps' + '_shadow_pickle', 'rb') as f:
    shadow_indices_list = pickle.load(f)

# read the pickle file 
mat_contents = io.loadmat('results/' + trial_num + 'phase_matrix_' + object_type +'_'+ str(spatial_freqency) + 'hz_')
phase_matrix = mat_contents['phase_matrix']



############################# Shadow filling using background captured ###############################
mat_contents = io.loadmat('results/' + trial_num + 'phase_matrix_background_' + str(spatial_freqency) + 'hz_')
compensator_im = mat_contents['phase_matrix']
compensator_im = eventpy.circular_shift_reference_signal(compensator_im, 0)   # circularly shift reference signal 
for shadow_index in shadow_indices_list:
    phase_matrix[shadow_index[0], shadow_index[1]] = compensator_im[shadow_index[0], shadow_index[1]]
#######################################################################################  


'''#########################################Shadow filling using pre generated background ###################################################
phase_matrix = eventpy.fill_shadows_in_phase_map(phase_matrix, shadow_indices_list, 10, Period_signal, scan_speed)
compensator_im = eventpy.create_reference_signal(camera_dims, 10, scan_speed, Period_signal)
################################################################################################################'''


image_unwrapped = unwrap_phase(phase_matrix)
#image_unwrapped -= np.min(image_unwrapped)
io.savemat('unwrapped_phase_matrix_' + object_name + '.mat', {'unwrapped_phase_matrix':image_unwrapped}) 

ref_phase_map = unwrap_phase(compensator_im)
#ref_phase_map -= np.min(ref_phase_map)

depth_map = -(ref_phase_map - image_unwrapped) * 0.56 * 100 / (2 * 0.125 * spatial_freqency * np.pi) # del_phi * distance_to_reference_plane/(2 * pi * spatial_frequency * baseline) 
depth_map -= np.min(depth_map)

depth_image = depth_map.astype(np.uint8)    # inorder to apply median filter
depth_image = cv2.medianBlur(depth_image,3) # median filter


'''val, bin_c = histogram(depth_image)
plt.plot(bin_c, val)
plt.show()
depth_image = depth_image*(depth_image < bin_c[np.argmax(val)])
'''


cv2.imwrite("images/depth_" + filename + ".png", depth_image) #save depth images somewhere 


############################################ Plotting Results ############################################
fig, axs = plt.subplots(2, 2) 

im1 = axs[0, 0].imshow(phase_matrix, cmap='gray')
axs[0, 0].set_title('Wrapped Phase Image')
axs[0, 0].axis('off')
fig.colorbar(im1, ax=axs[0, 0])

im2 = axs[0, 1].imshow(image_unwrapped, cmap='gray')
axs[0, 1].set_title('Unwrapped  Image')
axs[0, 1].axis('off')
fig.colorbar(im2, ax=axs[0, 1])
io.savemat('image_unwrapped.mat', {'image_unwrapped':image_unwrapped}) 

im3 = axs[1, 0].imshow(depth_image, cmap='gray')
axs[1, 0].set_title('Depth Map')
axs[1, 0].axis('off')
fig.colorbar(im3, ax=axs[1, 0])


im4 = axs[1, 1].imshow(ref_phase_map, cmap='gray')
axs[1, 1].set_title('Reference phase Map')
fig.colorbar(im4, ax=axs[1, 1])
io.savemat('carrier.mat', {'carrier_unwrapped_matrix':ref_phase_map}) 

plt.tight_layout()
plt.show()
############################################################################################################

eventpy.convert_to_xyz_and_store(object_name + ".xyz", depth_image) # convert to file
