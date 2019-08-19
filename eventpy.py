#############################################################################################################
#This library is made to work with event data from the DVS sensor in the Address Event representation format
# Author: Ashish Rao M
# email: ashish.rao.m@gmail.com
##################################################################################################################
import math
import time
import utils
import numba
import pickle
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from scipy.signal import correlate


def timing(f):
  '''
    Wrapper function to calculate execution times of other functions
  '''
  @wraps(f)
  def wrapper(*args, **kwargs):
      start = time.time()
      result = f(*args, **kwargs)
      end = time.time()
      print('Elapsed time for function {} : {}'.format(str(f), end-start))
      return result
  return wrapper


def init_phase_map(camera_dims):
  ''' 
    Function to define the intial size of the depth map matrix. Based on the size of the camera dimensions.

    Parameters:
    -----------
    camera_dims: <int tuple>
      Species resolution of the camera the event data comes from 

    Return:
    -------
    phase_map_list: <np.array of dimensions cam_width x cam_height with each index containing a list>
      Matrix of said dimensions containing empty lists at all positions
  '''
  phase_map_list = [[[] for j in range(camera_dims[1])] for i in range(camera_dims[0])] # investigate if this can be made a numpy array later(list need because need to average out noisy measurements may be there)
  return phase_map_list 


def compute_firing_rate(events, del_t, pixel_location):
  '''
    Compute and plot the firing rate of a particular pixel for a time interval defined by del_t. The pixel under observation is defined by the pixel_location. 
  
    Parameters:
    -----------
    events: <list> 
          List of event tuples

    del_t: <float>
            binning interval (in microseconds)

    pixel_location: <tuple> 
          Specify the location of the pixel whose firing rate has to be observed

    Return:
    --------
    firing_rates: <dict> 
      Dictionary of firing rates of all events for different time indices of resolution del_t starting from 0. 

    pos_firing_rates: <dict>
      Dictionary of firing rates of positive events for different time indices of resolution del_t starting from 0.

    neg_firing_rates: <dict>
      Dictionary of firing rates of negative events for different time indices of resolution del_t starting from 0.
  '''
  time, rate, pos_firing_rate, neg_firing_rate = events[0][0], 0, 0, 0
  firing_rates = dict()
  pos_firing_rate_dict = dict()
  neg_firing_rate_dict = dict()
  for event in events:
    if event[1]==pixel_location[1] and event[2]==pixel_location[0]:   
      #print(event)
      if event[0] < (time + del_t):
        
        if event[3] == 1:
          pos_firing_rate+= 1
        else:
          neg_firing_rate+= 1
        rate += 1
      else:
        print(rate)
        firing_rates[time] = rate # assign to previous bin or to current bin??
        pos_firing_rate_dict[time] = pos_firing_rate
        neg_firing_rate_dict[time] = neg_firing_rate
        
        # reset counters and move to next time bin
        rate, pos_firing_rate, neg_firing_rate = 0, 0, 0
        time = time + del_t

        if  event[0] < (time + del_t):
          if event[3] == 1:
            pos_firing_rate+= 1
          else:
            neg_firing_rate+= 1
        
          rate += 1     
  print(rate)
  
  # assigment for the last time bin
  firing_rates[time] = rate 
  pos_firing_rate_dict[time] = pos_firing_rate
  neg_firing_rate_dict[time] = neg_firing_rate
  return firing_rates, pos_firing_rate_dict, neg_firing_rate_dict


def compute_phase_change(event, ref_time, P_signal):
  '''
    Compute phase change at each pixel in the image sensor due to moving 
    patterns which generate events at varying rates
    
    Parameters:
    -----------
    event:<list>
          A single event in [time, x, y, polarity] format

    ref_time:<float> 
            Start time of the signal to be used as reference

    P_signal: <int>
            The time period of the signal changing at one pixel

    Return:
    --------
    phase_change: <float>
      Return the wrapped phase computed by the formula defined in the function

  '''
  time, y, x, _ = event #x and y are interchanged because the convention followed is different in code and in the dvs_drivers
  
  phase_change = 2*np.pi*((time - ref_time) % P_signal) / P_signal 
  
  return phase_change

@timing
def dict_single_pixel_events(events, pixel_location):
  '''
    Compute the firing rate of a particular pixel for a time interval defined by del_t. The pixel under observation is defined by the pixel_location. 

    Parameters:
    -----------
    events: <list> 
          List of event tuples

    pixel_location: <tuple> 
          Specify the location of the pixel which has to be observed

    Return:
    --------
    pixel_event_dict: <dict> 
      Dictionary of events happening at pixel location given in the input parameters 

  '''
  pixel_event_dict = dict()
  for event in events:
    if event[1]==pixel_location[1] and event[2]==pixel_location[0]:  
      if event[3] == 1:
        pixel_event_dict[event[0]] = event[3]
      else:
        pixel_event_dict[event[0]] = -1
  return pixel_event_dict


def get_wrapped_phase_map(events, phase_map, P_signal=20, ref_x=10):
  '''
    Compute and plot the firing rate of a particular pixel for a time interval defined by del_t. The pixel under observation is defined by the pixel_location. 

    Parameters:
    -----------
    events: <list> 
          List of event tuples

    phase_map: <np.array of dimensions cam_width x cam_height with each index containing a list>
            Matrix of said dimensions containing empty lists at all positions

    P_signal: <int>
            The time period of the signal changing at one pixel
    
    ref_x: <int> 
            Reference row for phase unwrapping

    Return:
    --------
    phase_map_mat: <np.array of dimensions cam_width x cam_height>
            The matrix containing the wrapped phase values at each location
  '''
  ref_time = 0
  print('calculating Wrapped Phase')
  for event in events:
    if event[2] == ref_x:
      phase_map[int(event[2])][int(event[1])].append(0)
      ref_time = event[0] 
   
    else:
      phase_map[int(event[2])][int(event[1])].append(compute_phase_change(event, ref_time, P_signal))

  phase_map_new = [[sum(np.array(phase_map[i][j])) / len(phase_map[i][j]) if len(phase_map[i][j])!= 0 else 0 for j in range(346)] for i in range(260)]
  phase_map_mat = np.array(phase_map_new)
  print('Wrapped phase Calculation Completed')
  return phase_map_mat



def resampled_signal_lists_with_zeros(events_dict, xlimits):
  '''
    Resamples the given signal with precision determined by the round function of the math library and pads the signal with zeros wherever the signal is not present
    
    Parameters:
    -----------
    events_dict: <dict> 
      Events_dict is a dictionary with timestamp as key and signal magnitude is the value of the key. 

    xlimits: <list>
      xlimits is an array containing the start and stop time of the signal.
      
    Return:
      -------
    t_list: <list>
        The resampled time indices
    s_list: <list>
        The resampled signal values corresponding to time indices in t_list 
        
  '''
  t_stamps_list = list(events_dict.keys())
  t_list = list(np.arange(int(math.floor(xlimits[0])), int(math.ceil(xlimits[1])), 0.005))
  t_list = [round(t, 2) for t in t_list]
  s_list = list()
  time_keys = [round(t, 2) for t in t_stamps_list]
  i = 0
  for t in t_list:
    if i < len(t_stamps_list):
      if t==time_keys[i]:
        s_list.append(events_dict[t_stamps_list[i]])
        i+=1
      else:
        s_list.append(0)
    else:
      s_list.append(0)
  return t_list, s_list

def fill_shadows_in_phase_map(phase_map, shadow_indices_list, ref_row, sig_period, scan_speed):
  '''
    Some regions of the phase map record zero phase because of shadows. This function fills the shadowy regions of the phase map by computing row where fringes were supposed to be
    
    Parameters:
    -----------
    phase_map: <np.array> 
      Matrix containing wrapped phase values

    shadow_indices_list: <list>
      A list containing tuples of shadow indices in the image
    
    ref_row: <int> 
      Reference row for phase unwrapping

    sig_period : <float>
      Time period of the signal

    scan_speed: <int>
      Scanning speed of the pattern in pixels per second
    
    Return:
    -------
    phase_map: <np.array> 
      Matrix containing wrapped phase values compensated for shadows
  '''
  # sort indices list by row number
  sorted_shadow_list = sorted(shadow_indices_list, key = lambda x: x[0])
  shadow_vals = np.zeros(phase_map.shape)
  # take first value, take value of preceding row in the same column of the phase map/reference row, add standard estimated phase difference(from ref row to the current row) to this value and iterate through all indices in the list in the same fashion
  for index in sorted_shadow_list:
    shadow_vals[index[0], index[1]] = 2*np.pi*(((ref_row - index[0])/scan_speed)%sig_period)/sig_period
  shadow_comp_phase_map = phase_map + shadow_vals 
  return shadow_comp_phase_map


def convert_to_xyz_and_store(filename, depth_map_matrix):
  '''
  Convert the depth map values to xyz file and store in the current directory
  
  Parameters:
  -----------
  filename: <string> 
    Name of the file to be saved

  depth_map_matrix: <np.array>
    Matrix containing depth values at all position of the image
  '''
  f = open(filename,"w+")
  for x in range(depth_map_matrix.shape[0]):
    for y in range(depth_map_matrix.shape[1]):
      if depth_map_matrix[x][y]:
        f.write("{}\t{}\t{}\n".format(x, y, depth_map_matrix[x][y]))
  num_lines = np.sum([1 for line in f])
  f.close()
  print("finished preparing {}. It has {} lines".format(filename, num_lines))


def calculate_shadows(directory, filename, trial_num, object_type, camera_dims):
  '''
    Identify shadows in the image and store their indices in a pickle and text file.

    Parameters:
    -----------
    directory: <string> 
      Directory name

    filename: <string>
      filename of the stored file
    
    trial_num: <string> 
      the trial num at which data was captured

    object_type : <string>
      Describe the object eg: Teddy, righthand, swan, etc

    camera_dims: <tuple>
      A tuple containg the dimensions of the camera    
  '''
  events = utils.read_data('data/' + directory + filename + '.txt')
  event_mat = create_event_matrix(events, camera_dims)
  shadow_indices_list = extract_shadow_indices(event_mat)
  with open('shadows/' + object_type + '/' + trial_num + filename +'_shadow_pickle', 'wb') as fp:
      pickle.dump(shadow_indices_list, fp)
  with open('shadows/' + object_type + '/' + trial_num + filename +'_shadow_tuple.txt', 'w') as f:
      for shadow_indices_tuple in shadow_indices_list:
          f.write("{}\n".format(str(shadow_indices_tuple)))


def create_reference_signal(camera_dims, ref_row, scan_speed, sig_period):
  '''
    Synthetically create the wrapped phase of reference signal(pattern without object placed) of camera dimensions 

    Parameters:
    -----------
    camera_dims: <tuple>
      A tuple containg the dimensions of the camera    

    ref_row: <int> 
      Reference row for phase unwrapping
    
    scan_speed: <int> 
      Speed in pixels per second at which the pattern moves

    sig_period : <float>
      Time period of the signal 

    Return:
    --------
    ref_map: <np.array>
      Wrapped phase of a reference signal
  '''
  ref_map = np.zeros(camera_dims)
  for col in range(camera_dims[1]):
    for row in range(camera_dims[0]):
      ref_map[row, col] = 2*np.pi*(((ref_row - row)/scan_speed)%sig_period)/sig_period
  return ref_map

#fix this function
def circular_shift_reference_signal(ref_map, offset):
  '''
    Circularly shift the pattern in the direction defined by 'axis' in the implementation

    Parameter:
    ----------
    ref_map: <np.array>
      The matrix to perform circular shift on

    offset: <int>
      The number of pixels by which the matrix has to be shifted

    Return:
    -------
    ref_map: <np.array> 
      The circularly shifted matrix
  '''
  ref_map=  np.roll(ref_map, offset, axis=0)
  return ref_map

@timing
def create_event_matrix(events, camera_dims):
  '''
    Function to create matrix containing events at each pixel stored in the form of a list. 
    This makes the compute phase function faster by converting the problem to a simple lookup

    Parameter:
    ---------
    events: <list>
      A list of events 

    camera_dims: <tuple>
      A tuple containg the dimensions of the camera  
    
    Return:
    -------
    event_mat_of_dict: <np.array containng lists>
      A matrix of camera dimensions containing events corresponding to that location of the camera
  '''
  event_mat_of_dict = [[{} for j in range(camera_dims[1])] for i in range(camera_dims[0])]
  for event in events:
    event_mat_of_dict[int(event[2])][int(event[1])].update({event[0]:event[3]}) # The event list is [timestamp, y, x, polarity]
  return event_mat_of_dict

@timing
def calc_wrapped_phase_list_in_matrix(event_mat_of_dict, phase_map, camera_dims, ref_row, Period_signal, exp_time_lim):
  '''
    Calculate wrapped phase as a list element for each matrix. (While each element is calculated only once, 
    it was made a list entry because of some design decisions taken prior. Not a good choice :(. However, 
    things are restored to normal later using another function)
    
    Parameter:
    ---------
    event_mat_of_dict: <np.array containng lists>
      A matrix of camera dimensions containing events corresponding to that location of the camera

    phase_map: <np.array of dimensions cam_width x cam_height with each index containing a list>
      A matrix of camera dimensions containing empty lists at each location
    
    camera_dims: <tuple>
      A tuple containg the dimensions of the camera

    ref_row: <int> 
      The reference row to calculate phase w.r.t
      
    Period_signal: <float>
      Time period of the signal 
    
    exp_time_lim: <list>
    start and end times of the signal
    
    Return:
    -------
    phase_map: <np.array containng lists>
      A matrix of camera dimensions containing a single element list of wrapped phase corresponding to that location of the camera
  '''
  for col in range(camera_dims[1]):
    ref_pixel_events = event_mat_of_dict[ref_row][col]
    t_ref, s_ref = resampled_signal_lists_with_zeros(ref_pixel_events, xlimits=exp_time_lim)
    t = time.time()
    print('starting calculations for column {}'.format(col))
    for row in range(camera_dims[0]):
      cur_pixel_events = event_mat_of_dict[row][col]
      t_cur, s_cur = resampled_signal_lists_with_zeros(cur_pixel_events, xlimits=exp_time_lim)

      output = correlate(s_ref, s_cur, mode='same')
      maxpos = np.argmax(output)
  
      phase = 2*np.pi*((t_ref[maxpos] - t_cur[0]) % Period_signal)/Period_signal
      phase_map[row][col].append(phase)
    print('Finished calculations for column {} in {} seconds '.format(col, time.time() - t))
  return phase_map

def extract_shadow_indices(event_mat_of_dict):
  '''
    Get the indices of values in the matrix which have an empty dictionary and return those values
  
    Parameters:
    -----------
    event_mat_of_dict: <np.array containing lists>
      A matrix of camera dimensions containing events corresponding to that location of the camera
    
    Return:
    -------
    shadow_indices_list: <list>
      list of shadow indices (tuples) in the image
  '''
  shadow_indices_list = []
  for row in range(len(event_mat_of_dict)):
    for col in range(len(event_mat_of_dict[0])):
      if not bool(event_mat_of_dict[row][col]):
        shadow_indices_list.append((row, col))
  return shadow_indices_list