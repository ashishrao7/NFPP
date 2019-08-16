#######################################################################################
# This is a utility library for common methods 
# Author: Ashish Rao M
# email: ashish.rao.m@gmail.com
#######################################################################################
import numpy as np 
import matplotlib.pyplot as plt
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import os
import cv2


def read_data(path):
  '''
    Function to read data from the file containing events

    Parameters:
    -----------
    path: <string>
      The location of data to be read from
    
    Return:
    -------
    events: <list>
      A list of events read from the file 
  '''
  file = open(path, "r")
  print('Reading events from file ')
  events = [list(map(float,line.split())) for line in file]
  start_time = events[0][0]
  file.close()
  print('Events have been read!')
  events = np.array(events, dtype=np.float_)
  events[: , 0] = events[:, 0] - start_time 
  return events

def create_new_event_file(filename):
  '''
  This function creates a new file. That's all!
  '''
  f= open(filename,"w+")
  return f

def append_to_event_file(file, event):
  '''
  This function is as useless as the earlier function. I don't know what i was thinking
  '''
  file.write(event)
  

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
  for x in tqdm(range(depth_map_matrix.shape[0])):
    for y in range(depth_map_matrix.shape[1]):
      f.write("{}\t{}\t{}\n".format(x, y, depth_map_matrix[x][y]))
  num_lines = sum([1 for line in f])
  f.close()
  print('finished preparing {}. The file has {} lines'.format(filename, num_lines))



def plot_dictionary(data, title, xlimits, xlabel, ylabel, type='default'):
  '''
    Plot the values of a dictionary

    Parameters:
    -----------
    data: <dict>
      The dictionary to be plotted
    
    title: <string>
      Yeah you know what this means. why are you even reading this
    
    xlimits: <list>
      The start and stop values on the x axis
    
    xlabel: <string>
      Seriously?

    ylabel: <string>
      Seriously? Seriously?
    
    type: <string>
      To make different types of plots. Currently only stem and linear interpolation have been implemented 
  
  '''
  lists = sorted(data.items()) # sorted by key, return a list of tuples
  x, y = zip(*lists) 
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.xlim(xlimits)
  plt.ylim(min(y) - 1 , max(y) + 1)
  if type=='stem':
    plt.stem(x, y)
  else:
    plt.plot(x, y)
  plt.savefig('plots/' + title + '.png')
  plt.show()
  
def make_video(image_folder):  

  '''
  Make video from images. 

  Parameters:
  ----------
  image_folder: <string>
    directory of the folder containing the images
  '''

  video_name = 'event_simulation.avi'
  images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
  frame = cv2.imread(os.path.join(image_folder, images[0]))
  height, width, _ = frame.shape
  video = cv2.VideoWriter(video_name, 0, 30, (width,height))
  for image in images:
      video.write(cv2.imread(os.path.join(image_folder, image)))
  cv2.destroyAllWindows()
  video.release()