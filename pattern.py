#######################################################################################
#This library is used to generate moving patterns
# Author: Ashish Rao M
# email: ashish.rao.m@gmail.com
#######################################################################################

import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc


class Pattern:
    '''
    General structure for moving patterns
        
    Init Parameters:
    ----------------
    height: <int>
        Height of video resolution

    width: <int>
        width of the video resolution
    
    fps: <int>
        Number of frames per seconds

    '''
    def __init__(self, height, width, fps):
        self.width = width
        self.height = height 
        self.fps = fps
        self.fourcc = VideoWriter_fourcc(*'MP42')


class Line(Pattern):
    '''
    Class to generate moving lines of different thicknesses, orientations and speeds
    
    Init Parameters:
    ----------------
    thickness: <int>
        pixelwise thickness of the moving line pattern

    '''
    def __init__(self, height, width, fps, thickness):

        Pattern.__init__(self, height, width, fps)
        self.thickness = thickness

    def generate_moving_line(self, direction):
        '''
            Generate moving line whose direction is defined by the variable direction
        
        Parameters:
        -----------
        direction: <string>
            Defines the direction in which the line has to move

        Return: None
        ------------
        '''
        self.video = VideoWriter('./videos/' + direction + 'line_pattern_gen_'+str(self.fps)+'_bit.avis', self.fourcc, float(self.fps), (self.width, self.height))
        
        if direction=='vertical':
            print('generating video of vertical line')
            for x_coord in range(0, self.width):
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.line(frame, (x_coord, 0), (x_coord, self.height), (0, 0, 255),self.thickness)
                self.video.write(frame)
        
        if direction=='horizontal':
            print('generating video of horizontal line')
            for y_coord in range(0, self.height):
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.line(frame, (0, y_coord), (self.width, y_coord), (0, 0, 255), self.thickness)
                self.video.write(frame)
        
        if direction=='tl_diag':
            print('generating line from top left')
            for i in range(0, 2*self.width):
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.line(frame, (0,i), (i,0), (0, 0, 255), self.thickness)
                self.video.write(frame) 
        
        if direction=='bl_diag':
            print('generating line from bottom left')
            for i in range(2*self.width):
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                cv2.line(frame, (0,self.width-i), (i,self.width), (0, 0, 255), self.thickness)
                self.video.write(frame) 
        
        cv2.destroyAllWindows()
        self.video.release()

class Wave_2d(Pattern):
    '''
    Class to generate moving wave patterns
    
    Init Parameters:
    ----------------
    frequency: <int>
        Spatial frequency of the wave patterns
    
    pixel_shifted: <int>
        Number of pixels to shift by in two successive frames of the video 

    '''
    
    # generate moving  patterns 
    def __init__(self, height, width, fps, frequency, pixel_shifted):
        Pattern.__init__(self, height, width, fps)
        self.f = frequency
        self.pixel_shift = pixel_shifted

    def generate_moving_wave(self, direction, type):
        '''
        Generate moving wave whose direction is defined by the variable, direction.
        
        Parameters:
        -----------
        direction: <string>
                Defines the direction in which the wave has to move

        type: <string>
            The type of wave being generated> Square and Sinusoidal waves have been implemented for now.

        Return: None
        -------
        '''
        self.video = VideoWriter('./videos/' + type +'/' + direction + '_'+type + '_' + str(self.f) + 'hz_pattern_gen_'+str(self.fps) + '_fps.mkv', self.fourcc, float(self.fps), (self.width, self.height))
        
        if direction=='horizontal':
            print('generating video of horizontal wave based phase patterns')
            dt =  1/self.width
            x = np.arange(0, self.width, dt)
            if type == 'sine':
                y = np.sin(2 * np.pi * x * self.f) 
            if type == 'square':
                y = signal.square(2 * np.pi * x * self.f)

            
            y += max(y) # to shift range of signals to positive values

            frame = np.array([[y[j]*127 for j in range(self.width)] for i in range(self.height)], dtype=np.uint8) # create 2-D array of sine-wave
            
            for _ in range(0, self.width):
                self.video.write(cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB))
                shifted_frame =  np.roll(frame, self.pixel_shift, axis=1)  # simulated the circular shifting effect to make it a continuous
                frame = shifted_frame 

        if direction=='vertical':

            print('generating video of horizontal wave based phase patterns')
            dt =  1/self.height
            x = np.arange(0, 1, dt)
            if type == 'sine':
                y = np.sin(2 * np.pi * x * self.f) 
            if type == 'square':
                y = signal.square(2 * np.pi * x * self.f)
            
            y += max(y)  # to shift range of signals to positive values

            frame = np.array([[y[j]*127 for j in range(self.width)] for i in range(self.height)], dtype=np.uint8).T  # create 2-D array of sine-wave. Transpose it to make it vertical
            
            for _ in range(0, self.height):
                self.video.write(cv2.cvtColor(frame,cv2.COLOR_GRAY2RGB))
                shifted_frame =  np.roll(frame, self.pixel_shift, axis=0)
                frame = shifted_frame 
            
        cv2.destroyAllWindows()
        self.video.release()
    
        print('generation of moving patterns done')
            
