# Phase of Events

This repository contains implementation of the Phase of Events method proposed by <a href="http://compphotolab.northwestern.edu/wordpress/wp-content/uploads/2015/04/dvs_031.pdf"> Rao Mangalore et. al</a>. 

The data in the experiments folder have been collected from a DAVIS346 Camera. The  <a href="https://github.com/uzh-rpg/rpg_dvs_ros">drivers </a> developed by the robotics and perception group were used to collect data from the DAVIS. 

An object is placed on a platform and a moving light source generated in `generator.py` is made to fall on the object. The videos folder contains different patterns used with the light source generated using the generator file.

The event camera, placed in a stereo configuration with the <a href="http://www.ti.com/tool/DLPLCR4500EVM">light projector</a> ,records the Wrapped phase of the incident light. The phase is then unwrapped and triangulated to get depth maps.


To run the files execute with the following commands

`python correlate_and_phase.py`

The above command would compute wrapped phase at all the pixel locations. We then run the following command,

`python shadow_fill_and_depth.py`

This commmand compensates for shadows and calculates the final unwrapped phase and the depth maps. The wrapped and unwrapped phase calculated for various objects are shown below,
<center><img src="images/wrapped_and_unwrapped.png"></center>

Note that the experiment has the correlate and phase script has to be first run for the reference fringes i.e We must first calculate wrapped phase without any objects placed. We then move on to change the object name to calculate depth maps for different objects. To run for different object change the `object_type` and `spatial_frequency` variables in the `main()` function in the `correlate_and_phase.py` file. The same applies to the `shadow_fill_and_depth.py` file. In case you are using a camera other than DAVIS346, camera dimensions can be changed by changing the `cam_dims` variable. The output is a .xyz file. The .xyz files can be imported in meshlab and meshed to get 3D results.

The depth maps obtained are as follows

<table style="width:100%">
    <tr>
        <td style="text-align:center"><img src="images/bright_scrnshots/ball.png"> </td>
        <td style="text-align:center"><img src="images/bright_scrnshots/hand.png"> </td>
        <td style="text-align:center"><img src="images/bright_scrnshots/swan.png"> </td>
    </tr>
</table>

To understand how events occur in an event based camera, please watch the video at this <a href="https://www.youtube.com/watch?v=kPCZESVfHoQ">link.</a>

Below are the links to different types of Event Cameras,

1. <a href="http://inivation.com/"> DAVIS </a>

2. <a href="https://www.prophesee.ai/"> ATIS </a>
