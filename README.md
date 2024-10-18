# Perceptual curvature estimation 

This repository contains code for the perceptual curvature estimation algorithm introduced and used in the following papers 📚:

> [1] Hénaff, O. J., Goris, R. L., & Simoncelli, E. P. (2019). Perceptual straightening of natural videos. Nature neuroscience, 22(6), 984-991.
> 
> [2] Hénaff, O. J., Bai, Y., Charlton, J. A., Nauhaus, I., Simoncelli, E. P., & Goris, R. L. (2021). Primary visual cortex straightens natural video trajectories. Nature communications, 12(1), 5982.

The code is tested on simulated trajectories where the ground truth curvature is known. The quality of the estimation is tested by performing a recovery analysis. That is, 

<p float="middle">
  <img src="https://github.com/Hiyeri/roiconnect/blob/master/resources/power_barplot_left.jpg?raw=true" width="400"/>     
  &nbsp; &nbsp;
</p>

Requires the [MATLAB Engine API for Python](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html), which itself requires the most version of MATLAB. 
