# Virtual-to-Real Mirrors/Windows (V2R)

Code for our paper: 

**"Keep It Real: a Window to Real Reality in Virtual Reality"** 

by [Baihan Lin](http://www.columbia.edu/~bl2681/) (Columbia). 



All the experimental results and analysis can be reproduced using the code in this repository. Feel free to contact me by doerlbh@gmail.com if you have any question about our work.



**Abstract**

This paper proposed a new interaction paradigm in the virtual reality (VR) environments, which consists of a virtual mirror or window projected onto a virtual surface, representing the correct perspective geometry of a mirror or window reflecting the real world. This technique can be applied to various videos, live streaming apps, augmented and virtual reality settings to provide an interactive and immersive user experience. To support such a perspective-accurate representation, we implemented computer vision algorithms for feature detection and correspondence matching. To constrain the solutions, we incorporated an automatically tuning scaling factor upon the homography transform matrix such that each image frame follows a smooth transition with the user in sight. The demonstration is a real-time rendering framework where users can engage their real-life presence with the virtual space.



## Info

Language: Python3

Platform: MacOS, Linux, Windows

by Baihan Lin, Jan 2020




## Citation

If you find this work helpful, please try the models out and cite our work. Thanks!

    @article{lin2020keep,
      title={{Keep It Real: a Window to Real Reality in Virtual Reality}},
      author={Lin, Baihan},
      journal={under review},
      year={2020}
    }

  

## Tasks

* Feature detection for mirror location to be projected in virtual space
* Homography computation for projective mirror transformation
* Augmented reality to project real-world to live feed in real-time



## Requirements

* Python 3
* OpenCV
* numpy 



## Example videos of this augmented reality app

![](./asset/vid_2.gif "")

![](./asset/vid_3.gif "")

![](./asset/vid_4.gif "")

![](./asset/vid_8.gif "")

![](./asset/vid_1.gif "")

![](./asset/vid_7.gif "")