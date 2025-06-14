# Photo-Album
This project is written in MATLAB and aims to convert a set of images into a video with transition effects. The input is all .jpg image files from a specified directory, and the output is an .mp4 video. 
The effects include Simple Fade (Alpha Dissolve),Checkerboard Fade,Left-to-Right Wipe,Top-to-Bottom Wipe,Random Block Dissolve,Vertical Venetian-Blind Wipe,RadialCenter-Out Reveal,Diagonal Wipe,Horizontal Wave Distort Fade,Pixelation Dissolve,Horizontal Venetian-Blind Wipe,Cross (Plus-Shape) Wipe,Dual-Diagonal Wipe,Expanding CentralRectangle Reveal,Diamond-Shape Reveal,Top-Down Gradient Overlay,Staggered Color-Channel Fade,Random Line (Row) Dissolve

1. Quick Start 
First, put source images (`.jpg`) in the `images/` folder (keep the order you want). 
Then, launch MATLAB with the Image Processing Toolbox. 
Third, Run `photo_album.m`. Frames are written to `video_images/`; the final video is exported as `photo_album.mp4` (24 fps, H.264). 
Tips: Change `main_dir`, `file_type`, or `animation.FrameRate` in `photo_album.m` to suit your own project.

2. Major Requirements
MATLAB R2018b or later  
Image Processing Toolbox (for `imresize`, `imtranslate`, `imrotate`)  
