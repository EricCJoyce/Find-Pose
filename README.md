# Borders
`borders.py` is a helper script that takes a 3D mesh (OBJ), along with all its texture maps, and creates new copies of the texture maps with the mesh's triangles drawn on. This is helpful for visualizing which corner of its texture maps the mesh considers to be the origin. You will need to know this in order to correctly locate 2D features in 3D space and estimate camera poses.

## Dependencies and Set-Up

### NumPy
http://www.numpy.org/
Needed for vector and matrix math
### OpenCV
https://opencv.org/
The `borders.py` script really only uses OpenCV to load and measure texture maps. You could replace this with any other image library, but the more intricate `find.py` script will put OpenCV to full use. More thorough notes on installing OpenCV are found in the "Dependencies and Set-Up" section under "Find-Pose."
### face.py
The classes in this file assist the `find.py` script. It should be in the same directory.

## Inputs

### Example Script Call
The only required argument is a 3D-mesh OBJ file, as seen here: `python borders.py mesh.obj`
Script performance can be modified by passing flags and arguments after the OBJ file. Please see these described below in "Parameters."

## Outputs

The script will create one new image file for every texture map in your OBJ mesh. These files will have the same name, plus `borders.` plus a two-character origin identifier, and finally the graphics extension of the original. These two-character identifiers are `ll` for "lower-left," `ul` for "upper-left," `ur` for "upper-right", and `lr` for "lower-right," indicating that the mesh considers (0, 0) to be in that corner. For example, if you called `python borders.py mesh.obj`, and the texture maps for `mesh.obj` are named `mesh_000.jpg`, `mesh_001.jpg`, `mesh_002.jpg`, etc. On assumption that the origin is in the lower-left, the resulting files would be `mesh_000.borders.ll.jpg`, `mesh_001.borders.ll.jpg`, `mesh_002.borders.ll.jpg`, etc.

## Parameters

### `-ll` Assume that the Origin is the Lower-Left
For example, `python borders.py mesh.obj -ll`

### `-ul` Assume that the Origin is the Upper-Left
For example, `python borders.py mesh.obj -ul`

### `-ur` Assume that the Origin is the Upper-Right
For example, `python borders.py mesh.obj -ur`

### `-lr` Assume that the Origin is the Lower-Right
For example, `python borders.py mesh.obj -lr`

### `-?`, `-help`, `--help` Help!
Display some notes on how to use this script.

# Find-Pose
Given an image, a 3D mesh (OBJ), and camera parameters, estimate where in the mesh was the camera that photographed the object in the image.

## Dependencies and Set-Up

### NumPy
http://www.numpy.org/
Needed for vector and matrix math
### OpenCV
https://opencv.org/
Needed for image manipulation, feature detection, and the PnP-solver.

This script only makes use of the Python installation of OpenCV, which is easy to do: https://docs.opencv.org/4.1.0/d2/de6/tutorial_py_setup_in_ubuntu.html. Other vision-realated repositories will need to compile C++ code using the OpenCV libraries, which is a more involved installation. The following worked for us, installing OpenCV 3.1.0 on Ubuntu 16.04:
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install build-essential cmake pkg-config
sudo apt-get install git libgtk-3-dev
sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev libdc1394-22-dev libeigen3-dev libtheora-dev libvorbis-dev
sudo apt-get install libtbb2 libtbb-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev sphinx-common yasm libfaac-dev libopencore-amrnb-dev libopencore-amrwb-dev libopenexr-dev libgstreamer-plugins-base1.0-dev libavutil-dev libavfilter-dev libavresample-dev
sudo apt-get install libatlas-base-dev gfortran
```
The foregone commands install all the requisite libraries. Now we install and build OpenCV:
```
sudo -s
cd /opt
wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.1.0.zip
unzip opencv.zip
wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip
unzip opencv_contrib.zip
mv /opt/opencv-3.1.0/ /opt/opencv/
mv /opt/opencv_contrib-3.1.0/ /opt/opencv_contrib/
cd opencv
mkdir release
cd release
cmake -D WITH_IPP=ON -D INSTALL_CREATE_DISTRIB=ON -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules /opt/opencv/
make
make install
ldconfig
exit
cd ~
```
If all went well, we should be able to query the version of the OpenCV installation:
```
pkg-config --modversion opencv
```
We should also be able to compile C++ code that uses OpenCV utilities. (Suppose we've written some such program, `helloworld.cpp`.)
```
g++ helloworld.cpp -o helloworld `pkg-config --cflags --libs opencv`
```
### face.py
The classes in this file assist the `find.py` script. It should be in the same directory.
### Matplotlib
https://matplotlib.org/
Only needed to create pop-up displays, if desired. See Parameters, section `-showSIFT`

## Inputs

### Example Script Call
The only required arguments are an image and a 3D-mesh OBJ file, as seen here: `python find.py image.jpg mesh.obj`
Script performance can be modified by passing flags and arguments after the OBJ file. Please see these described below in "Parameters."

## Outputs

The goal of this script is to estimate rotation and translation for the camera that took the given picture. A lot of intermediate work must be done in order to reach this end, and several files may be generated to help save time on subsequent script calls.

### No output format specified
When no output formats are specified in the command line, the script's default behavior is to output its rotation and translation estimates to the screen, both as 3-vectors.

## Parameters

### `-K` Specify a Camera Matrix
Some mathematical information about the camera that took the picture given to the algorithm is necessary to form an estimate. By default, the script looks for `K.mat`, which is really just a place-holder name. Whatever file you give to `-K` should be an ASCII text file with the following format:
* Required parameters are `fx`, `fy`, `cx`, `cy`, `w`, and `h`.
* Parameters each begin a single line and are separated with a tab from their values.
* Comment lines are ignored. They begin with the `#` character
* `fx` is the horizontal focal length in pixels
* `fy` is the vertical focal length in pixels
* `cx` is the X of camera's principal point (usually equal to half the image width) in pixels
* `cy` is the Y of camera's principal point (usually equal to half the image height) in pixels
* `w` is the image width in pixels
* `h` is the image height in pixels

These are the main mathematical details of a camera needed to make inferences about where observed points lie (we ignore such exotic factors as lens distortion). If you know your camera's model, look for its technical details here: https://www.digicamdb.com/. Our experiments used the SONY DSLR Alpha 580 (https://www.digicamdb.com/specs/sony_alpha-dslr-a580/), and the file `SONY-DSLR-A580-P30mm.mat` in this repository was made using the information on this page. Anticipate that you will have several `.mat` files for all the cameras you work with. The numbers in this script's equations change with the camera's focal length, and if the image given to it uses a landscape or a portrait composition. It was helpful to have a reminder of this; hence the "PORTRAIT" comment and the "P" in the file name `SONY-DSLR-A580-P30mm.mat`. For example: `python find.py image.jpg mesh.obj -K SONY-DSLR-A580-P30mm.mat`

### `-knn` Short for "K Nearest Neighbors"
This parameter sets the number of best candidates when finding matches among texture maps for features in the source photograph. Remember that feature-matching is at best a guess, so maybe we can improve our chances of striking upon a valid correspondence between a point in the photo and a point in the OBJ textures if we tell the script, "Find three points in the mesh that could match one point in the image." This is what `-knn` sets. It defaults to three. For example: `python find.py image.jpg mesh.obj -knn 5`

### `-x` Exclude an Image from Consideration
OBJ meshes are often packed with images other than texture maps. We will not find feature correspondences between our target image and a floorplan, so you can save time and computation by telling the script up front to ignore certain files belonging to the OBJ mesh. For example: `python find.py image.jpg mesh.obj -x ceilingcolorplan_000.jpg -x ceilingcolorplan_001.jpg`

### `-v` Enable Verbosity
This script runs an especially time-consuing algorithm. It is helpful for the process to exhibit signs of life. Enabling verbosity lets a user know what's going on. For example: `python find.py image.jpg mesh.obj -v`

### `-SIFT` Toggle the use of SIFT
### `-ORB` Toggle the use of ORB
### `-BRISK` Toggle the use of BRISK
SIFT, ORB, and BRISK are feature-detection technologies. They are particular methods for examining images and identifying points likely to be distinct. Often these are edges of light and dark, high-contrast corners, or blotches. As command-line parameters, each of these takes a following argument of either `Y` or `N`, depending on whether you want to enable or disable the use of that detector. All are enabled by default. Why would we ever want to disable something that could help find correspondences? Depending on your source image, some features will perform better than others. There is also a time-consideration; it is temporally expensive to find all features in all images with potentially several candidate matches. Pose-estimation suffers if it is drowned in "outliers": erroneous and misleading correspondences. Here's an example where we turn off SIFT and ORB, leaving BRISK: `python find.py image.jpg mesh.obj -SIFT N -ORB N`

### `-o` Add an Output Format
Several or no output formats can be specified on the command line. As mentioned above, when no output formats are given, the script prints its results to the screen as a pair of 3-vectors. The following output formats are recognized:
* `rt` Write the rotation and translation vectors to file
* `Rt` Write the rotation matrix and translation vector to file
* `P` Compute a projection matrix from the rotation and translation vectors and write that matrix to file
* `Meshlab` Write copy-pastable Meshlab markup to file. (The formatting of this markup may differ according to your version of Meshlab. It should be mentioned that I have not gotten this feature to work yet: https://sourceforge.net/p/meshlab/discussion/499533/thread/cc40efe0/)
* `Blender` Write copy-pastable Blender Python console code to file.

Any and all output formats are written to the same file, `find.log`. For example: `python find.py image.jpg mesh.obj -o Rt -o P -o Blender`

### `-m` Set Color to Mask Out
Feature detectors like those used in this script can be led astray by texture maps' "false edges." False edges are high-contrast areas in a texture map where a patch of mesh is next to a uniformly-colored, unused region. All the detectors see are numerically interesting disjunctions of pixels, and they sometimes mistake these for something in the source. We attempt to mitigate this by setting color masks. `-m` is followed by three integer components, one for red, one for green, and one for blue. When a color is masked out, the script will not admit for consideration any features that land inside the mask. This does risk keying out a color when it naturally occurs in a texture map, however this is unlikely in practice. What appear to us as pure black or pure white (colors often used to fill the "dead spaces" in texture maps) are actually muddles of finer valences. In our experiments, we found it helpful to key out pure black and pure blue: `python find.py image.jpg mesh.obj -m 0 0 0 -m 0 0 255`

### `-e` Set Mask Erosion
Simply keying out a color still leaves a misleading artificial edge, and an interest point placed right on this edge may not be ignored as we want it to be. Setting the `-e` parameter tells the script to eat away by the following number of pixels. `-e` must be used with at least one color mask, defined by `-m`. Essentially, masking out a color and then expanding that mask makes sure that the script discards from further consideration any features "too close" to an artificial edge. The hope is that, by discouraging erroneous features (ones which do not actually exist in the scene, but are artificts of texture-map packing) we leave only those features that capture some unique aspects of our target. Building on our example above: `python find.py image.jpg mesh.obj -m 0 0 0 -m 0 0 255 -e 5`

### `-reprojErr` Set RANSAC's Reprojection Error
The RANSAC algorithm evaluates its pose hypotheses by comparing where it thinks points should be with where those points really are. Like any estimate, this is subject to noise and floating-point precision. Setting `-reprojErr` tells the script within how many pixels it should consider a comparison ("reprojection" of the estimate against the evidence) "close enough." Set too high, and even a poor estimate will pass inspection; set too low, and no estimate will be good enough. This parameter defaults to 8. Set it like this: `python find.py image.jpg mesh.obj -reprojErr 5`

### `-iter` Set the Number of RANSAC Iterations
Set the number of iterations for which the RANSAC pose-estimation algorithm should run. By default, this number is 1000. Set it like this: `python find.py image.jpg mesh.obj -iter 10000`

### `-showSIFT` Show SIFT Matches
### `-showORB` Show ORB Matches
### `-showBRISK` Show BRISK Matches
When enabled, the script will pop up a window showing feature matches between the source image and every texture map for the respective detector. This is helpful initially, when you are getting a sense of how well your source is matched, but can become annoying when there is a large number of texture maps. These parameters do not require a following yes or no, though they obviously depend on the appropriate detector being enabled. For example, suppose we want to use SIFT, ORB, and BRISK (all enabled by default), but we only want to see matches for SIFT and BRISK: `python find.py image.jpg mesh.obj -showSIFT -showBRISK`

### `-showSIFTn` Set the Number of SIFT Matches to Draw
### `-showORBn` Set the Number of ORB Matches to Draw
### `-showBRISKn` Set the Number of BRISK Matches to Draw
Used in tandem with the `-show*` parameters, these tell the script how many matches to draw. The default is 40. Too many matches will crowd the image and tend to obscure which features are being detected. Too few matches does not give a thorough sense of how a given detector is performing. Obviously, these parameters only have application when their respective detectors are enabled, like so: `python find.py image.jpg mesh.obj -SIFT Y -showSIFT -showSIFTn 60`

### `-showMask` Show the Color Mask
When enabled, this parameter will pop up a window showing you the color mask being used. Like the other `-show*` parameters, this is helpful when initially getting a sense of which parameters will work best but would become tedious in a refined workflow. An example call masking out black, eroding edges by 5, and showing the mask: `python find.py image.jpg mesh.obj -m 0 0 0 -e 5 -showMask`

### `-showInliers` Generate an Image and a 3D PLY File for Points that Fit
To have some visible indication of how the algorithm performed, enabling this parameter tells the script to create a new image and a new 3D point-cloud (PLY) file once RANSAC finishes. The image `inliers.jpg` will be a black-and-white copy of the given image with features superimposed as color circles. The file `inliers.ply` will contain the same number of three-dimensional points, colored the same as their 2D counterparts. `python find.py image.jpg mesh.obj -showInliers`

### `-showFeatures` Generate an Image and a 3D PLY File for All Points
Another performance-monitoring parameter, enabling `-showFeatures` tells the script to create a new image and a new 3D point-cloud (PLY) file once feature-matching finishes. The image `features.jpg` will be a black-and-white copy of the given image with features superimposed as color circles. The file `features.ply` will contain the same number of three-dimensional points, colored the same as their 2D counterparts. `python find.py image.jpg mesh.obj -showFeatures`

### `-epsilon` Set Inter-Vertex Epsilon Distance
Redundant vertices may exist "triangle soup," that is, distinct vertices in the mesh that nevertheless exist at the same point in space, or within some negligible distance from each other. The default value is 0.0. Honestly this parameter does not have use in `find.py`. It does have use in the `vgroups.py` script when we expand groups of mesh faces and need to know whether two faces are close enough to connect. Since `-epsilon` is a user-settable parameter in the mutual helper class, `Mesh`, we include it here for completeness.

### `-?`, `-help`, `--help` Help!
Display some notes on how to use this script.
