'''
To use the code
python3 dpivAnalysis.py <path to parameter file>
The parameter file is the same as the original from DPIV with 3 additional fields
input_folder: '<Images Folder>/'
output_folder: '<Results Folder>/'
extension: 'tif' extension of the images to be processed ('jpg', 'png', 'tif', etc.)

TO DO: 
Add choice between single or double frame modes

The code processes the Images without preprocessing, 
save the results and plots the first PIV field.

This Code is based on the DPIVSoft code 
'''

import os
import argparse
import numpy as np  # array and matrix computation  
import matplotlib.pyplot as plt
import cv2
import time
import yaml

# DPIVSoft libraries
import dpivsoft.DPIV as DPIV  # Python PIV implementation
import dpivsoft.Cl_DPIV as Cl_DPIV  # OpenCL PIV implementation
from dpivsoft.Classes import Parameters, GPU

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PIV images using DPIVSoft.")
    # parser.add_argument("input_folder", type=str, help="Path to the folder containing input images.")
    # parser.add_argument("output_folder", type=str, help="Path to the folder to save results.")
    parser.add_argument("fileName", type=str, help="PIV parameter file.")
    # parser.add_argument("--extension", type=str, default=".jpg", help="File extension of the images (default: .jpg).")
    args = parser.parse_args()

# with open(args.fileName) as f:
#             param = yaml.load(f, Loader=yaml.FullLoader)
#             print("starting")

def readParameters(fileName):
        """
        Read parameters from a yaml file
        """
        with open(fileName) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            print("starting")

            #try:
            #Default first step parameters
            Parameters.box_size_1_x = data['box_size_1_x']
            Parameters.box_size_1_y = data['box_size_1_y']
            Parameters.no_boxes_1_x = data['no_boxes_1_x']
            Parameters.no_boxes_1_y = data['no_boxes_1_y']
            Parameters.window_1_x =   data['window_1_x']
            Parameters.window_1_y =   data['window_1_y']

            #Number of pass of first step
            Parameters.no_iter_1 = data['no_iter_1']
            Parameters.no_iter_2 = data['no_iter_2']

            #Direct calculation or FFT
            Parameters.direct_calc = data['direct_calc']

            #Default first step parameters
            Parameters.box_size_2_x = data['box_size_2_x']
            Parameters.box_size_2_y = data['box_size_2_y']
            Parameters.no_boxes_2_x = data['no_boxes_2_x']
            Parameters.no_boxes_2_y = data['no_boxes_2_y']
            Parameters.window_2_x =   data['window_2_x']
            Parameters.window_2_y =   data['window_2_y']

            #default general parameters

            try:
                Parameters.mask = data['mask']
            except:
                Parameters.mask = 0
            try:
                Parameters.stereo = data['stereo']
            except:
                Parameters.stereo = 0

            Parameters.peak_ratio = data['peak_ratio']
            Parameters.weighting = data['weighting']
            Parameters.gaussian_size = data['gaussian_size']
            Parameters.median_limit = data['median_limit']
            Parameters.calibration = data['calibration']
            Parameters.delta_t = data['delta_t']

            #Extra data
            if Parameters.mask:
                if data['path_mask'].endswith('.np'):
                    Parameters.Data.mask = bool(np.load(data['path_mask']))
                else:
                    Parameters.Data.mask = np.asarray(cv2.cvtColor(cv2.imread(
                        data['path_mask']), cv2.COLOR_BGR2GRAY)).astype(bool)

            if Parameters.stereo:
                Parameters.stereo_calibration = Parameters.Stereo_Calibration(
                        data['path_stereo'])
        return data        

param=readParameters(args.fileName)

# =========================================================================
# WORKING FOLDERS
# =========================================================================
# dirImg = args.input_folder  # Images folder
dirImg = param['input_folder']
# dirRes = args.output_folder  # Results folder
dirRes = param['output_folder']
extension = param['extension'] if param['extension'].startswith(".") else f".{param['extension']}"   
overlap_x = param['overlap_x']/100
overlap_y = param['overlap_y']/100
processAll= param['processAll']
startImage= param['startImage']
endImage= param['endImage']
processEvery= param['processEvery']


if not os.path.exists(dirImg):
    print(f"Error: Input folder '{dirImg}' does not exist.")
    exit(1)
if not os.path.exists(dirRes):
    os.makedirs(dirRes)

# =========================================================================
# LIST OF IMAGES TO PROCESS
# =========================================================================
files = sorted([f for f in os.listdir(dirImg) if f.endswith(extension)])
if len(files) < 2:
    print("Error: Not enough images to process. At least two images are required.")
    exit(1)
print(f"Found {len(files)} images to process: {files}")



# # =========================================================================
# # SET PIV PARAMETERS
# # =========================================================================
# Parameters.readParameters(os.path.join(dirImg, parameterFile))

# =========================================================================
# OPENCL PROCESSING
# =========================================================================
thr = Cl_DPIV.select_Platform("selection")
os.environ["PYOPENCL_COMPILER_OUTPUT"] = "1"

# Load first pair of images to start the computation and set arrays sizes
name_img_1 = dirImg+files[0]
name_img_2 = dirImg+files[1]
Img1, Img2 = DPIV.load_images(name_img_1, name_img_2)
[height, width] = Img1.shape
print("Image size: ", width, "x", height)
Parameters.no_boxes_1_x = int(np.floor((width/Parameters.box_size_1_x) / overlap_x) + 1)
Parameters.no_boxes_1_y = int(np.floor((height/Parameters.box_size_1_y) / overlap_y) + 1)
Parameters.no_boxes_2_x = int(np.floor((width/Parameters.box_size_2_x) / overlap_x) + 1)
Parameters.no_boxes_2_y = int(np.floor((height/Parameters.box_size_2_y) / overlap_y) +1)
print("Number of interrogation windows 1st step: x ", Parameters.no_boxes_1_x, ",y ", Parameters.no_boxes_1_y)
print("Number of interrogation windows 2nd step: x ", Parameters.no_boxes_2_x, ",y ", Parameters.no_boxes_2_y)

GPU.img1 = thr.to_device(Img1)
GPU.img2 = thr.to_device(Img2)

Cl_DPIV.compile_Kernels(thr)
Cl_DPIV.initialization(width, height, thr)

start = time.time()
if processAll:
    startImage=0
    endImage=len(files)
    

for i in range(startImage,endImage,processEvery):#len(files),2):
    # Change the name of next iteration images only if needed
    if i < len(files) - 3:
        name_img_1 = os.path.join(dirImg, files[i + 2])
        name_img_2 = os.path.join(dirImg, files[i + 3])

    # Process images
    Cl_DPIV.processing(name_img_1, name_img_2, thr)

    # Get final results from GPU
    x2 = GPU.x2.get()
    y2 = GPU.y2.get()
    u2 = GPU.u2_f.get()
    v2 = GPU.v2_f.get()

    #flip images in y direction to match images orientation
    x=x2
    u=u2
    y = y2.max()-y2
    v = -v2

    # Save results in numpy file compatible with DPIVSoft format
    saveName = os.path.join(dirRes, f"gpu_field_{i:05d}")
    DPIV.save(x, y, u, v, saveName)

print("OpenCL algorithm finished. Time =", time.time() - start, "s")

# # =========================================================================
# # WORK WITH RESULTS
# # =========================================================================
# # Load PIV results
# Data = np.load(os.path.join(dirRes, "gpu_field_000.npz"))
# x = Data["x"]
# y = Data["y"]
# u = Data["u"]
# v = Data["v"]

# fig, ax1 = plt.subplots()
# ax1.quiver(x, y, u, v, scale=1 / 0.003)
# ax1.set_xlabel("x (pixels)", fontsize=18)
# ax1.set_ylabel("y (pixels)", fontsize=18)
# plt.show()