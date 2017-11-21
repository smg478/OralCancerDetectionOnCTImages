## Oral Cancer Detection on CT images

This repository contains C++ implementation of oral cancer detection method described in the paper ["Computer Aided Detection of Oral Lesions on CT Images"](https://arxiv.org/abs/1611.09769)

In this package, there are two ways to test the algorithm developed for CT images: 

A. By running the stand-alone executable file. 

B. By running the code in visual studio environment. 

### A. Testing by running the stand-alone executable file:

In the package, “executable” folder contains a setup file. Run the setup file and a shortcut named “CAD-CT” will be created in the desktop. From that shortcut folder, run the CT_CAD.exe file. A folder browsing window will pop up and you can select a folder containing CT images from there. The program will run and will show detected images (if any). To exit from the program, close the console window. 
If you want, you can uninstall the software anytime from “add/remove program” section in control panel. 

### B. Testing the code in Visual Studio environment

The project was tested using Visual Studio 2010 (Service Pack 1). Therefore, it may not run properly in other versions of visual studio. 

“OpenCV” was used as a third party library for this project. Only free for commercial use modules were used. 

## Supported Image Specification:

Input image: Reconstructed oral CT scan (.raw) 

Image dimension: 800x800x(400~500: around 400-500 slices for a single patient) 

16-bit signed integer (raw)

One folder should contain all image slices of a patient.

## Environment

Visual Studio 2010

OpenCV 2.4.9


## Usage

Please see Readme.pdf

## Citation

Computer aided detection of oral lesions on CT images

S. Galib F. Islam M. Abir H.K. Lee

Journal of Instrumentation vol. 10 issue 12 (2015) pp: C12030-C12030
