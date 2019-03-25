#pragma once
//#include "cvaux.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"

#define CV_MAX_NUM_GREY_LEVELS_8U 256
class cl_Texture
{
public:
	struct GLCM
	{
		int matrixSideLength;
		int numMatrices;
		double*** matrices;

		int numLookupTableElements;
		int forwardLookupTable[CV_MAX_NUM_GREY_LEVELS_8U];
		int reverseLookupTable[CV_MAX_NUM_GREY_LEVELS_8U];

		double** descriptors;
		int numDescriptors;
		int descriptorOptimizationType;
		int optimizationType;
	};
	cl_Texture(void);
	~cl_Texture(void);
	/* srcStepDirections should be static array..or if not the user should handle de-allocation */

	GLCM* CreateGLCM( const IplImage* srcImage, int stepMagnitude, const int* srcStepDirections, int numStepDirections, int optimizationType );
	void CreateGLCMDescriptors( GLCM* destGLCM, int descriptorOptimizationType);
	double GetGLCMDescriptor( GLCM* GLCM, int step, int descriptor );
	void GetGLCMDescriptorStatistics( GLCM* GLCM, int descriptor, double*_average, double* _standardDeviation );
	IplImage* CreateGLCMImage( GLCM* GLCM, int step );
	void CreateGLCM_LookupTable_8u_C1R( const uchar* srcImageData, int srcImageStep, CvSize srcImageSize, GLCM* destGLCM, int* steps, int numSteps, int* memorySteps );
	void CreateGLCMDescriptors_AllowDoubleNest( GLCM* destGLCM, int matrixIndex );
	void ReleaseGLCM( GLCM** GLCM, int flag );

};
#pragma once









//The main file:
//
//#include "highgui.h"
//
//#include "cvGLCM.h"
//#include <iostream>
//
//using namespace std;
//
//void main()
//{
//IplImage *img = cvLoadImage("test1.jpg",0);
//IplImage *img1 = cvLoadImage("test2.jpg", 0);
//IplImage *img2 = cvLoadImage("test.jpg", 0);
///*cvSetImageROI(img,cvRect(100,100,100,100));
//cvSetImageROI(img1,cvRect(100,100,100,100));
//cvSetImageROI(img2,cvRect(100,100,100,100));*/
//
//Cv_GLCM *glcm = cv_CreateGLCM(img, 1, NULL, 4, CV_GLCM_OPTIMIZATION_LUT);
//Cv_GLCM *glcm1 = cv_CreateGLCM(img1, 1, NULL, 4, CV_GLCM_OPTIMIZATION_LUT);
//Cv_GLCM *glcm2 = cv_CreateGLCM(img2, 1, NULL, 4, CV_GLCM_OPTIMIZATION_LUT);
//
//// Cv_GLCM *glcm2 = cv_CreateGLCM(img, 1, NULL, 4, CV_GLCM_OPTIMIZATION_HISTOGRAM);
//cv_CreateGLCMDescriptors(glcm, CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST);
//cv_CreateGLCMDescriptors(glcm1, CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST);
//cv_CreateGLCMDescriptors(glcm2, CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST);
//
//double d0 = cv_GetGLCMDescriptor(glcm, 0, CV_GLCMDESC_ENTROPY);
//double d00 = cv_GetGLCMDescriptor(glcm1, 0, CV_GLCMDESC_ENTROPY);
//double d000 = cv_GetGLCMDescriptor(glcm2, 0, CV_GLCMDESC_ENTROPY);
//cout<<"Entropy: "<<d0<<" "<<d00<<" "<<d000<<endl;
//
//double d1 = cv_GetGLCMDescriptor(glcm, 0, CV_GLCMDESC_ENERGY);
//double d11 = cv_GetGLCMDescriptor(glcm1, 0, CV_GLCMDESC_ENERGY);
//double d111 = cv_GetGLCMDescriptor(glcm2, 0, CV_GLCMDESC_ENERGY);
//cout<<"Energy: "<<d1<<" "<<d11<<" "<<d111<<endl;
//
//double d2 = cv_GetGLCMDescriptor(glcm, 0, CV_GLCMDESC_CONTRAST);
//double d22 = cv_GetGLCMDescriptor(glcm1, 0, CV_GLCMDESC_CONTRAST);
//double d222 = cv_GetGLCMDescriptor(glcm2, 0, CV_GLCMDESC_CONTRAST);
//cout<<"Contrast: "<<d2<<" "<<d22<<" "<<d222<<endl;
//
//double d3 = cv_GetGLCMDescriptor(glcm, 0, CV_GLCMDESC_CORRELATION );
//double d33 = cv_GetGLCMDescriptor(glcm1, 0, CV_GLCMDESC_CORRELATION );
//double d333 = cv_GetGLCMDescriptor(glcm2, 0, CV_GLCMDESC_CORRELATION );
//cout<<"Correlation: "<<d3<<" "<<d33<<" "<<d333<<endl;
//
//double d4 = cv_GetGLCMDescriptor(glcm, 0, CV_GLCMDESC_HOMOGENITY );
//double d44 = cv_GetGLCMDescriptor(glcm1, 0, CV_GLCMDESC_HOMOGENITY );
//double d444 = cv_GetGLCMDescriptor(glcm2, 0, CV_GLCMDESC_HOMOGENITY );
//cout<<"Homogenity: "<<d4<<" "<<d44<<" "<<d444<<endl;
//
//double a = 1; double *ave = &a;
//double s = 1; double *sd = &s;
//cv_GetGLCMDescriptorStatistics(glcm, CV_GLCMDESC_ENERGY, ave, sd);
//double a1 = 1; double *ave1 = &a1;
//double s1 = 1; double *sd1 = &s1;
//cv_GetGLCMDescriptorStatistics(glcm1, CV_GLCMDESC_ENERGY, ave1, sd1);
//double a2 = 1; double *ave2 = &a2;
//double s2 = 1; double *sd2 = &s2;
//cv_GetGLCMDescriptorStatistics(glcm2, CV_GLCMDESC_ENERGY, ave2, sd2);
//cout<<"Energy ave: "<<*ave<<" "<<*ave1<<" "<<*ave2<<endl;
//cout<<"Energy sd : "<<*sd<<" " <<*sd1<<" " <<*sd2<<endl;
//
//
//cvNamedWindow("test1");
//cvShowImage("test1",img);
//cvNamedWindow("test2");
//cvShowImage("test2",img1);
//cvNamedWindow("test3");
//cvShowImage("test3",img2);
//cvWaitKey(0);
//}



//#define CV_GLCM_OPTIMIZATION_NONE -2 //optimizationType
//#define CV_GLCM_OPTIMIZATION_LUT -1
//#define CV_GLCM_OPTIMIZATION_HISTOGRAM 0
//
//#define CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST 10
//descriptorOptimizationType
//#define CV_GLCMDESC_OPTIMIZATION_ALLOWTRIPLENEST 11
//#define CV_GLCMDESC_OPTIMIZATION_HISTOGRAM 4
//
//#define CV_GLCMDESC_ENTROPY 0 //descriptor
//#define CV_GLCMDESC_ENERGY 1
//#define CV_GLCMDESC_HOMOGENITY 2
//#define CV_GLCMDESC_CONTRAST 3
//#define CV_GLCMDESC_CLUSTERTENDENCY 4
//#define CV_GLCMDESC_CLUSTERSHADE 5
//#define CV_GLCMDESC_CORRELATION 6
//#define CV_GLCMDESC_CORRELATIONINFO1 7
//#define CV_GLCMDESC_CORRELATIONINFO2 8
//#define CV_GLCMDESC_MAXIMUMPROBABILITY 9
//
//#define CV_GLCM_ALL 0 //release flag
//#define CV_GLCM_GLCM 1
//#define CV_GLCM_DESC 2