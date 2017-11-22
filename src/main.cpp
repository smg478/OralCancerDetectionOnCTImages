//****************************************************************************************************************
//Algorithm for Automatic detection of lesion from CT images.
//This software detects lesion from both mandible and maxilla and consists of three algorithms.
//Algorithm 1: Closed boundary lesion detection from mandible 
//Algorithm 2: Bone deformation detection from mandible
//Algorithm 3: Bone deformation detection from maxilla
//
//Shaikat Galib
//Nuclear Engineering Dept.
//Missouri S&T
//Date:12.17.2014
//****************************************************************************************************************

//openCV Headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/ml/ml.hpp>

//c++ standard Headers
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <iomanip>
#include <fstream>
#include <windows.h>
#include <tchar.h>
#include <strsafe.h>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <functional>
#include "shlobj.h"

//Header for finding start slice
#include <persistence1d.hpp>

//Header for calculating GLCM texture
#include <cl_Texture.h>

using namespace std;
using namespace cv;
using namespace p1d;
using namespace std::placeholders;

#define ATTRIBUTES 15				//Number of feature Vectors
#define CLASSES 2                   //Number of Classes (Lesion/Normal).

//-------------------------------------------------------------------------------------------------------------
//Truth parameters
//-------------------------------------------------------------------------------------------------------------

double max_mean = 110;
double min_mean = 45;
double max_stddev = 24.0;
double min_stddev = 5.0;
double max_skew = 3.0;
double min_skew = -0.7;
double max_kurt = 14.0;
double min_kurt = 2.0;

//---------------------------------------------------------------------------------------------------------------
// Function Declaration
//---------------------------------------------------------------------------------------------------------------

// Get folder GUI

bool GetFolder(std::string& folderpath, const char* szCaption = NULL, HWND hOwner = NULL)
{
	bool retVal = false;

	// The BROWSEINFO struct tells the shell
	// how it should display the dialog.
	BROWSEINFO bi;
	memset(&bi, 0, sizeof(bi));

	bi.ulFlags = BIF_USENEWUI;
	bi.hwndOwner = hOwner;
	bi.lpszTitle = szCaption;

	// if using BIF_USENEWUI
	::OleInitialize(NULL);

	// Show the dialog and get the itemIDList for the selected folder.
	LPITEMIDLIST pIDL = ::SHBrowseForFolder(&bi);

	if (pIDL != NULL)
	{
		// Create a buffer to store the path, then get the path.
		char buffer[_MAX_PATH] = { '\0' };
		if (::SHGetPathFromIDList(pIDL, buffer) != 0)
		{
			// Set the string value.
			folderpath = buffer;
			retVal = true;
		}

		// free the item id list
		CoTaskMemFree(pIDL);
	}

	::OleUninitialize();

	return retVal;
}

// Get files in directory
void GetFilesInDirectory(std::vector<string> &out, const string &directory);

// Calculate Kurtosis
float calculateKurt(Mat image_400);

// Calculate Threshold Value
int maxEntropy(Mat image_400);

// Calculate Skewness and Kurtosis
struct SkewKurt
{
	float skewness_val;
	float kurtosis_val;
};
SkewKurt calculateSkewAndKurt(Mat image_400, Mat mask, double mean_val, double stddev_val);

// Class declaration
//---------------------------------------------------------------------------------------------------------------
// For calculating and comparing 3D coordinates
class Points {
	double v, w, x, y, z;
public:
	double V() const { return v; }
	double W() const { return w; }
	double X() const { return x; }
	double Y() const { return y; }
	double Z() const { return z; }

	Points(double v = 0.0, double w = 0.0, double x = 0.0, double y = 0.0, double z = 0.0) : v(v), w(w), x(x), y(y), z(z) {}
};

namespace std {
	ostream &operator<<(ostream &os, Points const &p) {
		return os << "(" << p.V() << ","<< p.W() << ", " << p.X() << ", " << p.Y() << ", " << p.Z() << ")";
	}
}
struct byZ {
	bool operator()(Points const &a, Points const &b) {
		return a.Z() < b.Z();
	}
};

// Find Mandible Bone Deformation

struct BDdata
{
	int slice_num;
	float x_val;
	float y_val;
	float Area_val;
};
BDdata findBoneDeform(Mat image_400, Mat fuse_image, Mat image_binary, int m);

// Find Maxilla bone deformation
BDdata findBoneDeform_maxl( Mat image_binary, int m);

//****************************************************************************************************************
// Main Function
//****************************************************************************************************************

int _tmain(int argc, _TCHAR* argv[])
{
	// Select CT folder (Dialogue Window for file browsing)
	std::string szPath("");			//szPath: selected folder path

	if (GetFolder(szPath, "Select CT image folder.") == true)
	{
		printf("Selected folder: \"%s\".\n", szPath.c_str());
	}

	else
	{
		printf("No folder selected!\n");
		system("pause");
		return -1;
	}

	//---------------------------------------------------------------------------------------------------------------
	// Time Calculation: Start clock
	clock_t t1, t2, t3;
	t1 = clock();

	//---------------------------------------------------------------------------------------------------------------
	// Get file names from directory
	//---------------------------------------------------------------------------------------------------------------
	cout << "Getting File names from folder...." << endl;

	vector<string> file_name;			// Stores all file names in folder
	vector<const string>full_list;		// Stores full path of every image

	const string directory = szPath;

	GetFilesInDirectory(file_name, directory);

	if (file_name.size() == 0)
	{
		cout << "No files(.slice) found" << endl;
		cout << "Program will exit now" << endl;
		system("pause");
		return -1;
	}

	cout << "Total images found:" << file_name.size() << endl << endl;

	for (size_t i = 0; i < file_name.size(); i++)
	{
		const string full_list_i = file_name[i];
		full_list.push_back(full_list_i);
	}

	//---------------------------------------------------------------------------------------------------------------
	//Read raw image and calculate kurtosis
	//---------------------------------------------------------------------------------------------------------------
	cout << "Analyzing Image Pattern. Please wait for a few seconds..." << endl;

	vector<float> kurto_list;						// Stores kurtosis value of every image

	for (int i = 200; i < 400; i++)
	{
		const char *imgPath = full_list[i].c_str();  //string to const char conversion
		FILE * pFile;
		long lSize;
		char * buffer;
		size_t result;

		pFile = fopen(imgPath, "rb");
		if (pFile == NULL) { fputs("File error", stderr); exit(1); }

		// obtain file size:
		fseek(pFile, 0, SEEK_END);
		lSize = ftell(pFile);

		if (lSize != 1280000)
		{
			cout << "Image size is not 800 x 800" << endl;
			system("pause");
			return -1;
		}

		rewind(pFile);

		// allocate memory to contain the whole file:
		buffer = (char*)malloc(sizeof(char)*lSize);

		if (buffer == NULL) { fputs("Memory error", stderr); exit(2); }

		// copy the file into the buffer:
		result = fread(buffer, 1, lSize, pFile);
		if (result != lSize) { fputs("Reading error", stderr); exit(3); }

		// clean up
		fclose(pFile);

		Mat image;
		int16_t *imageMap = (int16_t*)buffer;
		image.create(800, 800, CV_16SC1);
		memcpy(image.data, imageMap, 800 * 800 * sizeof(int16_t));

		std::free((void*)buffer);

		// Preprocessing

		Mat image16s_norm, image8u, image_400, image8s;

		normalize(image, image16s_norm, -32768, 32767, NORM_MINMAX);

		image8s = image16s_norm / 255.0;
		image8u = image8s + 128;
		image8u.convertTo(image8u, CV_8U);

		resize(image8u, image_400, Size(), 0.5, 0.5, 1);

		// Calculate Kurtosis

		float kurto_this_image = calculateKurt(image_400);

		kurto_list.push_back(kurto_this_image);
	}

	//---------------------------------------------------------------------------------------------------------------
	// Find Start slice (using persistance1D Header file)
	//---------------------------------------------------------------------------------------------------------------

	vector<int>MaxIndexes;

	//Run persistence on data
	Persistence1D p;
	p.RunPersistence(kurto_list);

	//Get all extrema with a persistence larger than 2.
	vector< TPairedExtrema > Extrema;
	p.GetPairedExtrema(Extrema, 1.5);   

	if (Extrema.size() == 0)
	{
		p.GetPairedExtrema(Extrema, 0.4f);
	}

	//pairs are sorted ascending wrt. persistence.
	for (vector< TPairedExtrema >::iterator it = Extrema.begin(); it != Extrema.end(); it++)
	{
		MaxIndexes.push_back((*it).MaxIndex);
	}

	int initial_slice = *max_element(MaxIndexes.begin(), MaxIndexes.end());

	initial_slice += 200;
	int final_slice = initial_slice + 130;
	if ((size_t)final_slice > file_name.size())
		final_slice = file_name.size();

	int intermediate_slice = initial_slice + 40;

	int maxilla_start = initial_slice - 190;
	int maxilla_end = maxilla_start + 140;

	// Time for finding start slice
	t3 = clock();
	float diff((float)t3 - (float)t1);
	float seconds = diff / CLOCKS_PER_SEC;
	cout << "Time for image pattern analysis(seconds):" << seconds << endl;

	cout << "Scanning from Slice # " << maxilla_start << endl;

	//---------------------------------------------------------------------------------------------------------------
	//Scan every image
	//---------------------------------------------------------------------------------------------------------------
	cout << "Scanning...." << endl;

	//-------------------------------------------------------------------------------------------------------------
	//Load the trained Neural Network parameters
	//read the model from the XML file and create the neural network.
	CvANN_MLP nnetwork;
	CvFileStorage* storage = cvOpenFileStorage("paramAll15.xml", 0, CV_STORAGE_READ);
	CvFileNode *n = cvGetFileNodeByName(storage, 0, "DigitOCR");
	nnetwork.read(storage, n);
	cvReleaseFileStorage(&storage);

	//-------------------------------------------------------------------------------------------------------------
	// Read raw image
	vector< vector<double> > vec;					// Stores slice #, centroid and eucledian distance of detected slices
	vector<Points> detected_slice_data;				// Stores slice #, centroid and eucledian distance of detected slices
	vector<Points> BD_detected_slice_data;
	vector<Points> BDM_detected_slice_data;
	
	int slice_num = 0;


	//******************************************************************************************************************************************************************
	//main for loop starts
	//******************************************************************************************************************************************************************


	for (int m = maxilla_start; m < final_slice; m++)			//Loop through every slice for lesion detection
	{
		
	//-------------------------------------------------------------------------------------------------------------	
	//if: maxilla scan starts
	
		if(m >= maxilla_start && m < maxilla_end )
	{
		const char *imgPath = full_list[m].c_str();  //string to const char conversion
		FILE * pFile;
		long lSize;
		char * buffer;
		size_t result;

		pFile = fopen(imgPath, "rb");
		if (pFile == NULL) { fputs("File error", stderr); exit(1); }

		// obtain file size:
		fseek(pFile, 0, SEEK_END);
		lSize = ftell(pFile);
		rewind(pFile);

		// allocate memory to contain the whole file:
		buffer = (char*)malloc(sizeof(char)*lSize);

		if (buffer == NULL) { fputs("Memory error", stderr); exit(2); }

		// copy the file into the buffer:
		result = fread(buffer, 1, lSize, pFile);
		if (result != lSize) { fputs("Reading error", stderr); exit(3); }

		// clean up
		fclose(pFile);

		Mat image;						// Store raw data (16-bit signed)
		int16_t *imageMap = (int16_t*)buffer;
		image.create(800, 800, CV_16SC1);
		memcpy(image.data, imageMap, 800 * 800 * sizeof(int16_t));

		std::free((void*)buffer);

		//---------------------------------------------------------------------------------------------------------------
		// Preprocessing

		Mat image16s_norm, image8u, image_400, image8s;

		normalize(image, image16s_norm, -32768, 32767, NORM_MINMAX);		// Normalize raw data between max and min value of 16-bit signed range

		image8s = image16s_norm / 255.0;				//pixel value manipulation: 16-bit signed to 8-bit signed
		image8u = image8s + 128;						//pixel value manipulation: 8-bit signed to 8-bit unsigned
		image8u.convertTo(image8u, CV_8U);				//Data structure conversion: 16-bit signed to 8-bit unsigned

		resize(image8u, image_400, Size(), 0.5, 0.5, 1);	// Resize image to 400x400 pixels

		//---------------------------------------------------------------------------------------------------------------
		// Calculate binary image (image_binary)

		int threshold_val = maxEntropy(image_400);
		//threshold_val = threshold_val + 10;

		if (m < maxilla_start + 100)
		{
			threshold_val = threshold_val - 6;
		}

		Mat image_binary;

		threshold(image_400, image_binary, threshold_val, 255, THRESH_BINARY);

		
		//**************************************************************************************************************
		//Find Bone Deformation
		//---------------------------------------------------------------------------------------------------------------

		BDdata loc_data_maxl = findBoneDeform_maxl(image_binary, m);
		int slc_maxl = loc_data_maxl.slice_num;
		float x_val_maxl = loc_data_maxl.x_val;
		float y_val_maxl = loc_data_maxl.y_val;
		float Ar_val_maxl = loc_data_maxl.Area_val;

		if (slc_maxl != 0)
		{
			BDM_detected_slice_data.push_back(Points(slc_maxl, slc_maxl, x_val_maxl, y_val_maxl, Ar_val_maxl));
		}

		}   //Maxilla if ends
		
//-----------------------------------------------------------------------------------------------------------------------------------------------------		
		
	// elase if : mandible scan starts	
		
	else if ( m >= initial_slice && m < final_slice)
		{
		
		const char *imgPath = full_list[m].c_str();  //string to const char conversion
		FILE * pFile;
		long lSize;
		char * buffer;
		size_t result;

		pFile = fopen(imgPath, "rb");
		if (pFile == NULL) { fputs("File error", stderr); exit(1); }

		// obtain file size:
		fseek(pFile, 0, SEEK_END);
		lSize = ftell(pFile);
		rewind(pFile);

		// allocate memory to contain the whole file:
		buffer = (char*)malloc(sizeof(char)*lSize);

		if (buffer == NULL) { fputs("Memory error", stderr); exit(2); }

		// copy the file into the buffer:
		result = fread(buffer, 1, lSize, pFile);
		if (result != lSize) { fputs("Reading error", stderr); exit(3); }

		// clean up
		fclose(pFile);

		Mat image;						// Store raw data (16-bit signed)
		int16_t *imageMap = (int16_t*)buffer;
		image.create(800, 800, CV_16SC1);
		memcpy(image.data, imageMap, 800 * 800 * sizeof(int16_t));

		std::free((void*)buffer);

		//---------------------------------------------------------------------------------------------------------------
		// Preprocessing

		Mat image16s_norm, image8u, image_400, image8s;

		normalize(image, image16s_norm, -32768, 32767, NORM_MINMAX);		// Normalize raw data between max and min value of 16-bit signed range

		image8s = image16s_norm / 255.0;				//pixel value manipulation: 16-bit signed to 8-bit signed
		image8u = image8s + 128;						//pixel value manipulation: 8-bit signed to 8-bit unsigned
		image8u.convertTo(image8u, CV_8U);				//Data structure conversion: 16-bit signed to 8-bit unsigned

		resize(image8u, image_400, Size(), 0.5, 0.5, 1);	// Resize image to 400x400 pixels

		//---------------------------------------------------------------------------------------------------------------
		// Calculate binary image (image_binary)

		int threshold_val = maxEntropy(image_400);

		if (m < intermediate_slice)
		{
			threshold_val = threshold_val - 8;
		}

		Mat image_binary;

		threshold(image_400, image_binary, threshold_val, 255, THRESH_BINARY);

		//---------------------------------------------------------------------------------------------------------------
		// Edge Detection (edge_image)

		Mat edge_image, detected_edges;

		/// Reduce noise with a kernel 3x3
		blur(image_400, detected_edges, Size(3, 3));

		/// Canny detector
		Canny(detected_edges, detected_edges, 50, threshold_val + 30, 3);

		/// Using Canny's output as a mask, display result
		edge_image = Scalar::all(0);

		image_400.copyTo(edge_image, detected_edges);

		for (int i = 0; i < edge_image.rows; i++)
		{
			for (int j = 0; j < edge_image.cols; j++)
			{
				if ((int)edge_image.at<uchar>(i, j) == 0)
					edge_image.at<uchar>(i, j) = 0;
				else
					edge_image.at<uchar>(i, j) = 255;
			}
		}

		//---------------------------------------------------------------------------------------------------------------
		// Fuse Binary and Edge images (fuse_image)

		Mat fuse_image;
		bitwise_or(image_binary, edge_image, fuse_image);
		
		//---------------------------------------------------------------------------------------------------------------
		// Morphological operation (First closing and then opening )

		Mat image_fuse_close, image_fuse_open, complement_image;

		// Creating the disk-shaped structuring element
		Mat disk(7, 7, CV_8U, cv::Scalar(1));

		disk.at<uchar>(0, 0) = 0;
		disk.at<uchar>(0, 1) = 0;
		disk.at<uchar>(0, 2) = 0;
		disk.at<uchar>(0, 4) = 0;
		disk.at<uchar>(0, 5) = 0;
		disk.at<uchar>(0, 6) = 0;
		disk.at<uchar>(1, 0) = 0;
		disk.at<uchar>(2, 0) = 0;
		disk.at<uchar>(4, 0) = 0;
		disk.at<uchar>(5, 0) = 0;
		disk.at<uchar>(6, 0) = 0;
		disk.at<uchar>(6, 1) = 0;
		disk.at<uchar>(6, 2) = 0;
		disk.at<uchar>(6, 4) = 0;
		disk.at<uchar>(6, 5) = 0;
		disk.at<uchar>(6, 6) = 0;
		disk.at<uchar>(1, 6) = 0;
		disk.at<uchar>(2, 6) = 0;
		disk.at<uchar>(4, 6) = 0;
		disk.at<uchar>(5, 6) = 0;;

		Mat element_closing = disk;
		Mat element_opening = getStructuringElement(MORPH_RECT, Size(2, 2));

		morphologyEx(fuse_image, image_fuse_close, MORPH_CLOSE, element_closing);
		morphologyEx(image_fuse_close, image_fuse_open, MORPH_OPEN, element_opening);

		//**************************************************************************************************************
		//Find Bone Deformation
		//---------------------------------------------------------------------------------------------------------------
		if (m > intermediate_slice + 5)
		{
			BDdata loc_data = findBoneDeform(image_400, fuse_image, image_binary, m);
			int slc = loc_data.slice_num;
			float x_val = loc_data.x_val;
			float y_val = loc_data.y_val;
			float Ar_val = loc_data.Area_val;

			if (slc != 0)
			{
				BD_detected_slice_data.push_back(Points(slc, slc, x_val, y_val, Ar_val));
			}
		}

		//***************************************************************************************************************
		
		//---------------------------------------------------------------------------------------------------------------
		// Calculate complement image

		bitwise_not(image_fuse_open, complement_image);

		//---------------------------------------------------------------------------------------------------------------
		// Contour Detection
		// input : complement image

		vector<vector<Point> > contours;			// stores boundary coordinates of contours

		/// Find contours: create a copy of complement image (complement_copy)
		Mat complement_copy;
		complement_image.copyTo(complement_copy);

		findContours(complement_copy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);   // we can get area of each contour from 'contourArea'

		//---------------------------------------------------------------------------------------------------------------
		//Calculate Area and Centriod
		// Calculate Centroid (by finding moments of each contour)

		vector<Moments> contourMoments(contours.size());			// stores moments of each contour
		for (size_t i = 0; i < contours.size(); i++)
		{
			contourMoments[i] = moments(contours[i], false);
		}

		vector<Point2f> ContCenter(contours.size());				// stores centroid of each contour
		vector<double> cAr;

		for (size_t i = 0; i < contours.size(); i++)
		{
			ContCenter[i] = Point2f((float)contourMoments[i].m10 / (float)contourMoments[i].m00, (float)contourMoments[i].m01 / (float)contourMoments[i].m00);
			cAr.push_back(contourArea(contours[i]));
		}

		//---------------------------------------------------------------------------------------------------------------
		// Sort blobs by Area and Centroid
		// Now we have Area and Centroid of each contour. We will sort out contours from these two parameters

		vector<int> ContourIndex;			// Stores sorted contour #
		// Stores area of sorted contours

		for (size_t i = 0; i < contours.size(); i++)
		{
			if (contourArea(contours[i]) > 120.0		&&		contourArea(contours[i]) < 800.0		&&		ContCenter[i].y < 250)
			{
				ContourIndex.push_back(i);			// ContourIndex contains sorted contour #
				//cAr.push_back(contourArea(contours[i]));
			}
		}

		//---------------------------------------------------------------------------------------------------------------
		// Feature Calculation
		//---------------------------------------------------------------------------------------------------------------
		// Sequential masking
		// Find Mean and Standard deviation of sorted contours

		Scalar  mean;
		Scalar  stddev;

		for (size_t i = 0; i < ContourIndex.size(); i++)			// Go for each sorted contour
		{
			Mat mask = Mat::zeros(image_400.rows, image_400.cols, CV_8UC1);

			drawContours(mask, contours, ContourIndex[i], Scalar(255), CV_FILLED); // i= contour index #

			Mat masked_image_400 = Mat::zeros(image_400.rows, image_400.cols, CV_8UC1);

			image_400.copyTo(masked_image_400, mask);  // Mask is original image masked by a contour

			meanStdDev(image_400, mean, stddev, mask);

			double mean_val = mean[0];			// scaler contains 4 values. Taking first value only
			double stddev_val = stddev[0];

			SkewKurt result = calculateSkewAndKurt(image_400, mask, mean_val, stddev_val);

			double kurtosis_value = result.kurtosis_val;
			double skewness_value = result.skewness_val;

			//-------------------------------------------------------------------------------------
			// Calculate GLCM features and Hu moments of suspected area

			if (mean_val < max_mean			&&		mean_val > min_mean			&&
				stddev_val < max_stddev     &&      stddev_val > min_stddev     &&
				skewness_value < max_skew	&&		skewness_value > min_skew	&&
				kurtosis_value < max_kurt	&&		kurtosis_value > min_kurt)		// Sort contours by mean, standard deviation, skewness and kurtosis values
			{
				Rect boundRect;
				boundRect = boundingRect(Mat(contours[ContourIndex[i]]));
				boundRect.x = boundRect.x - 5;
				boundRect.y = boundRect.y - 5;
				boundRect.height = boundRect.height + 10;
				boundRect.width = boundRect.width + 10;

				Mat cropedImage = image_400(boundRect);			// Crop the suspected area

				Moments croppedMoments;								// Stores moments of suspected area
				croppedMoments = moments(cropedImage, false);

				// Calculate Hu moments
				double hu[7];
				HuMoments(croppedMoments, hu);						// Computing Hu moments of suspected area

				IplImage* image_masked_ipl = cvCloneImage(&(IplImage)cropedImage);

				// GLCM parameters
				int count_steps = 4;
				const int StepDirections[] = { 0, 1, -1, 1, -1, 0, -1, -1 };
				cl_Texture* texture = new cl_Texture();
				cl_Texture::GLCM* glcm;

				double d0, d1, d2, d3;
				vector <double> d0s, d1s, d2s, d3s;
				double * features = new double[4 * count_steps];

				glcm = texture->CreateGLCM(image_masked_ipl, 2, StepDirections, count_steps, CV_GLCM_OPTIMIZATION_NONE);	// build the GLCM
				texture->CreateGLCMDescriptors(glcm, CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST);		// get the features from GLCM

				for (int k = 0; k < count_steps; k++)
				{
					d0 = texture->GetGLCMDescriptor(glcm, k, CV_GLCMDESC_ENTROPY);
					d0s.push_back(d0);
					features[k*count_steps] = d0;
					d1 = texture->GetGLCMDescriptor(glcm, k, CV_GLCMDESC_ENERGY);
					d1s.push_back(d1);
					features[k*count_steps + 1] = d1;
					d2 = texture->GetGLCMDescriptor(glcm, k, CV_GLCMDESC_HOMOGENITY);
					d2s.push_back(d2);
					features[k*count_steps + 2] = d2;
					d3 = texture->GetGLCMDescriptor(glcm, k, CV_GLCMDESC_CONTRAST);
					d3s.push_back(d3);
					features[k*count_steps + 3] = d3;
				}

				double sumd0 = std::accumulate(d0s.begin(), d0s.end(), 0.0);
				double ent = sumd0 / d0s.size();
				double sumd1 = std::accumulate(d1s.begin(), d1s.end(), 0.0);
				double enr = sumd1 / d1s.size();
				double sumd2 = std::accumulate(d2s.begin(), d2s.end(), 0.0);
				double hom = sumd2 / d2s.size();
				double sumd3 = std::accumulate(d3s.begin(), d3s.end(), 0.0);
				double con = sumd3 / d3s.size();

				cvReleaseImage(&image_masked_ipl);

				delete[] features;

				//---------------------------------------------------------------------------------------------------------------
				// Data preparation for Neural Network input

				cv::Mat data(1, ATTRIBUTES, CV_32F);		// Container for NN input

				data.at<float>(0, 0) = (float)mean_val;
				data.at<float>(0, 1) = (float)stddev_val;
				data.at<float>(0, 2) = (float)skewness_value;
				data.at<float>(0, 3) = (float)kurtosis_value;
				data.at<float>(0, 4) = (float)con;
				data.at<float>(0, 5) = (float)hom;
				data.at<float>(0, 6) = (float)enr;
				data.at<float>(0, 7) = (float)ent;
				data.at<float>(0, 8) = (float)hu[0];
				data.at<float>(0, 9) = (float)hu[1];
				data.at<float>(0, 10) = (float)hu[2];
				data.at<float>(0, 11) = (float)hu[3];
				data.at<float>(0, 12) = (float)hu[4];
				data.at<float>(0, 13) = (float)hu[5];
				data.at<float>(0, 14) = (float)hu[6];

				//---------------------------------------------------------------------------------------------------------------
				// Classification
				//---------------------------------------------------------------------------------------------------------------

				int maxIndex = 0;
				cv::Mat classOut(1, CLASSES, CV_32F);

				//prediction
				nnetwork.predict(data, classOut);
				float value;
				float maxValue = classOut.at<float>(0, 0);
				for (int index = 1; index < CLASSES; index++)
				{
					value = classOut.at<float>(0, index);
					if (value > maxValue)
					{
						maxValue = value;
						maxIndex = index;
					}
				}
				//maxIndex is the predicted class.

				if (maxIndex == 1)			// 1 : lesion
				{
					// Show features in console window
					//cout << setprecision(3)
					//<< m << " " <<
					//setw(8) << i << " " <<
					//setw(8) << mean_val << " " <<
					//setw(8) << stddev_val << " " <<
					//setw(8) << skewness_value << " " <<
					//setw(8) << kurtosis_value << " " <<
					//setw(8) << con << " " <<
					//setw(8) << hom << " " <<
					//setw(8) << enr << " " <<
					//setw(8) << ent << " " <<
					//setw(8) << hu[0] << " " <<
					//setw(8) << hu[1] << " " <<
					//setw(8) << hu[2] << " " <<
					//setw(8) << hu[3] << " " <<
					//setw(8) << hu[4] << " " <<
					//setw(8) << hu[5] << " " <<
					//setw(8) << hu[6] << " " <<
					//setw(8) << ContCenter[ContourIndex[i]].x << " " <<
					//setw(8) << ContCenter[ContourIndex[i]].y << " " <<
					//endl;

					//Save slice number and centroid in a vector
					vector<double> row;

					row.push_back(m);
					row.push_back(ContCenter[ContourIndex[i]].x);
					row.push_back(ContCenter[ContourIndex[i]].y);
					double euclid_dist = sqrt(pow(ContCenter[ContourIndex[i]].x, 2) + pow(ContCenter[ContourIndex[i]].y, 2)); // Euclid distance represents relative distance of lesion coordinate from (0,0) position
					row.push_back(euclid_dist);

					vec.push_back(row);

					detected_slice_data.push_back(Points(m, cAr[ContourIndex[i]], ContCenter[ContourIndex[i]].x, ContCenter[ContourIndex[i]].y, euclid_dist));
				}
			}
		}
	 }    // else if (mandible) ends

	}  // main For loop ends




	//**********************************************************************************************************************************************************************************************************************	
	//Closed boundary image position calculation and display
	//-------------------------------------------------------------------------------------------------------------
	// Sort out those consecutive slices where same positions are detected
	//-------------------------------------------------------------------------------------------------------------
	//-------------------------------------------------------------------------------------
	
	// Sort detected_slice_data according to eucledean distance
	
	std::sort(detected_slice_data.begin(), detected_slice_data.end(), byZ());

	//indexing detected_slice_data according to euclidean distance

	vector <int> Idx;
	Idx.push_back(1);
	int index = 1;

	for (size_t i = 1; i < (detected_slice_data.size()); i++)
	{
		double diff = abs(detected_slice_data[i - 1].Z() - detected_slice_data[i].Z());
		if (diff < 8)
			Idx.push_back(index);
		else
		{
			index = index + 1;
			Idx.push_back(index);
		}
	}
	cout << endl;

	//--------------------------------------------------------------------------------------

	vector<int> freq_uv;
	freq_uv.push_back(0);
	int prev = Idx[0];
	//auto prev = Idx[0];        // ensure !uv.empty()

	for (size_t p = 0; p < Idx.size(); p++)
	{
		if (prev != Idx[p])
		{
			freq_uv.push_back(0);
			prev = Idx[p];
		}
		++freq_uv.back();
	}

	//---------------------------------------------------------------------------------------------------------------
	// Find high frequency indexes

	vector<int>high_freq_idx;
	for (size_t i = 0; i < freq_uv.size(); i++)
	{
		if (freq_uv[i] > 4)		//Detection in more than 4 slices
			high_freq_idx.push_back(i + 1);
	}

	//---------------------------------------------------------------------------------------------------------------
	// Store detected image data in a vector

	vector<Points> detected_slice_data_serial;	// Stores final detected slice data serially (Slice #, Centroid)

	for (size_t i = 0; i < (detected_slice_data.size()); i++)
	{
		for (size_t j = 0; j < high_freq_idx.size(); j++)
		{
			if (Idx[i] == high_freq_idx[j])
			{
				double detected_slice_number = detected_slice_data[i].V();
				double detected_slice_X = detected_slice_data[i].X();
				double detected_slice_Y = detected_slice_data[i].Y();
				double detected_slice_Area = detected_slice_data[i].W();

				detected_slice_data_serial.push_back(Points(detected_slice_number,detected_slice_Area, detected_slice_X, detected_slice_Y, detected_slice_number));
			}
		}
	}

	//---------------------------------------------------------------------------------------------------------------
	//---------------------------------------------------------------------------------------------------------------
	//Sort detected slices according to slice number
	//---------------------------------------------------------------------------------------------------------------
	std::sort(detected_slice_data_serial.begin(), detected_slice_data_serial.end(), byZ());

	//---------------------------------------------------------------------------------------------------------------
	//indexing detected_slice_data according to slice number

	vector <int> Idx_slice;
	Idx_slice.push_back(1);
	int index_slice = 1;

	for (size_t i = 1; i < (detected_slice_data_serial.size()); i++)
	{
		double diff = abs(detected_slice_data_serial[i - 1].Z() - detected_slice_data_serial[i].Z());
		if (diff < 5)
			Idx_slice.push_back(index_slice);
		else
		{
			index_slice = index_slice + 1;
			Idx_slice.push_back(index_slice);
		}
	}
	//---------------------------------------------------------------------------------------------------------------
	// Calculate frequency of Indexes

	vector<int> freq_slice;
	freq_slice.push_back(0);
	int prev_slice = Idx_slice[0];
	//auto prev_slice = Idx_slice[0];        // ensure !slice.empty()

	for (size_t q = 0; q < Idx_slice.size(); q++)
	{
		if (prev_slice != Idx_slice[q])
		{
			freq_slice.push_back(0);
			prev_slice = Idx_slice[q];
		}
		++freq_slice.back();
	}

	//---------------------------------------------------------------------------------------------------------------
	// Find high frequency numbers

	vector<int>high_freq_idx_slice;
	for (size_t i = 0; i < freq_slice.size(); i++)
	{
		if (freq_slice[i] > 5)		//Detection in more than 4 slices
			high_freq_idx_slice.push_back(i + 1);
	}

	vector<Points> detected_slice_data_serial_sorted;	// Stores final detected slice data serially (Slice #, Centroid)
	//---------------------------------------------------------------------------------------------------------------
	// Store detected image data in a vector
	for (size_t i = 0; i < (detected_slice_data_serial.size()); i++)
	{
		for (size_t j = 0; j < high_freq_idx_slice.size(); j++)
		{
			if (Idx_slice[i] == high_freq_idx_slice[j])
			{
				double detected_slice_number_sort	= detected_slice_data_serial[i].V();
				double detected_slice_X_sort		= detected_slice_data_serial[i].X();
				double detected_slice_Y_sort		= detected_slice_data_serial[i].Y();
				double detected_slice_Area_sort		= detected_slice_data_serial[i].W();

				detected_slice_data_serial_sorted.push_back(Points(detected_slice_number_sort,detected_slice_Area_sort, detected_slice_X_sort, detected_slice_Y_sort, Idx_slice[i]));
			}
		}
	}


	//-------------------------------------------------------------------------------------------------------------------------------------------------------
	//Calculate Approx. lesion centre and average co-ordinate
	if (detected_slice_data_serial_sorted.size() > 0)
	{
		std::sort(detected_slice_data_serial_sorted.begin(), detected_slice_data_serial_sorted.end(), byZ());

		int j=0;
		vector<double> total_x ;		total_x.push_back(0);
		vector<double> total_y ;		total_y.push_back(0);
		vector<double> total_slice ;	total_slice.push_back(0);
		vector<double> total_area ;		total_area.push_back(0);

		vector<double> av_x ;			av_x.push_back(0);
		vector<double> av_y ;			av_y.push_back(0);
		vector<double> av_slice ;		av_slice.push_back(0);
		vector<double> av_area ;		av_area.push_back(0);

		int k=1;

		for (size_t i = 1; i < (detected_slice_data_serial_sorted.size()); i++)
		{
			if (detected_slice_data_serial_sorted[i].Z() == detected_slice_data_serial_sorted[i-1].Z() )
			{

				total_x[j] += detected_slice_data_serial_sorted[i].X();
				av_x[j] = total_x[j] / k;
				total_y[j] += detected_slice_data_serial_sorted[i].Y();
				av_y[j] = total_y[j] / k;
				total_slice[j] += detected_slice_data_serial_sorted[i].V();
				av_slice[j] = total_slice[j] / k;
				total_area[j] += detected_slice_data_serial_sorted[i].W();
				av_area[j] = total_area[j] / k;
				k++;
			}
			else
			{
				j++;
				total_x.push_back(j);
				total_y.push_back(j);
				total_slice.push_back(j);
				total_area.push_back(j);

				av_x.push_back(j);
				av_y.push_back(j);
				av_slice.push_back(j);
				av_area.push_back(j);
				k=1;
			}
		}
		//--------------------------------------------------------------------------------------------------------------------------------
		// Closed Boundary Lesion image display

		char winName[20];

		if(av_slice.size()>0)
		{
			for (size_t i=0; i < av_slice.size(); i++)
			{
				int detected_slice_num_s = (int)av_slice[i];
				int detected_slice_x_s = (int)av_x[i];
				int detected_slice_y_s = (int)av_y[i];
				int detected_slice_Area_s = (int)av_area[i];
				double equiv_dia = 1.5 * 2 * sqrt( 4*detected_slice_Area_s/ 3.1416);  //1.5 is the correction coefficient
				double dia_mm = equiv_dia * 0.2;		// 1 pixel = 0.2 mm

				int slice_size = detected_slice_data_serial_sorted.size();
				int last_slice_detected = (int)detected_slice_data_serial_sorted[slice_size - 1].V();
				int first_slice_detected = (int)detected_slice_data_serial_sorted[0].V();
				int total_slice_detected = last_slice_detected - first_slice_detected + 1;

				int lesion_centre = abs((first_slice_detected + last_slice_detected)/2);

				cout << "Close Boundary Lesion" << endl;
				cout << setw(8) << "Lesion detected from slice:" << setw(4) << first_slice_detected << setw(4) << "to" << setw(4) << last_slice_detected << endl;

				cout << "Approx. Lesion Center:" <<  endl;
				cout << "x-coordinate:" << setw(4) << detected_slice_x_s*2 <<  endl;
				cout << "y-coordinate:" << setw(4) << detected_slice_y_s*2 <<  endl;
				cout << "z-coordinate (slice number):" << setw(4) << lesion_centre <<  endl << endl;
				cout << "Approx. Lesion size (Dia):" << setw(4) << dia_mm << setw(8) << "mm (1 pixel = 0.2 mm)" << endl << endl;
				//---------------------------------------------------------------------------------------------------------------------------------------------------
				// Read detected image from original location
				const char *imgPath = full_list[detected_slice_num_s].c_str();  //string to const char conversion
				FILE * pFile;
				long lSize;
				char * buffer;
				size_t result;

				pFile = fopen(imgPath, "rb");
				if (pFile == NULL) { fputs("File error", stderr); exit(1); }

				// obtain file size:
				fseek(pFile, 0, SEEK_END);
				lSize = ftell(pFile);
				rewind(pFile);

				// allocate memory to contain the whole file:
				buffer = (char*)malloc(sizeof(char)*lSize);

				if (buffer == NULL) { fputs("Memory error", stderr); exit(2); }

				// copy the file into the buffer:
				result = fread(buffer, 1, lSize, pFile);
				if (result != lSize) { fputs("Reading error", stderr); exit(3); }

				// clean up
				fclose(pFile);

				Mat image;
				int16_t *imageMap = (int16_t*)buffer;
				image.create(800, 800, CV_16SC1);
				memcpy(image.data, imageMap, 800 * 800 * sizeof(int16_t));

				std::free((void*)buffer);

				// Preprocessing
				Mat image16s_norm, image8u, image_400, image8s;

				normalize(image, image16s_norm, -32768, 32767, NORM_MINMAX);

				image8s = image16s_norm / 255.0;
				image8u = image8s + 128;
				image8u.convertTo(image8u, CV_8U);

				resize(image8u, image_400, Size(), 0.5, 0.5, 1);
				cvtColor(image_400, image_400, CV_GRAY2RGB);		// Converting grayscale to color to put red marker on image

				//Show detected image
				//Rect boundRect = Rect(detected_slice_x_s - 20, detected_slice_y_s - 20, 40, 40);

				//rectangle(image_400, boundRect, Scalar(0, 0, 255), +1, 8);
				sprintf(winName, "Slice no:%d", lesion_centre);

				std::ostringstream str;
				//str << "x:" << detected_slice_x_s*2 << ", y:" << detected_slice_y_s*2;
				//cv::putText(image_400, str.str(), cv::Point(detected_slice_x_s - 30, detected_slice_y_s - 30), FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(0, 255, 255));
				str << "x";
				cv::putText(image_400, str.str(), cv::Point(detected_slice_x_s, detected_slice_y_s), FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0, 0, 255),2);

				namedWindow(winName, WINDOW_AUTOSIZE);
				imshow(winName, image_400);
				cv::waitKey(60);
			}
		}
	}

	//****************************************************************************************************************************************************************
	//-------------------------------------------------------------------------------------------------------------------------------------------------------
	//Mandible Bone deformation image position calculation and display
	//-------------------------------------------------------------------------------------------------------------------------------------------------------

	//Calculate Approx. lesion centre and average co-ordinate

	vector<double> BD_total_x ;			BD_total_x.push_back(0);
	vector<double> BD_total_y ;			BD_total_y.push_back(0);
	vector<double> BD_total_slice ;		BD_total_slice.push_back(0);
	vector<double> BD_total_area ;		BD_total_area.push_back(0);

	vector<double> BD_av_x ;			BD_av_x.push_back(0);
	vector<double> BD_av_y ;			BD_av_y.push_back(0);
	vector<double> BD_av_slice ;		BD_av_slice.push_back(0);
	vector<double> BD_av_area ;			BD_av_area.push_back(0);

	int BD_k = 1;
	int BD_j = 0;

	if(BD_detected_slice_data.size() > 0)
	{
		for (size_t i = 0; i < (BD_detected_slice_data.size()); i++)
		{

			BD_total_x[BD_j]		+= BD_detected_slice_data[i].X();
			BD_av_x[BD_j]			= BD_total_x[BD_j] / BD_k;
			BD_total_y[BD_j]		+= BD_detected_slice_data[i].Y();
			BD_av_y[BD_j]			= BD_total_y[BD_j] / BD_k;
			BD_total_slice[BD_j]	+= BD_detected_slice_data[i].V();
			BD_av_slice[BD_j]		= BD_total_slice[BD_j] / BD_k;
			BD_total_area[BD_j]		+= BD_detected_slice_data[i].Z();
			BD_av_area[BD_j]		= BD_total_area[BD_j] / BD_k;
			BD_k++;
		}
	}
	
	// Image display

	char winName2[40];

	if(BD_detected_slice_data.size() >0 )
	{
		for (size_t i=0; i < BD_av_slice.size(); i++)
		{

			//for (size_t k =0; k < )
			int BD_detected_slice_num_s = (int)BD_av_slice[i];
			int BD_detected_slice_x_s = (int)BD_av_x[i];
			int BD_detected_slice_y_s = (int)BD_av_y[i];
			int BD_detected_slice_Area_s = (int)BD_av_area[i];
			double BD_equiv_dia = 1.5 * 2 * sqrt( 4*BD_detected_slice_Area_s/ 3.1416);  //1.5 is the correction coefficient
			double BD_dia_mm = BD_equiv_dia * 0.2;		// 1 pixel = 0.2 mm

			int BD_slice_size = BD_detected_slice_data.size();
			int BD_last_slice_detected = (int)BD_detected_slice_data[BD_slice_size - 1].V();
			int BD_first_slice_detected = (int)BD_detected_slice_data[0].V();
			int BD_total_slice_detected = BD_last_slice_detected - BD_first_slice_detected + 1;

			int BD_lesion_centre = abs((BD_first_slice_detected + BD_last_slice_detected)/2);

			cout << "Bone Deformation" << endl;
			cout << setw(8) << "Lesion detected from slice:" << setw(4) << BD_first_slice_detected << setw(4) << "to" << setw(4) << BD_last_slice_detected << endl;

			cout << "Approx. Lesion Center:" <<  endl;
			cout << "x-coordinate:" << setw(4) << BD_detected_slice_x_s*2 <<  endl;
			cout << "y-coordinate:" << setw(4) << BD_detected_slice_y_s*2 <<  endl;
			cout << "z-coordinate (slice number):" << setw(4) << BD_lesion_centre <<  endl << endl;
			cout << "Approx. Lesion size (Dia):" << setw(4) << BD_dia_mm << setw(8) << "mm (1 pixel = 0.2 mm)" << endl << endl;

			//---------------------------------------------------------------------------------------------------------------------------------------------------
			// Read detected image from original location
			const char *imgPath = full_list[BD_detected_slice_num_s].c_str();  //string to const char conversion
			FILE * pFile;
			long lSize;
			char * buffer;
			size_t result;

			pFile = fopen(imgPath, "rb");
			if (pFile == NULL) { fputs("File error", stderr); exit(1); }

			// obtain file size:
			fseek(pFile, 0, SEEK_END);
			lSize = ftell(pFile);
			rewind(pFile);

			// allocate memory to contain the whole file:
			buffer = (char*)malloc(sizeof(char)*lSize);

			if (buffer == NULL) { fputs("Memory error", stderr); exit(2); }

			// copy the file into the buffer:
			result = fread(buffer, 1, lSize, pFile);
			if (result != lSize) { fputs("Reading error", stderr); exit(3); }

			// clean up
			fclose(pFile);

			Mat image;
			int16_t *imageMap = (int16_t*)buffer;
			image.create(800, 800, CV_16SC1);
			memcpy(image.data, imageMap, 800 * 800 * sizeof(int16_t));

			std::free((void*)buffer);

			// Preprocessing
			Mat image16s_norm, image8u, image_400, image8s;

			normalize(image, image16s_norm, -32768, 32767, NORM_MINMAX);

			image8s = image16s_norm / 255.0;
			image8u = image8s + 128;
			image8u.convertTo(image8u, CV_8U);

			resize(image8u, image_400, Size(), 0.5, 0.5, 1);
			cvtColor(image_400, image_400, CV_GRAY2RGB);		// Converting grayscale to color to put red marker on image

			// Display detected image
			//Rect boundRect = Rect(BD_detected_slice_x_s - 20, BD_detected_slice_y_s - 20, 40, 40);

			//rectangle(image_400, boundRect, Scalar(0, 0, 255), +1, 8);
			sprintf(winName2, "Slice no:%d (Bone Deformation) ", BD_lesion_centre);

			std::ostringstream str;
			//str << "x:" << BD_detected_slice_x_s*2 << ", y:" << BD_detected_slice_y_s*2;
			//cv::putText(image_400, str.str(), cv::Point(BD_detected_slice_x_s - 30, BD_detected_slice_y_s - 30), FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(0, 255, 255));

			str << "x";
			cv::putText(image_400, str.str(), cv::Point(BD_detected_slice_x_s, BD_detected_slice_y_s),  FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0, 0, 255), 2);

			namedWindow(winName2, WINDOW_AUTOSIZE);
			imshow(winName2, image_400);
			cv::waitKey(60);
		}
	}
	//**********************************************************************************************************************************************************************
	//-------------------------------------------------------------------------------------------------------------------------------------------------------
		//Maxilla Bone deformation image position calculation and display
		//-------------------------------------------------------------------------------------------------------------------------------------------------------

		//Calculate Approx. lesion centre and average co-ordinate

		vector<double> BDM_total_x;			BDM_total_x.push_back(0);
		vector<double> BDM_total_y;			BDM_total_y.push_back(0);
		vector<double> BDM_total_slice;		BDM_total_slice.push_back(0);
		vector<double> BDM_total_area;		BDM_total_area.push_back(0);

		vector<double> BDM_av_x;			BDM_av_x.push_back(0);
		vector<double> BDM_av_y;			BDM_av_y.push_back(0);
		vector<double> BDM_av_slice;		BDM_av_slice.push_back(0);
		vector<double> BDM_av_area;			BDM_av_area.push_back(0);

		int BDM_k = 1;
		int BDM_j = 0;

		if (BDM_detected_slice_data.size() > 0)
		{
			for (size_t i = 0; i < (BDM_detected_slice_data.size()); i++)
			{
				BDM_total_x[BDM_j] += BDM_detected_slice_data[i].X();
				BDM_av_x[BDM_j] = BDM_total_x[BDM_j] / BDM_k;
				BDM_total_y[BDM_j] += BDM_detected_slice_data[i].Y();
				BDM_av_y[BDM_j] = BDM_total_y[BDM_j] / BDM_k;
				BDM_total_slice[BDM_j] += BDM_detected_slice_data[i].V();
				BDM_av_slice[BDM_j] = BDM_total_slice[BDM_j] / BDM_k;
				BDM_total_area[BDM_j] += BDM_detected_slice_data[i].Z();
				BDM_av_area[BDM_j] = BDM_total_area[BDM_j] / BDM_k;
				BDM_k++;
			}
		}

		// Image display

		char winName3[60];

		if (BDM_detected_slice_data.size() > 0)
		{
			for (size_t i = 0; i < BDM_av_slice.size(); i++)
			{
				//for (size_t k =0; k < )
				int BDM_detected_slice_num_s = (int)BDM_av_slice[i];
				int BDM_detected_slice_x_s = (int)BDM_av_x[i];
				int BDM_detected_slice_y_s = (int)BDM_av_y[i];
				int BDM_detected_slice_Area_s = (int)BDM_av_area[i];
				double BDM_equiv_dia = 1.5 * 2 * sqrt(4 * BDM_detected_slice_Area_s / 3.1416);  //1.5 is the correction coefficient
				double BDM_dia_mm = BDM_equiv_dia * 0.2;		// 1 pixel = 0.2 mm

				int BDM_slice_size = BDM_detected_slice_data.size();
				int BDM_last_slice_detected = (int)BDM_detected_slice_data[BDM_slice_size - 1].V();
				int BDM_first_slice_detected = (int)BDM_detected_slice_data[0].V();
				int BDM_total_slice_detected = BDM_last_slice_detected - BDM_first_slice_detected + 1;

				int BDM_lesion_centre = abs((BDM_first_slice_detected + BDM_last_slice_detected) / 2);

				cout << "Maxilla Bone Deformation" << endl;
				cout << setw(8) << "Lesion detected from slice:" << setw(4) << BDM_first_slice_detected << setw(4) << "to" << setw(4) << BDM_last_slice_detected << endl;

				cout << "Approx. Lesion Center:" <<  endl;
				cout << "x-coordinate:" << setw(4) << BDM_detected_slice_x_s*2 <<  endl;
				cout << "y-coordinate:" << setw(4) << BDM_detected_slice_y_s*2 <<  endl;
				cout << "z-coordinate (slice number):" << setw(4) << BDM_lesion_centre <<  endl << endl;
				cout << "Approx. Lesion size (Dia):" << setw(4) << BDM_dia_mm << setw(8) << "mm (1 pixel = 0.2 mm)" << endl << endl;

				//---------------------------------------------------------------------------------------------------------------------------------------------------
				// Read detected image from original location
				const char *imgPath = full_list[BDM_detected_slice_num_s].c_str();  //string to const char conversion
				FILE * pFile;
				long lSize;
				char * buffer;
				size_t result;

				pFile = fopen(imgPath, "rb");
				if (pFile == NULL) { fputs("File error", stderr); exit(1); }

				// obtain file size:
				fseek(pFile, 0, SEEK_END);
				lSize = ftell(pFile);
				rewind(pFile);

				// allocate memory to contain the whole file:
				buffer = (char*)malloc(sizeof(char)*lSize);

				if (buffer == NULL) { fputs("Memory error", stderr); exit(2); }

				// copy the file into the buffer:
				result = fread(buffer, 1, lSize, pFile);
				if (result != lSize) { fputs("Reading error", stderr); exit(3); }

				// clean up
				fclose(pFile);

				Mat image;
				int16_t *imageMap = (int16_t*)buffer;
				image.create(800, 800, CV_16SC1);
				memcpy(image.data, imageMap, 800 * 800 * sizeof(int16_t));

				std::free((void*)buffer);

				// Preprocessing
				Mat image16s_norm, image8u, image_400, image8s;

				normalize(image, image16s_norm, -32768, 32767, NORM_MINMAX);

				image8s = image16s_norm / 255.0;
				image8u = image8s + 128;
				image8u.convertTo(image8u, CV_8U);

				resize(image8u, image_400, Size(), 0.5, 0.5, 1);
				cvtColor(image_400, image_400, CV_GRAY2RGB);		// Converting grayscale to color to put red marker on image

				// Display detected image
				//Rect boundRect = Rect(BDM_detected_slice_x_s - 20, BDM_detected_slice_y_s - 20, 40, 40);

				//rectangle(image_400, boundRect, Scalar(0, 0, 255), +1, 8);
				sprintf(winName3, "Slice no:%d ( Maxilla Bone Deformation) ", BDM_lesion_centre);

				std::ostringstream str;
				//str << "x:" << BDM_detected_slice_x_s * 2 << ", y:" << BDM_detected_slice_y_s * 2;
				//cv::putText(image_400, str.str(), cv::Point(BDM_detected_slice_x_s - 30, BDM_detected_slice_y_s - 30), FONT_HERSHEY_COMPLEX_SMALL, 0.5, cvScalar(0, 255, 255));

				str << "x";
				cv::putText(image_400, str.str(), cv::Point(BDM_detected_slice_x_s, BDM_detected_slice_y_s),  FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(0, 0, 255), 2);

				namedWindow(winName3, WINDOW_AUTOSIZE);
				imshow(winName3, image_400);
				cv::waitKey(60);
			}
		}

	//---------------------------------------------------------------------------------------------------------------
	// Total time
	t2 = clock();
	float diff2((float)t2 - (float)t1);
	float seconds2 = diff2 / CLOCKS_PER_SEC;
	cout << "Total running time (seconds):" << seconds2 << endl;

	if (detected_slice_data_serial_sorted.size() == 0 && BD_detected_slice_data.size() == 0  &&  BDM_detected_slice_data.size() == 0)		//if no lesion detected
	{
		cout << "No lesion found" << endl;
		system("pause");
	}

	//---------------------------------------------------------------------------------------------------------------
	waitKey();
	return 0;
}

//***************************************************************************************************************
// Custom Functions
//***************************************************************************************************************

//---------------------------------------------------------------------------------------------------------------
// Get file names from image directory
//---------------------------------------------------------------------------------------------------------------

void GetFilesInDirectory(std::vector<string> &out, const string &directory)
{
	HANDLE dir;
	WIN32_FIND_DATA file_data;

	if ((dir = FindFirstFile((directory + "\\*.slice").c_str(), &file_data)) == INVALID_HANDLE_VALUE)
		return; /* No files found */

	do {
		const string file_name = file_data.cFileName;
		const string full_file_name = directory + "\\" + file_name;
		const bool is_directory = (file_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

		if (file_name[0] == '.')
			continue;

		if (is_directory)
			continue;

		out.push_back(full_file_name);
	} while (FindNextFile(dir, &file_data));

	FindClose(dir);
}

//---------------------------------------------------------------------------------------------------------------
// Calculate Kurtosis
//---------------------------------------------------------------------------------------------------------------

float calculateKurt(Mat image_400)
{
	Scalar  mean;
	Scalar  stddev;
	meanStdDev(image_400, mean, stddev);

	double mean_val = mean[0];			// scaler contains 4 values. Taking first value only
	double stddev_val = stddev[0];

	MatND hist;
	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	calcHist(&image_400, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

	float totalPixel = hist.at<float>(0);

	for (int h = 1; h < histSize; h++)
	{
		float binVal_thisContour = hist.at<float>(h);
		totalPixel = totalPixel + binVal_thisContour;
	}

	//kurtosis
	float yi_kur = 0;
	float sum_kur = 0;
	float kurtosis_val = 0;
	for (int h = 0; h < histSize; h++)
	{
		yi_kur = hist.at<float>(h);

		sum_kur += pow((h - (float)mean_val), 4) * yi_kur;
	}
	kurtosis_val = sum_kur / ((totalPixel - 1) * pow((float)stddev_val, 4));

	return kurtosis_val;
}

//---------------------------------------------------------------------------------------------------------------
// Calculate Max Entropy Threshold
//---------------------------------------------------------------------------------------------------------------

int maxEntropy(Mat image_400)
{
	// Initialize parameters
	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	// Calculate histogram
	MatND hist;
	calcHist(&image_400, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

	//normalize the histogram
	Mat hn = hist / (image_400.rows * image_400.cols);

	//Cumulative distribution function
	Mat c = Mat::zeros(256, 1, CV_32F);

	c.at<float>(0, 0) = hn.at<float>(0, 0);

	for (int l = 1; l < 256; l++)
	{
		c.at<float>(l, 0) = c.at<float>(l - 1, 0) + hn.at<float>(l, 0);
	}

	Mat hl = Mat::zeros(256, 1, CV_32F);
	Mat hh = Mat::zeros(256, 1, CV_32F);

	for (int t = 0; t < 256; t++)
	{
		//Low range entropy
		double cl = double(c.at<float>(t, 0));
		if (cl > 0)
		{
			for (int i = 0; i < t; i++)
			{
				if (hn.at<float>(i, 0) > 0)
				{
					hl.at<float>(t, 0) = hl.at<float>(t, 0) - (hn.at<float>(i, 0) / (float)cl) * log(hn.at<float>(i, 0) / (float)cl);
				}
			}
		}

		// High range entropy
		double ch = double(1.0 - cl);
		if (ch > 0)
		{
			for (int i = t; i < 256; i++)
			{
				if (hn.at<float>(i, 0) > 0)
				{
					hh.at<float>(t, 0) = hh.at<float>(t, 0) - (hn.at<float>(i, 0) / (float)ch) * log(hn.at<float>(i, 0) / (float)ch);
				}
			}
		}
	}

	//Choose best threshold
	double h_max = hl.at<float>(0, 0) + hh.at<float>(0, 0);
	double threshold_val = 0;
	Mat entropie = Mat::zeros(256, 1, CV_32F);
	entropie.at<float>(0, 0) = (float)h_max;

	for (int j = 1; j < 256; j++)
	{
		entropie.at<float>(j, 0) = hl.at<float>(j, 0) + hh.at<float>(j, 0);

		if (entropie.at<float>(j, 0) > h_max)
		{
			h_max = entropie.at<float>(j, 0);
			threshold_val = j - 1;
		}
	}

	//cout << " " << threshold_val << endl;

	return((int)threshold_val);
}

//---------------------------------------------------------------------------------------------------------------
// Calculate Skewness and Kurtosis
//---------------------------------------------------------------------------------------------------------------

SkewKurt calculateSkewAndKurt(Mat image_400, Mat mask, double mean_val, double stddev_val)
{
	MatND hist_contour;
	int histSize = 256;    // bin size
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	calcHist(&image_400, 1, 0, mask, hist_contour, 1, &histSize, ranges, true, false);

	float totalPixelinContour = hist_contour.at<float>(0);

	for (int h = 1; h < histSize; h++)
	{
		float binVal_thisContour = hist_contour.at<float>(h);
		totalPixelinContour = totalPixelinContour + binVal_thisContour;
	}

	//Skewness

	float yi_skw = 0;
	float sum_skw = 0;
	float skewness_val = 0;
	for (int h = 0; h < histSize; h++)
	{
		yi_skw = hist_contour.at<float>(h);

		sum_skw += pow((h - (float)mean_val), 3)*yi_skw;
	}
	skewness_val = sum_skw / ((totalPixelinContour - 1) * pow((float)stddev_val, 3));

	//kurtosis

	float yi_kur = 0;
	float sum_kur = 0;
	float kurtosis_val = 0;
	for (int h = 0; h < histSize; h++)
	{
		yi_kur = hist_contour.at<float>(h);

		sum_kur += pow((h - (float)mean_val), 4) * yi_kur;
	}
	kurtosis_val = sum_kur / ((totalPixelinContour - 1) * pow((float)stddev_val, 4));

	SkewKurt result = { skewness_val, kurtosis_val };

	return result;
}

//---------------------------------------------------------------------------------------------------------------
// Find Mandible Bone Deformation
//---------------------------------------------------------------------------------------------------------------

BDdata findBoneDeform(Mat image_400, Mat fuse_image, Mat image_binary, int m)
{
	Mat image_fuse_close, image_fuse_open, complement_image;

	// Creating the diamond-shaped structuring element
	Mat diamond(7, 7, CV_8U, cv::Scalar(1));

	diamond.at<uchar>(0, 0) = 0;
	diamond.at<uchar>(0, 1) = 0;
	diamond.at<uchar>(0, 2) = 0;
	diamond.at<uchar>(1, 0) = 0;
	diamond.at<uchar>(1, 1) = 0;
	diamond.at<uchar>(2, 0) = 0;
	diamond.at<uchar>(4, 0) = 0;
	diamond.at<uchar>(5, 0) = 0;
	diamond.at<uchar>(5, 1) = 0;
	diamond.at<uchar>(5, 0) = 0;
	diamond.at<uchar>(6, 0) = 0;
	diamond.at<uchar>(6, 1) = 0;
	diamond.at<uchar>(6, 2) = 0;
	diamond.at<uchar>(0, 4) = 0;
	diamond.at<uchar>(0, 5) = 0;
	diamond.at<uchar>(0, 6) = 0;
	diamond.at<uchar>(1, 5) = 0;
	diamond.at<uchar>(1, 6) = 0;
	diamond.at<uchar>(4, 6) = 0;
	diamond.at<uchar>(5, 5) = 0;
	diamond.at<uchar>(5, 6) = 0;
	diamond.at<uchar>(6, 4) = 0;
	diamond.at<uchar>(6, 5) = 0;
	diamond.at<uchar>(6, 6) = 0;

	Mat element_closing = diamond;
	Mat element_opening = getStructuringElement(MORPH_RECT, Size(2, 2));

	int morph_size = 5;
	Mat element_closing_big = getStructuringElement(MORPH_ELLIPSE, Size(5 * morph_size + 1, 5 * morph_size + 1), Point(13, 13));

	//Close Fuse image
	morphologyEx(fuse_image, image_fuse_close, MORPH_CLOSE, element_closing);

	//Open closed image
	morphologyEx(image_fuse_close, image_fuse_open, MORPH_OPEN, element_opening);

	// Contour Detection

	vector<vector<Point> > contours;			// stores boundary coordinates of contours

	/// Find contours: create a copy of image_fuse_open (image_fuse_open_copy)
	Mat image_fuse_open_copy;
	image_fuse_open.copyTo(image_fuse_open_copy);

	findContours(image_fuse_open_copy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);   // we can get area of each contour from 'contourArea'

	// Calculate Centroid of each contour
	vector<Moments> contourMoments(contours.size());			// stores moments of each contour
	for (size_t i = 0; i < contours.size(); i++)
	{
		contourMoments[i] = moments(contours[i], false);
	}

	vector<Point2f> ContCenter(contours.size());				// stores centroid of each contour
	for (size_t i = 0; i < contours.size(); i++)
	{
		ContCenter[i] = Point2f((float)contourMoments[i].m10 / (float)contourMoments[i].m00, (float)contourMoments[i].m01 / (float)contourMoments[i].m00);
	}

	// Sort and save contours by area and centroid (Expected as mandible)
	vector<int> ContourIndex;			// Stores sorted contour #
	vector<double> cAr;

	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) > 6000 && ContCenter[i].y < 250)
		{
			ContourIndex.push_back(i);			// ContourIndex contains sorted contour # (Mandible)
			cAr.push_back(contourArea(contours[i]));		// Mandible center
		}
	}

	if (ContourIndex.size() > 0)		// Mandible only
	{
		for (size_t i = 0; i < ContourIndex.size(); i++)			// Go for each sorted contour
		{
			Mat mask_small = Mat::zeros(image_400.rows, image_400.cols, CV_8UC1);		// Image with small closing element
			drawContours(mask_small, contours, ContourIndex[i], Scalar(255), CV_FILLED); // i= contour index #

			Rect boundRect;
			boundRect = boundingRect(Mat(contours[ContourIndex[i]]));
			int max_lesion_height = boundRect.y + 70;

			// process mask image
			Mat mask_big = Mat::zeros(image_400.rows, image_400.cols, CV_8UC1);		//image with big closing element

			morphologyEx(mask_small, mask_big, MORPH_CLOSE, element_closing_big);

			//---------------------------------------------------------------------------------------------------------------
			// Subtract Image
			Mat subtract_image;				//Subtract the small closed image from big closed image
			absdiff(mask_big, mask_small, subtract_image);

			vector<vector<Point> > contours_sub;

			findContours(subtract_image, contours_sub, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

			vector<Moments> contourMoments_sub(contours_sub.size());			// stores moments of each contour
			for (size_t i = 0; i < contours_sub.size(); i++)
			{
				contourMoments_sub[i] = moments(contours_sub[i], false);
			}

			vector<Point2f> ContCenter_sub(contours_sub.size());				// stores centroid of each contour
			for (size_t i = 0; i < contours_sub.size(); i++)
			{
				ContCenter_sub[i] = Point2f((float)contourMoments_sub[i].m10 / (float)contourMoments_sub[i].m00, (float)contourMoments_sub[i].m01 / (float)contourMoments_sub[i].m00);
			}

			vector<int> ContourIndex_sub;			// Stores sorted contour #
			vector<double> ContourCenter_sub_x;
			vector<double> ContourCenter_sub_y;

			int slice_num = 0;
			float x_val = 0;
			float y_val = 0;
			float Area_val = 0;

			for (size_t i = 0; i < contours_sub.size(); i++)
			{
				if (contourArea(contours_sub[i]) > 270 && contourArea(contours_sub[i]) < 600 && ContCenter_sub[i].y > max_lesion_height)
				{
					slice_num = m;
					x_val = ContCenter_sub[i].x;
					y_val = ContCenter_sub[i].y;
					Area_val = (float)contourArea(contours_sub[i]);
				}
			}

			BDdata loc_data = { slice_num, x_val, y_val, Area_val };

			return loc_data;
		}
	}
	else
	{
		int slice_num = 0;
		float x_val = 0;
		float y_val = 0;
		float Area_val = 0;

		BDdata loc_data = { slice_num, x_val, y_val, Area_val };

		return loc_data;
	}
}

//---------------------------------------------------------------------------------------------------------------
// Find Maxilla Bone Deformation
//---------------------------------------------------------------------------------------------------------------
BDdata findBoneDeform_maxl(Mat image_binary, int m)
{
	Mat image_fuse_close, image_binary_open, complement_image;

	int morph_size = 5;
	Mat element_opening = getStructuringElement(MORPH_RECT, Size(4, 4));
	Mat element_closing_big = getStructuringElement(MORPH_ELLIPSE, Size(8 * morph_size + 1, 8 * morph_size + 1), Point(21, 21));
	Mat element_closing = getStructuringElement(MORPH_ELLIPSE, Size(10, 10), Point(5, 5));
	//-------------------------------------------------------------------------------------------------------------------------------------
	//Open binary image

	morphologyEx(image_binary, image_binary_open, MORPH_OPEN, element_opening);
	//-------------------------------------------------------------------------------------------------------------------------------------
	// Contour Detection

	vector<vector<Point> > contours;			// stores boundary coordinates of contours
	vector<vector<Point> > contours_mask_big;

	/// Find contours: create a copy of image_fuse_open (image_fuse_open_copy)
	Mat image_binary_open_copy;
	image_binary_open.copyTo(image_binary_open_copy);

	findContours(image_binary_open_copy, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);   // we can get area of each contour from 'contourArea'

	// Calculate Centroid of each contour
	vector<Moments> contourMoments(contours.size());			// stores moments of each contour
	for (size_t i = 0; i < contours.size(); i++)
	{
		contourMoments[i] = moments(contours[i], false);
	}

	vector<Point2f> ContCenter(contours.size());				// stores centroid of each contour
	for (size_t i = 0; i < contours.size(); i++)
	{
		ContCenter[i] = Point2f((float)contourMoments[i].m10 / (float)contourMoments[i].m00, (float)contourMoments[i].m01 / (float)contourMoments[i].m00);
	}

	// Sort and save contours by area and centroid (Expected as maxilla)
	vector<int> ContourIndex;			// Stores sorted contour #
	vector<double> cAr;

	for (size_t i = 0; i < contours.size(); i++)
	{
		if (contourArea(contours[i]) < 15000 && contourArea(contours[i]) > 1000 && ContCenter[i].y < 250)
		{
			ContourIndex.push_back(i);			// ContourIndex contains sorted contour # (Mandible)
			cAr.push_back(contourArea(contours[i]));		// Mandible center
		}
	}

	if (ContourIndex.size() > 0)
	{
		Mat mask_open = Mat::zeros(image_binary.rows, image_binary.cols, CV_8UC1);	// Image with small closing element
		Mat mask_big = Mat::zeros(image_binary.rows, image_binary.cols, CV_8UC1);		//image with big closing element
		Mat mask_small = Mat::zeros(image_binary.rows, image_binary.cols, CV_8UC1);

		for (size_t i = 0; i < ContourIndex.size(); i++)			// Go for each sorted contour
		{
			drawContours(mask_open, contours, ContourIndex[i], Scalar(255), CV_FILLED); // i= contour index #
		}

		// process mask image
		//Close Fuse image
		morphologyEx(mask_open, mask_small, MORPH_CLOSE, element_closing);

		morphologyEx(mask_small, mask_big, MORPH_CLOSE, element_closing_big);
		//find position of mask_big

		Mat image_mask_big_copy;
		mask_big.copyTo(image_mask_big_copy);

		findContours(image_mask_big_copy, contours_mask_big, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);   // we can get area of each contour from 'contourArea'

		Rect boundRect_mask_big;
		vector<Point> point_all;

		for (size_t i = 0; i < contours_mask_big.size(); i++)
		{
			for (size_t j = 0; j < contours_mask_big[i].size(); j++)
			{
				point_all.push_back(contours_mask_big[i][j]);
			}
		}

		boundRect_mask_big = boundingRect(point_all);
		int min_lesion_height = boundRect_mask_big.y + 30;

		// Subtract Image
		Mat subtract_image;				//Subtract the small closed image from big closed image
		absdiff(mask_big, mask_small, subtract_image);

		// Find contours in subtracted image
		vector<vector<Point> > contours_sub;

		findContours(subtract_image, contours_sub, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

		vector<Moments> contourMoments_sub(contours_sub.size());			// stores moments of each contour
		for (size_t i = 0; i < contours_sub.size(); i++)
		{
			contourMoments_sub[i] = moments(contours_sub[i], false);
		}

		vector<Point2f> ContCenter_sub(contours_sub.size());				// stores centroid of each contour
		for (size_t i = 0; i < contours_sub.size(); i++)
		{
			ContCenter_sub[i] = Point2f((float)contourMoments_sub[i].m10 / (float)contourMoments_sub[i].m00, (float)contourMoments_sub[i].m01 / (float)contourMoments_sub[i].m00);
		}

		vector<int> ContourIndex_sub;			// Stores sorted contour #
		vector<double> ContourCenter_sub_x;
		vector<double> ContourCenter_sub_y;

		int slice_num = 0;
		float x_val = 0;
		float y_val = 0;
		float Area_val = 0;

		for (size_t i = 0; i < contours_sub.size(); i++)
		{
			if (contourArea(contours_sub[i]) > 600 && contourArea(contours_sub[i]) < 2000 && ContCenter_sub[i].y < min_lesion_height)
			{
				slice_num = m;
				x_val = ContCenter_sub[i].x;
				y_val = ContCenter_sub[i].y;
				Area_val = (float)contourArea(contours_sub[i]);
			}
		}

		BDdata loc_data = { slice_num, x_val, y_val, Area_val };

		return loc_data;
	}
	else
	{
		int slice_num = 0;
		float x_val = 0;
		float y_val = 0;
		float Area_val = 0;

		BDdata loc_data = { slice_num, x_val, y_val, Area_val };

		return loc_data;
	}
}

