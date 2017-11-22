//#include "stdafx.h"
//#include <precomp.hpp>
#include "cl_Texture.h"
#include "opencv2/legacy/legacy.hpp"
#include "opencv2/legacy/compat.hpp"

cl_Texture::cl_Texture(void)
{
}

cl_Texture::~cl_Texture(void)
{
}
// Calculation of a texture descriptors from GLCM (Grey Level Co-occurrence Matrix'es) The code was submitted by Daniel Eaton [danieljameseaton@...]

cl_Texture::GLCM* cl_Texture::CreateGLCM(const IplImage* srcImage, int stepMagnitude, const int* srcStepDirections, int numStepDirections, int optimizationType)
{
	static const int defaultStepDirections[] = { 0, 1, -1, 1, -1, 0, -1, -1 };
	int* memorySteps = 0;
	cl_Texture::GLCM* newGLCM = 0;
	int* stepDirections = 0;
	uchar* srcImageData = 0;
	CvSize srcImageSize;
	int srcImageStep;
	int stepLoop;
	const int maxNumGreyLevels8u = CV_MAX_NUM_GREY_LEVELS_8U;

	if (!srcImage) return NULL;
	if (srcImage->nChannels != 1) return NULL;
	if (srcImage->depth != IPL_DEPTH_8U) return NULL;
	// Schrittrichtung zur Berechnung der GLCM
	if (!srcStepDirections){
		srcStepDirections = defaultStepDirections;
	}
	stepDirections = new int[numStepDirections * 2];
	memcpy(stepDirections, srcStepDirections, numStepDirections * 2 * sizeof(stepDirections[0]));

	cvGetImageRawData(srcImage, &srcImageData, &srcImageStep, &srcImageSize);

	// roll together Directions and magnitudes together with knowledge of image (step)
	memorySteps = new int[numStepDirections];
	for (stepLoop = 0; stepLoop < numStepDirections; stepLoop++)
	{
		stepDirections[stepLoop * 2 + 0] *= stepMagnitude;
		stepDirections[stepLoop * 2 + 1] *= stepMagnitude;
		memorySteps[stepLoop] = stepDirections[stepLoop * 2 + 0] * srcImageStep + stepDirections[stepLoop * 2 + 1];
	}
	//CV_CALL( newGLCM = (Cv_GLCM*)cvAlloc(sizeof(newGLCM)));
	newGLCM = new cl_Texture::GLCM;
	size_t size = sizeof(*newGLCM);
	memset(newGLCM, 0, size);
	newGLCM->matrices = 0;
	newGLCM->numMatrices = numStepDirections;
	newGLCM->optimizationType = optimizationType;
	if (optimizationType <= CV_GLCM_OPTIMIZATION_LUT){
		int lookupTableLoop, imageColLoop, imageRowLoop, lineOffset = 0;
		// if optimization type is set to lut, then make one for the image
		if (optimizationType == CV_GLCM_OPTIMIZATION_LUT){
			for (imageRowLoop = 0; imageRowLoop < srcImageSize.height; imageRowLoop++, lineOffset += srcImageStep){
				for (imageColLoop = 0; imageColLoop < srcImageSize.width; imageColLoop++){
					newGLCM->forwardLookupTable[srcImageData[lineOffset + imageColLoop]] = 1;
				}
			}
			newGLCM->numLookupTableElements = 0;
			for (lookupTableLoop = 0; lookupTableLoop < maxNumGreyLevels8u; lookupTableLoop++){
				if (newGLCM->forwardLookupTable[lookupTableLoop] != 0){
					newGLCM->forwardLookupTable[lookupTableLoop] = newGLCM->numLookupTableElements;
					newGLCM->reverseLookupTable[newGLCM->numLookupTableElements] = lookupTableLoop;
					newGLCM->numLookupTableElements++;
				}
			}
		}
		// otherwise make a "LUT" which contains all the gray-levels (for code-reuse)
		else if (optimizationType == CV_GLCM_OPTIMIZATION_NONE){
			for (lookupTableLoop = 0; lookupTableLoop < maxNumGreyLevels8u; lookupTableLoop++){
				newGLCM->forwardLookupTable[lookupTableLoop] = lookupTableLoop;
				newGLCM->reverseLookupTable[lookupTableLoop] = lookupTableLoop;
			}
			newGLCM->numLookupTableElements = maxNumGreyLevels8u;
		}
		newGLCM->matrixSideLength = newGLCM->numLookupTableElements;

		CreateGLCM_LookupTable_8u_C1R(srcImageData, srcImageStep, srcImageSize, newGLCM, stepDirections, numStepDirections, memorySteps);
	}
	else if (optimizationType == CV_GLCM_OPTIMIZATION_HISTOGRAM){
		throw std::exception("Histogram-based method is not implemented");
		/* newGLCM->numMatrices *= 2;
		newGLCM->matrixSideLength = maxNumGreyLevels8u*2;
		icv_CreateGLCM_Histogram_8uC1R( srcImageStep, srcImageSize,	srcImageData,newGLCM, numStepDirections,stepDirections, memorySteps );*/
	}
	delete[] memorySteps;
	delete[] stepDirections;
	if (cvGetErrStatus() < 0){
		delete[] newGLCM;
	}
	return newGLCM;
}

void cl_Texture::ReleaseGLCM(cl_Texture::GLCM** GLCM, int flag)
{
	int matrixLoop;
	if (!GLCM)
		throw std::exception("!GLMC");
	if (*GLCM){
		if ((flag == CV_GLCM_GLCM || flag == CV_GLCM_ALL) && (*GLCM)->matrices){
			for (matrixLoop = 0; matrixLoop < (*GLCM)->numMatrices; matrixLoop++){
				if ((*GLCM)->matrices[matrixLoop]){
					delete[](*GLCM)->matrices[matrixLoop];
					delete[]((*GLCM)->matrices + matrixLoop);
				}
			}
			delete[](*GLCM)->matrices;
		}
		if ((flag == CV_GLCM_DESC || flag == CV_GLCM_ALL) && (*GLCM)->descriptors)
		{
			for (matrixLoop = 0; matrixLoop < (*GLCM)->numMatrices; matrixLoop++)
			{
				delete[]((*GLCM)->descriptors + matrixLoop);
			}
			delete[](*GLCM)->descriptors;
		}
		if (flag == CV_GLCM_ALL)
		{
			delete *GLCM;
		}
	}
}

void cl_Texture::CreateGLCM_LookupTable_8u_C1R(const uchar* srcImageData, int srcImageStep, CvSize srcImageSize, cl_Texture::GLCM* destGLCM, int* steps, int numSteps, int* memorySteps){
	int* stepIncrementsCounter = 0;
	int matrixSideLength = destGLCM->matrixSideLength;
	int stepLoop, sideLoop1, sideLoop2;
	int colLoop, rowLoop, lineOffset = 0;
	double*** matrices = 0;
	// allocate memory to the matrices
	//CV_CALL( destGLCM->matrices = (double***)cvAlloc(	sizeof(matrices[0])*numSteps ));
	destGLCM->matrices = new double**[numSteps];
	matrices = destGLCM->matrices;
	for (stepLoop = 0; stepLoop < numSteps; stepLoop++)
	{
		/*CV_CALL( matrices[stepLoop] = (double**)cvAlloc( sizeof(matrices[0])*matrixSideLength ));
		CV_CALL( matrices[stepLoop][0] = (double*)cvAlloc(sizeof(matrices[0][0])* matrixSideLength*matrixSideLength ));*/
		matrices[stepLoop] = new double*[matrixSideLength];
		matrices[stepLoop][0] = new double[matrixSideLength*matrixSideLength];
		size_t size = sizeof(matrices[stepLoop][0][0])*matrixSideLength*matrixSideLength;
		memset(matrices[stepLoop][0], 0, size);
		for (sideLoop1 = 1; sideLoop1 < matrixSideLength; sideLoop1++){
			matrices[stepLoop][sideLoop1] = matrices[stepLoop][sideLoop1 - 1] + matrixSideLength;
		}
	}
	//CV_CALL( stepIncrementsCounter = (int*)cvAlloc( numSteps*sizeof(stepIncrementsCounter[0])));
	stepIncrementsCounter = new int[numSteps];
	memset(stepIncrementsCounter, 0, numSteps*sizeof(stepIncrementsCounter[0]));
	// generate GLCM for each step
	for (rowLoop = 0; rowLoop < srcImageSize.height; rowLoop++, lineOffset += srcImageStep){
		for (colLoop = 0; colLoop < srcImageSize.width; colLoop++){
			int pixelValue1 = destGLCM->forwardLookupTable[srcImageData[lineOffset + colLoop]];
			for (stepLoop = 0; stepLoop < numSteps; stepLoop++){
				int col2, row2;
				row2 = rowLoop + steps[stepLoop * 2 + 0];
				col2 = colLoop + steps[stepLoop * 2 + 1];
				if (col2 >= 0 && row2 >= 0 && col2 < srcImageSize.width && row2 < srcImageSize.height){
					int memoryStep = memorySteps[stepLoop];
					int pixelValue2 = destGLCM->forwardLookupTable[srcImageData[lineOffset + colLoop + memoryStep]];
					// maintain symmetry
					matrices[stepLoop][pixelValue1][pixelValue2] ++;
					matrices[stepLoop][pixelValue2][pixelValue1] ++;
					// incremenet counter of total number of increments
					stepIncrementsCounter[stepLoop] += 2;
				}
			}
		}
	}
	// normalize matrices. each element is a probability of gray value i,j adjacency in direction/magnitude k
	for (sideLoop1 = 0; sideLoop1 < matrixSideLength; sideLoop1++){
		for (sideLoop2 = 0; sideLoop2 < matrixSideLength; sideLoop2++){
			for (stepLoop = 0; stepLoop < numSteps; stepLoop++){
				matrices[stepLoop][sideLoop1][sideLoop2] /= double(stepIncrementsCounter[stepLoop]);
			}
		}
	}
	destGLCM->matrices = matrices;
	delete stepIncrementsCounter;
	if (cvGetErrStatus() < 0)
		ReleaseGLCM(&destGLCM, CV_GLCM_GLCM);
}
void cl_Texture::CreateGLCMDescriptors(cl_Texture::GLCM* destGLCM, int descriptorOptimizationType){
	int matrixLoop;
	if (!destGLCM)
		throw std::exception("!destGLCM");
	if (!(destGLCM->matrices))
		throw std::exception("Matrices are not allocated");
	ReleaseGLCM(&destGLCM, CV_GLCM_DESC);
	if (destGLCM->optimizationType != CV_GLCM_OPTIMIZATION_HISTOGRAM){
		destGLCM->descriptorOptimizationType = destGLCM->numDescriptors = descriptorOptimizationType;
	}
	else{
		throw std::exception("Histogram-based method is not implemented");
		// destGLCM->descriptorOptimizationType = destGLCM->numDescriptors =	CV_GLCMDESC_OPTIMIZATION_HISTOGRAM;
	}
	//CV_CALL( destGLCM->descriptors = (double**)
	//cvAlloc( destGLCM->numMatrices*sizeof(destGLCM->descriptors[0])));
	destGLCM->descriptors = new double*[destGLCM->numMatrices];
	memset(destGLCM->descriptors, 0, destGLCM->numMatrices*sizeof(destGLCM->descriptors[0]));
	for (matrixLoop = 0; matrixLoop < destGLCM->numMatrices; matrixLoop++){
		//CV_CALL( destGLCM->descriptors[ matrixLoop ] =//(double*)cvAlloc(destGLCM->numDescriptors*sizeof(destGLCM->descriptors[0][0])));
		destGLCM->descriptors[matrixLoop] = new double[destGLCM->numDescriptors];
		memset(destGLCM->descriptors[matrixLoop], 0, destGLCM->numDescriptors*sizeof(destGLCM->descriptors[0][0]));
		switch (destGLCM->descriptorOptimizationType){
		case CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST:
			CreateGLCMDescriptors_AllowDoubleNest(destGLCM, matrixLoop);
			break;
		default:
			throw std::exception("descriptorOptimizationType different from CV_GLCMDESC_OPTIMIZATION_ALLOWDOUBLENEST\n" "is not supported");
			/*
			case CV_GLCMDESC_OPTIMIZATION_ALLOWTRIPLENEST:
			icvCreateGLCMDescriptors_AllowTripleNest( destGLCM, matrixLoop			);
			break;
			case CV_GLCMDESC_OPTIMIZATION_HISTOGRAM:
			if(matrixLoop < destGLCM->numMatrices>>1)
			icvCreateGLCMDescriptors_Histogram( destGLCM, matrixLoop);
			break;
			*/
		}
	}
	if (cvGetErrStatus() < 0)
		ReleaseGLCM(&destGLCM, CV_GLCM_DESC);
}
void::cl_Texture::CreateGLCMDescriptors_AllowDoubleNest(GLCM* destGLCM, int matrixIndex){
	int sideLoop1, sideLoop2;
	int matrixSideLength = destGLCM->matrixSideLength;
	double** matrix = destGLCM->matrices[matrixIndex];
	double* descriptors = destGLCM->descriptors[matrixIndex];
	//double* marginalProbability = //(double*)cvAlloc( matrixSideLength * sizeof(marginalProbability[0]));
	double* marginalProbability = new double[matrixSideLength];
	memset(marginalProbability, 0, matrixSideLength * sizeof(double));
	double maximumProbability = 0;
	double marginalProbabilityEntropy = 0;
	double correlationMean = 0, correlationStdDeviation = 0,
		correlationProductTerm = 0;
	for (sideLoop1 = 0; sideLoop1 < matrixSideLength; sideLoop1++){
		int actualSideLoop1 = destGLCM->reverseLookupTable[sideLoop1];
		for (sideLoop2 = 0; sideLoop2 < matrixSideLength; sideLoop2++){
			double entryValue = matrix[sideLoop1][sideLoop2];
			int actualSideLoop2 = destGLCM->reverseLookupTable[sideLoop2];
			int sideLoopDifference = actualSideLoop1 - actualSideLoop2;
			int sideLoopDifferenceSquared = sideLoopDifference*sideLoopDifference;
			marginalProbability[sideLoop1] += entryValue;
			correlationMean += actualSideLoop1*entryValue;
			maximumProbability = MAX(maximumProbability, entryValue);
			if (actualSideLoop2 > actualSideLoop1){
				descriptors[CV_GLCMDESC_CONTRAST] += sideLoopDifferenceSquared * entryValue;
			}
			descriptors[CV_GLCMDESC_HOMOGENITY] += entryValue / (1.0 + sqrt((double)sideLoopDifferenceSquared));
			if (entryValue > 0){
				descriptors[CV_GLCMDESC_ENTROPY] += entryValue * log(entryValue);
			}
			descriptors[CV_GLCMDESC_ENERGY] += entryValue*entryValue;
		}
		if (marginalProbability > 0)
			marginalProbabilityEntropy += marginalProbability[actualSideLoop1] * log(marginalProbability[actualSideLoop1]);
	}
	marginalProbabilityEntropy = -marginalProbabilityEntropy;
	descriptors[CV_GLCMDESC_CONTRAST] += descriptors[CV_GLCMDESC_CONTRAST];
	descriptors[CV_GLCMDESC_ENTROPY] = -descriptors[CV_GLCMDESC_ENTROPY];
	descriptors[CV_GLCMDESC_MAXIMUMPROBABILITY] = maximumProbability;
	double HXY = 0, HXY1 = 0, HXY2 = 0;
	HXY = descriptors[CV_GLCMDESC_ENTROPY];
	for (sideLoop1 = 0; sideLoop1 < matrixSideLength; sideLoop1++){
		double sideEntryValueSum = 0;
		int actualSideLoop1 = destGLCM->reverseLookupTable[sideLoop1];
		for (sideLoop2 = 0; sideLoop2 < matrixSideLength; sideLoop2++){
			double entryValue = matrix[sideLoop1][sideLoop2];
			sideEntryValueSum += entryValue;
			int actualSideLoop2 = destGLCM->reverseLookupTable[sideLoop2];
			correlationProductTerm += (actualSideLoop1 - correlationMean) *(actualSideLoop2 - correlationMean) * entryValue;
			double clusterTerm = actualSideLoop1 + actualSideLoop2 - correlationMean - correlationMean;
			descriptors[CV_GLCMDESC_CLUSTERTENDENCY] += clusterTerm * clusterTerm * entryValue;
			descriptors[CV_GLCMDESC_CLUSTERSHADE] += clusterTerm * clusterTerm * clusterTerm * entryValue;
			double HXYValue = marginalProbability[actualSideLoop1] * marginalProbability[actualSideLoop2];
			if (HXYValue > 0){
				double HXYValueLog = log(HXYValue);
				HXY1 += entryValue * HXYValueLog;
				HXY2 += HXYValue * HXYValueLog;
			}
		}
		correlationStdDeviation += (actualSideLoop1 - correlationMean) * (actualSideLoop1 - correlationMean) * sideEntryValueSum;
	}
	HXY1 = -HXY1;
	HXY2 = -HXY2;
	descriptors[CV_GLCMDESC_CORRELATIONINFO1] = (HXY - HXY1) / (correlationMean);
	descriptors[CV_GLCMDESC_CORRELATIONINFO2] = sqrt(1.0 - exp(-2.0 * (HXY2 - HXY)));
	correlationStdDeviation = sqrt(correlationStdDeviation);
	descriptors[CV_GLCMDESC_CORRELATION] = correlationProductTerm / (correlationStdDeviation*correlationStdDeviation);
	delete[] marginalProbability;
}

double cl_Texture::GetGLCMDescriptor(cl_Texture::GLCM* GLCM, int step, int descriptor){
	double value = DBL_MAX;
	if (!GLCM)
		throw std::exception("!GLCM");
	if (!(GLCM->descriptors))
		throw std::exception("Descriptors are not calculated");
	if ((unsigned)step >= (unsigned)(GLCM->numMatrices))
		throw std::exception("step is not in 0 .. GLCM->numMatrices - 1");
	if ((unsigned)descriptor >= (unsigned)(GLCM->numDescriptors))
		throw std::exception("descriptor is not in 0 ..GLCM->numDescriptors - 1");
	value = GLCM->descriptors[step][descriptor];
	return value;
}
void cl_Texture::GetGLCMDescriptorStatistics(cl_Texture::GLCM* GLCM, int descriptor, double* _average, double* _standardDeviation){
	if (_average)
		*_average = DBL_MAX;
	if (_standardDeviation)
		*_standardDeviation = DBL_MAX;
	int matrixLoop, numMatrices;
	double average = 0, squareSum = 0;
	if (!GLCM)
		throw std::exception("!GLCM");
	if (!(GLCM->descriptors))
		throw std::exception("Descriptors are not calculated");
	if ((unsigned)descriptor >= (unsigned)(GLCM->numDescriptors))
		throw std::exception("Descriptor index is out of range");
	numMatrices = GLCM->numMatrices;
	for (matrixLoop = 0; matrixLoop < numMatrices; matrixLoop++){
		double temp = GLCM->descriptors[matrixLoop][descriptor];
		average += temp;
		squareSum += temp*temp;
	}
	average /= numMatrices;
	if (_average)
		*_average = average;
	if (_standardDeviation)
		*_standardDeviation = sqrt((squareSum - average*average*numMatrices) / (numMatrices - 1));
}
IplImage* cl_Texture::CreateGLCMImage(cl_Texture::GLCM* GLCM, int step){
	IplImage* dest = 0;
	float* destData;
	int sideLoop1, sideLoop2;
	if (!GLCM)
		throw std::exception("!GLCM");
	if (!(GLCM->matrices))
		throw std::exception("Matrices are not allocated");
	if ((unsigned)step >= (unsigned)(GLCM->numMatrices))
		throw std::exception("The step index is out of range");
	dest = cvCreateImage(cvSize(GLCM->matrixSideLength, GLCM->matrixSideLength), IPL_DEPTH_32F, 1);
	destData = (float*)(dest->imageData);
	for (sideLoop1 = 0; sideLoop1 < GLCM->matrixSideLength; sideLoop1++, (float*&)destData += dest->widthStep){
		for (sideLoop2 = 0; sideLoop2 < GLCM->matrixSideLength; sideLoop2++){
			double matrixValue = GLCM->matrices[step][sideLoop1][sideLoop2]; destData[sideLoop2] = (float)matrixValue;
		}
	}
	if (cvGetErrStatus() < 0)
		cvReleaseImage(&dest);
	return dest;
}