// TrainNetwork.cpp : Defines the entry point for the console application.

#include "opencv2/opencv.hpp"    // opencv general include file
#include "opencv2/ml/ml.hpp"          // opencv machine learning include file
#include <stdio.h>
#include <fstream>
using namespace std;
/******************************************************************************/

#define TRAINING_SAMPLES 1002   //1183		//Number of samples in training dataset
#define ATTRIBUTES 15				//Number of features per sample.
#define TEST_SAMPLES 1002		//Number of samples in test dataset
#define CLASSES 2					//Number of distinct labels.

/********************************************************************************
This function will read the csv files(training and test dataset) and convert them
into two matrices. classes matrix have 10 columns, one column for each class label. If the label of nth row in data matrix
is, lets say 5 then the value of classes[n][5] = 1.
********************************************************************************/
void read_dataset(char *filename, cv::Mat &data, cv::Mat &classes, int total_samples)
{

	int label;
	float pixelvalue;
	//open the file
	FILE* inputfile = fopen(filename, "r");

	//read each row of the csv file
	for (int row = 0; row < total_samples; row++)
	{
		//for each attribute in the row
		for (int col = 0; col <= ATTRIBUTES; col++)
		{
			//if its the pixel value.
			if (col < ATTRIBUTES){

				fscanf(inputfile, "%f,", &pixelvalue);
				data.at<float>(row, col) = pixelvalue;

			}//if its the label
			else if (col == ATTRIBUTES){
				//make the value of label column in that row as 1.
				fscanf(inputfile, "%i", &label);
				classes.at<float>(row, label) = 1.0;

			}
		}
	}

	fclose(inputfile);

}

/******************************************************************************/

int main(int argc, char** argv)
{
	//matrix to hold the training sample
	cv::Mat training_set(TRAINING_SAMPLES, ATTRIBUTES, CV_32F);
	//matrix to hold the labels of each taining sample
	cv::Mat training_set_classifications(TRAINING_SAMPLES, CLASSES, CV_32F);
	//matric to hold the test samples
	cv::Mat test_set(TEST_SAMPLES, ATTRIBUTES, CV_32F);
	//matrix to hold the test labels.
	cv::Mat test_set_classifications(TEST_SAMPLES, CLASSES, CV_32F);

	//
	cv::Mat classificationResult(1, CLASSES, CV_32F);
	//load the training and test data sets.
	//read_dataset("C:\\Users\\NITHIN\\Desktop\\OCR\\NN\\OCR\\training_dataset.txt", training_set, training_set_classifications, TRAINING_SAMPLES);
	//read_dataset("C:\\Users\\NITHIN\\Desktop\\OCR\\NN\\OCR\\test_dataset.txt", test_set, test_set_classifications, TEST_SAMPLES);
	read_dataset("C:\\Users\\Galib\\OneDrive\\Documents\\nn validation\\val2.txt", training_set, training_set_classifications, TRAINING_SAMPLES);
	read_dataset("C:\\Users\\Galib\\OneDrive\\Documents\\nn validation\\val2.txt", test_set, test_set_classifications, TEST_SAMPLES);


	// define the structure for the neural network (MLP)
	// The neural network has 3 layers.
	// - one input node per attribute in a sample so 8 input nodes
	// - 10 hidden nodes
	// - 2 output node, one for each class.

	cv::Mat layers(3, 1, CV_32S);
	layers.at<int>(0, 0) = ATTRIBUTES;//input layer
	layers.at<int>(1, 0) = 15;//hidden layer
	layers.at<int>(2, 0) = CLASSES;//output layer

	//create the neural network.
	//for more details check http://docs.opencv.org/modules/ml/doc/neural_networks.html
	CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);

	CvANN_MLP_TrainParams params(

		// terminate the training after either 1000
		// iterations or a very small change in the
		// network wieghts below the specified value
		cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001),
		// use backpropogation for training
		CvANN_MLP_TrainParams::BACKPROP,
		// co-efficents for backpropogation training
		// recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
		0.1,
		0.1);

	// train the neural network (using training data)

	printf("\nUsing training dataset\n");
	int iterations = nnetwork.train(training_set, training_set_classifications, cv::Mat(), cv::Mat(), params);
	printf("Training iterations: %i\n\n", iterations);

	// Save the model generated into an xml file.
	CvFileStorage* storage = cvOpenFileStorage("C:\\Users\\Galib\\Desktop\\paramval2.xml", 0, CV_STORAGE_WRITE);
	nnetwork.write(storage, "DigitOCR");
	cvReleaseFileStorage(&storage);

	// Test the generated model with the test samples.
	cv::Mat test_sample;
	//count of correct classifications
	int correct_class = 0;
	//count of wrong classifications
	int wrong_class = 0;

	//classification matrix gives the count of classes to which the samples were classified.
	int classification_matrix[CLASSES][CLASSES] = { {} };

	// for each sample in the test set.
	for (int tsample = 0; tsample < TEST_SAMPLES; tsample++) {

		// extract the sample

		test_sample = test_set.row(tsample);

		//try to predict its class

		nnetwork.predict(test_sample, classificationResult);
		/*The classification result matrix holds weightage  of each class.
		we take the class with the highest weightage as the resultant class */

		// find the class with maximum weightage.
		int maxIndex = 0;
		float value = 0.0f;
		float maxValue = classificationResult.at<float>(0, 0);
		for (int index = 1; index<CLASSES; index++)
		{
			value = classificationResult.at<float>(0, index);
			if (value>maxValue)
			{
				maxValue = value;
				maxIndex = index;

			}
		}

		printf("Testing Sample %i -> class result (digit %d)\n", tsample, maxIndex);

		//Now compare the predicted class to the actural class. if the prediction is correct then\
				            //test_set_classifications[tsample][ maxIndex] should be 1.
		//if the classification is wrong, note that.
		if (test_set_classifications.at<float>(tsample, maxIndex) != 1.0f)
		{
			// if they differ more than floating point error => wrong class

			wrong_class++;

			//find the actual label 'class_index'
			for (int class_index = 0; class_index<CLASSES; class_index++)
			{
				if (test_set_classifications.at<float>(tsample, class_index) == 1.0f)
				{

					classification_matrix[class_index][maxIndex]++;// A class_index sample was wrongly classified as maxindex.
					break;
				}
			}

		}
		else {

			// otherwise correct

			correct_class++;
			classification_matrix[maxIndex][maxIndex]++;
		}
	}

	printf("\nResults on the testing dataset\n"
		"\tCorrect classification: %d (%g%%)\n"
		"\tWrong classifications: %d (%g%%)\n",
		correct_class, (double)correct_class * 100 / TEST_SAMPLES,
		wrong_class, (double)wrong_class * 100 / TEST_SAMPLES);
	cout << "   ";
	for (int i = 0; i < CLASSES; i++)
	{
		cout << i << "\t";
	}
	cout << "\n";
	for (int row = 0; row<CLASSES; row++)
	{
		cout << row << "  ";
		for (int col = 0; col<CLASSES; col++)
		{
			cout << classification_matrix[row][col] << "\t";
		}
		cout << "\n";
	}
	
	double TP = classification_matrix[1][1];
	double FP = classification_matrix[1][0];
	double TN = classification_matrix[0][0];
	double FN = classification_matrix[0][1];
	
	double Sen = ( TP/ (TP + FN));
	double Spec = (TN / (FP + TN));
	double FPrate = (FP / (FP + TN));




	cout << "Sensitivity = " << Sen << "\n";
	cout << "Specifity = " << Spec << "\n";
	cout << "FP rate = " << FPrate << "\n";
	system("pause");
	return 0;

}