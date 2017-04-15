#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "tinydir.h"

// Our Headers
#include "Process.h"
#include "SVMClassifier.h"
#include "MLPClassifier.h"
#include "KNNClassifier.h"
#include "SVMSGDClassifier.h"
#include "ClusteringPreprocess.h"
#include "KMClustering.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	//cout << CV_VERSION;

	/*Process *process = new Process();

	process->setTrainingFiles();
	vector<string> TrainFileNames = process->getTrainFileNames();
	vector<int> MatrixLabels = process->getMatrixLabels();

	process->setTestFiles();
	vector<string> testFNames = process->getTestFileNames();
	vector<int> testLabels = process->getTestMatrixLabels();*/
	/*
	cout << "Starting Support Vector Machine Classification..." << endl;
	SVMClassifier* SVMclasy = new SVMClassifier();
	//SVMclasy->trainSVM(TrainFileNames, MatrixLabels);
	SVMclasy->testSVM(testFNames, testLabels);
	cout << endl << endl;

	// MLP Neural Network
	//cout << "Starting Multilayer Perceptron Neural Network..." << endl;
	//MLPClassifier* MLPclasy = new MLPClassifier();
	//MLPclasy->trainMLP(TrainFileNames, MatrixLabels);
	//MLPclasy->testMLP(testFNames, testLabels);
	MLPclasy->testMLP(testFNames, testLabels);
	cout << endl << endl;
	*/

	//// KNN CLassification
	//cout << "Starting K-Nearest Neighbour Classification..." << endl;
	//KNNClassifier* KNNclasy = new KNNClassifier();
	////KNNclasy->trainKNN(TrainFileNames, MatrixLabels);
	//KNNclasy->testKNN(testFNames, testLabels);

	//cout << "Starting SVMSGD Classification..." << endl;
	//SVMSGDClassifier * SVMSGDclasy = new SVMSGDClassifier();
	////SVMSGDclasy->trainSVMSGD(TrainFileNames, MatrixLabels);
	//SVMSGDclasy->testSVMSGD(testFNames, testLabels);

	// Cluster certain digit
	ClusteringPreprocess *cp = new ClusteringPreprocess();
	cp->setFiles(0);
	vector<string> clusteringFileNames = cp->getTestFileNames();
	vector<int> clusteringLabels = cp->getTestMatrixLabels();

	// K-Means Clustering
	KMClustering* kmeans = new KMClustering(clusteringFileNames, clusteringLabels);
	kmeans->Cluster(3);
	kmeans->CalculateDistance(cv::NORM_L2); // Euclidean distance
	kmeans->CalculateDistance(cv::NORM_HAMMING); // Hamming distance
	kmeans->CalculateDistance(cv::NORM_L1); // Manhattan distance

	system("pause");

	return 0;
}