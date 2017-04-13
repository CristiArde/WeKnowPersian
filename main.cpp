#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include "Process.h"
#include "SVMClassifier.h"
#include "KNNClassifier.h"
#include "MLPClassifier.h"

using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	//cout << CV_VERSION;


	Process *process = new Process();

	process->setTrainingFiles();
	vector<string> TrainFileNames = process->getTrainFileNames();
	vector<int> MatrixLabels = process->getMatrixLabels();

	process->setTestFiles();
	vector<string> testFNames = process->getTestFileNames();
	vector<int> testLabels = process->getTestMatrixLabels();

	//cout << "Starting Support Vector Machine Classification..." << endl;
	//SVMClassifier* SVMclasy = new SVMClassifier();
	//SVMclasy->trainSVM(TrainFileNames, MatrixLabels);
	//SVMclasy->testSVM(testFNames, testLabels);
	

	// MLP Neural Network
	cout << "Starting Multilayer Perceptron Neural Network..." << endl;
	MLPClassifier* MLPclasy = new MLPClassifier();
	//MLPclasy->trainMLP(TrainFileNames, MatrixLabels);
	MLPclasy->testMLP(testFNames, testLabels);


	// KNN CLassification
	//cout << "Starting K-Nearest Neighbour Classification..." << endl;
	//KNNClassifier* KNNclasy = new KNNClassifier();
	//KNNclasy->trainKNN(TrainFileNames, MatrixLabels);
	//KNNclasy->testKNN(testFNames, testLabels);


	system("pause");

	return 0;

}