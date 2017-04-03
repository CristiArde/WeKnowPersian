#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include "Process.h"
#include "SVMClassifier.h"

using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	cout << CV_VERSION;


	Process *process = new Process();

	process->setTrainingFiles();
	vector<string> TrainFileNames = process->getTrainFileNames();
	vector<int> MatrixLabels = process->getMatrixLabels();

	process->setTestFiles();
	vector<string> testFNames = process->getTestFileNames();
	vector<int> testLabels = process->getTestMatrixLabels();

	
	SVMClassifier* SVMclasy = new SVMClassifier();
	//SVMclasy->trainSVM(TrainFileNames, MatrixLabels);


	SVMclasy->testSVM(testFNames, testLabels);
	
	system("pause");

	return 0;

}