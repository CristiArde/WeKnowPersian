#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2\opencv.hpp>
#include <iostream>
#include "Process.h"
using namespace cv;
using namespace std;
int main(int argc, char** argv)
{
	cout << CV_VERSION;
	system("pause");
	

	Process *process = new Process();

	process->getTrainingFiles();
	
	return 0;

}