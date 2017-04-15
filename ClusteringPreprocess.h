#pragma once

#include <iostream>
#include <vector>
#include <string>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

// TinyDir
#include "tinydir.h"

using namespace std;

class ClusteringPreprocess
{
public:
	ClusteringPreprocess();
	~ClusteringPreprocess();

	void setFiles(int);

	vector<string> getTestFileNames();
	vector<int> getTestMatrixLabels();

private:
	vector<int> matrixLabels;
	vector<string> trainingFileNames;
	vector<int> testLabels;
	vector<string> testFileNames;

};

