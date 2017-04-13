#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <iomanip> 

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

// TinyDir
//#include "tinydir.h"

using namespace std;

class KNNClassifier
{
public:
	KNNClassifier();
	~KNNClassifier();
	void trainKNN(vector<string>, vector<int>);
	void testKNN(vector<string>, vector<int>);
private:
	int imageMatrix;
	cv::Size s_resize;
};

