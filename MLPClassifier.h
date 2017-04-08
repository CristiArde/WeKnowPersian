#pragma once
#include <iostream>
#include <vector>
#include <string>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

#include "ml.h"

// TinyDir
#include "tinydir.h"

using namespace std;
class MLPClassifier
{
public:
	MLPClassifier();
	~MLPClassifier();
	int getPredictedClass(const cv::Mat &);
	void trainMLP(vector<string>, vector<int>);
	void testMLP(vector<string>, vector<int>);

private:
	int imageMatrix;
	cv::Size s_resize;
};