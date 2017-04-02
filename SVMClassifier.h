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

class SVMClassifier
{
public:
	SVMClassifier();
	~SVMClassifier();
	void trainSVM(vector<string>, vector<int>);
private:
	int imageMatrix;
};

