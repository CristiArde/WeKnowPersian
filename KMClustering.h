#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <windows.h>

// OpenCV
//#include <opencv2/core.hpp>
//#include <opencv2/imgproc.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/ml.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

// TinyDir
#include "tinydir.h"

using namespace std;
//using namespace cv;

struct FeatureDistance
{
	int imgIndex;
	double norm;
	cv::NormTypes normType;
};

class KMClustering
{
public:
	vector<string> trainingFilenames;
	vector<int> trainingLabels;

	KMClustering(vector<string>, vector<int>);
	~KMClustering();
	void Cluster(const int numClusters = 3, const int attempts = 500);
	tuple<vector<vector<int>*>, vector<FeatureDistance*>> CalculateDistance(cv::NormTypes normType = cv::NormTypes::NORM_L2);

private:
	cv::Mat dataMat;
	cv::Mat outputArray;
	cv::Mat centers;

	void FeatureExtraction();
};

