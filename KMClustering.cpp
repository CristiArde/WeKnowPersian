#include "KMClustering.h"

KMClustering::KMClustering(vector<string> trainingFilenames, vector<int> trainingLabels)
{
	this->trainingFilenames = trainingFilenames;
	this->trainingLabels = trainingLabels;

	FeatureExtraction();
}


KMClustering::~KMClustering()
{
}

void KMClustering::FeatureExtraction()
{
	// For resizing
	int imageMatrix = 60 * 60;

	cv::Size s_resize;
	s_resize.height = 60;
	s_resize.width = 60;

	dataMat = cv::Mat(trainingFilenames.size(), imageMatrix, CV_32F);

	// Read images
	for (int index = 0; index < trainingFilenames.size(); index++)
	{
		cv::Mat imgMat = cv::imread(trainingFilenames[index], 0);

		// Resize image matrix to 60x60
		cv::resize(imgMat, imgMat, s_resize);

		int column = 0;

		for (int i = 0; i < imgMat.rows; i++)
		{
			for (int j = 0; j < imgMat.cols; j++)
			{
				dataMat.at<float>(index, column++) = imgMat.at<uchar>(i, j);
			}
		}
	}
}

void KMClustering::Cluster(const int k, const int attempts)
{
	cv::kmeans(dataMat, k, outputArray, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), attempts, cv::KMEANS_RANDOM_CENTERS, centers);
}

tuple<vector<vector<int>*>, vector<FeatureDistance*>> KMClustering::CalculateDistance(cv::NormTypes normType)
{
	if (outputArray.empty() || centers.empty())
	{
		this->Cluster();
	}

	vector<vector<int>*> clusters;
	vector<FeatureDistance*> centroids;

	// Get Clusters
	for (int i = 0; i < centers.size().height; i++)
	{
		clusters.push_back(new vector<int>);

		centroids.push_back(new FeatureDistance({ 0, 0, normType }));
	}

	for (int row = 0; row < outputArray.size().height; row++)
	{
		clusters[outputArray.at<int>(row, 0)]->push_back(row);
	}

	// Given true centroids find closest feature;
	for (int centroid = 0; centroid < centers.size().height; centroid++)
	{
		for (int i = 0; i < clusters[centroid]->size(); i++)
		{
			//cout << "center: " << centroid << endl;
			//cout << "trainingDataMat row: " << (*clusters[centroid])[i] << endl;
			//cout << "centroid index: " << centroids[centroid]->imgIndex << endl;

			dataMat.convertTo(dataMat, CV_8U);
			centers.convertTo(centers, CV_8U);

			double n = cv::norm(dataMat.row((*clusters[centroid])[i]), centers.row(centroid), normType);

			if (i == 0 || centroids[centroid]->norm > n)
			{
				centroids[centroid]->imgIndex = i;
				centroids[centroid]->norm = n;
			}
		}
	}

	return make_tuple(clusters, centroids);

	//cout << "Centroids Size: " << centroids.size() << endl;
	//cout << "Centroids: \n" << centroids[0] << endl;


	//cout << "Labels: \n" << outputArray << endl;

	//cout << "Training Size: " << trainingDataMat.size() << endl;
	//cout << "Data: \n" << trainingDataMat << endl;

	//cout << "Centers Size: " << centers.size() << endl;
	//cout << "Centers: \n" << centers << endl;

	//outputArray.size();

	//cout << "Carp: "<< outputArray.at<int>(0, 0);

	//TCHAR *file = TEXT("KMClustering");

	//CreateDirectory(file, NULL);

	//delete[] clusters;
}
