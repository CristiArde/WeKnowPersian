#include "MLPClassifier.h";

MLPClassifier::MLPClassifier()
{
	// For resizing
	s_resize.height = 60;
	s_resize.width = 60;

	imageMatrix = 60 * 60;
}

MLPClassifier::~MLPClassifier()
{
}

void MLPClassifier::trainMLP(vector<string> trainingFilenames, vector<int> trainingLabels)
{
	trainingMat = cv::Mat(trainingFilenames.size(), imageMatrix, CV_32FC1);
	cv::Mat classificationResult(1, 10, CV_32FC1);

	cout << "Analyzing features -> files..." << endl;
	//read images
	for (int index = 0; index < trainingFilenames.size(); index++)
	{
		//cout << "Analyzing label -> file: " << trainingLabels[index] << "|" << trainingFilenames[index] << endl;
		cv::Mat imgMat = cv::imread(trainingFilenames[index], 0);

		// Resize image matrix to 60x60
		cv::resize(imgMat, imgMat, s_resize);

		int column = 0;
		for (int i = 0; i < imgMat.rows; i++)
		{
			for (int j = 0; j < imgMat.cols; j++)
			{
				trainingMat.at<float>(index, column++) = imgMat.at<uchar>(i, j);
			}
		}
	}

	int layerSizes[] = { trainingMat.cols, 50, 10 };
	cv::Mat layers = cv::Mat(1, 3, CV_32SC1);
	layers.at<int>(0, 0) = layerSizes[0];	// inputs
	layers.at<int>(0, 1) = layerSizes[1];	// hidden layers
	layers.at<int>(0, 2) = layerSizes[2];	// outputs

	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::ANN_MLP::create();

	mlp->setLayerSizes(layers);

	//create the network using a sigmoid function with alpha and beta parameters 0.6 and 1 specified respectively
	mlp->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 0.6, 1);

	// terminate the training after either 1000 iterations or a very small change in the network wieghts below the specified value
	mlp->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 300, 0.0001));

	// use backpropogation for training
	mlp->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.0001);
	
	cv::Mat trainingClass = cv::Mat::zeros(trainingMat.rows, 10, CV_32FC1);
	for (int i = 0; i < trainingMat.rows; i++)
	{
		trainingClass.at<float>(i, trainingLabels.at(i)) = 1.f;
	}

	//cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(trainingMat, cv::ml::ROW_SAMPLE, trainingClass);

	// Train
	cout << "Training MLP. Grab a coffee, this may take a while..." << std::endl;
	mlp->train(trainingMat, cv::ml::ROW_SAMPLE, trainingClass);
	cout << "Saving MLP data" << std::endl;
	mlp->save("MLPClassifier.yml");
}

void MLPClassifier::testMLP(vector<string> testFilenames, vector<int> testLabels)
{
	cv::Mat testMat(testFilenames.size(), imageMatrix, CV_32FC1);
	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::StatModel::load<cv::ml::ANN_MLP>("MLPClassifier.yml");
	
	//read images
	for (int index = 0; index < testFilenames.size(); index++)
	{
		cv::Mat imgMat = cv::imread(testFilenames[index], 0);

		// Resize image matrix to 60x60
		cv::resize(imgMat, imgMat, s_resize);

		int column = 0;
		for (int i = 0; i < imgMat.rows; i++)
		{
			for (int j = 0; j < imgMat.cols; j++)
			{
				testMat.at<float>(index, column++) = imgMat.at<uchar>(i, j);
			}
		}
	}

	// Will hold test results
	cv::Mat confusion(10, 10, CV_32S, cv::Scalar(0));
	
	// Run tests on validation set
	for (int i = 0; i < testMat.rows; i++)
	{
		int pred = mlp->predict(testMat.row(i), cv::noArray());
		int truth = testLabels.at(i);
		confusion.at<int>(pred, truth)++;
	}

	// Get the diagonal values of the matrix (correct values)
	cv::Mat correct = confusion.diag();
	float accuracy = (sum(correct)[0] / sum(confusion)[0]) * 100;
	cout << "Correct:  " << sum(correct)[0] << " (" << accuracy << "%)" << std::endl;
	//cout << "confusion:\n" << confusion << endl;

	// Print pretty confusion matrix
	cout << "MLP CONFUSION MATRIX" << endl;
	cout << setw(5) << "0" << setw(8) << "1" << setw(8) << "2" << setw(8) << "3" << setw(8) << "4" << setw(8) << "5" << setw(8) << "6" << setw(8) << "7" << setw(8) << "8" << setw(8) << "9" << endl;
	cout << "_________________________________________________________________________________" << endl;
	for (int i = 0; i < 10; i++)
	{
	cout << i << "|" << setw(2);
	for (int j = 0; j < 10; j++)
	{
	cout << setw(3) << confusion.at<int>(i, j) << "   | ";
	}
	cout << endl;
	}

	//plot_binary(testMat, correct, "Predictions MLP");
}

// plot data and class
void MLPClassifier::plot_binary(cv::Mat& data, cv::Mat& classes, string name) {
	int size = 200;
	cv::Mat plot(size, size, CV_8UC3);
	plot.setTo(cv::Scalar(255.0, 255.0, 255.0));
	for (int i = 0; i < data.rows; i++) {

		float x = data.at<float>(i, 0) * size;
		float y = data.at<float>(i, 1) * size;

		if (classes.at<float>(i, 0) > 0) {
			cv::circle(plot, cv::Point(x, y), 2, CV_RGB(255, 0, 0), 1);
		}
		else {
			cv::circle(plot, cv::Point(x, y), 2, CV_RGB(0, 255, 0), 1);
		}
	}
	cv::imshow(name, plot);
}
