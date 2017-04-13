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
	cv::Mat trainingMat(trainingFilenames.size(), imageMatrix, CV_32FC1);
	cv::Mat classificationResult(1, 10, CV_32FC1);

	cout << "Analyzing labels -> files..." << endl;
	//read images
	for (int index = 0; index < trainingFilenames.size(); index++)
	{
		cout << "Analyzing label -> file: " << trainingLabels[index] << "|" << trainingFilenames[index] << endl;
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

	int layerSizes[] = { trainingMat.cols, 10, 10 };
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
	// stats information
	int totalClassifications = 0;
	int totalCorrect = 0;
	int totalWrong = 0;

	cv::Mat testMat(testFilenames.size(), imageMatrix, CV_32FC1);
	cv::Ptr<cv::ml::ANN_MLP> mlp = cv::ml::StatModel::load<cv::ml::ANN_MLP>("MLPClassifier.yml");
	
	int resultArray[10][10] = {
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
	};


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


		int number = (int)floor(pred + 0.5);
		switch (number)
		{
		case 0:
			resultArray[testLabels[i]][0]++;
			break;

		case 1:
			resultArray[testLabels[i]][1]++;
			break;

		case 2:
			resultArray[testLabels[i]][2]++;
			break;
		case 3:
			resultArray[testLabels[i]][3]++;
			break;

		case 4:
			resultArray[testLabels[i]][4]++;
			break;

		case 5:
			resultArray[testLabels[i]][5]++;
			break;
		case 6:
			resultArray[testLabels[i]][6]++;

			break;
		case 7:
			resultArray[testLabels[i]][7]++;
			break;
		case 8:
			resultArray[testLabels[i]][8]++;
			break;
		case 9:
			resultArray[testLabels[i]][9]++;
			break;
		}
		totalClassifications++;
		if (testLabels[i] == pred) { totalCorrect++; }
		else { totalWrong++; }

	}

	// Get the diagonal values of the matrix (correct values)
	cv::Mat correct = confusion.diag();
	float accuracy = (sum(correct)[0] / sum(confusion)[0]) * 100;
	// calculate percentages
	float percentageCorrect = ((float)totalCorrect / totalClassifications) * 100;
	float percentageIncorrect = 100 - percentageCorrect;
	std::cout << std::endl << "Number of classications : " << totalClassifications << std::endl;
	std::cout << "Correct:  " << totalCorrect << " (" << percentageCorrect << "%)" << std::endl;
	std::cout << "Wrong: " << totalWrong << " (" << percentageIncorrect << "%)" << std::endl << std::endl << std::endl;
	//cerr << "confusion:\n" << confusion << endl;


	//matrix evaluation
	cout << "MLP RECOGNITION MATRIX" << endl;
	cout << setw(5) << "0" << setw(8) << "1" << setw(8) << "2" << setw(8) << "3" << setw(8) << "4" << setw(8) << "5" << setw(8) << "6" << setw(8) << "7" << setw(8) << "8" << setw(8) << "9" << endl;
	cout << "_________________________________________________________________________________" << endl;
	for (int i = 0; i < 10; i++)
	{
		cout << i << "|" << setw(2);
		for (int j = 0; j < 10; j++)
		{
			cout << setw(3) << resultArray[i][j] << "   | ";
		}
		cout << endl;
	}
}
