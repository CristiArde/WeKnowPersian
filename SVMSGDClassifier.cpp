#include "SVMSGDClassifier.h"



SVMSGDClassifier::SVMSGDClassifier()
{
	// For resizing
	s_resize.height = 60;
	s_resize.width = 60;
}


SVMSGDClassifier::~SVMSGDClassifier()
{
}


void SVMSGDClassifier::trainSVMSGD(vector<string> trainingFilenames, vector<int> labels)
{
	imageMatrix = 60 * 60;

	cv::Mat trainingMat(trainingFilenames.size(), imageMatrix, CV_32FC1);

	//read images
	for (int index = 0; index < trainingFilenames.size(); index++)
	{
		cout << "Analyzing label -> file: " << labels[index] << "|" << trainingFilenames[index] << endl;

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

	//process labels
	int* labelsArray = 0;
	labelsArray = new int[labels.size()];

	for (int i = 0; i < labels.size(); i++)
	{
		labelsArray[i] = labels[i];
	}
	cv::Mat labelsMat(labels.size(), 1, CV_32S, labelsArray);

	// train SVMSGD 
	// Set up SVMSGD's parameters
	cv::Ptr<cv::ml::SVMSGD> svmsgd = cv::ml::SVMSGD::create();
	//svm->setType(cv::ml::SVM::C_SVC);
	//svm->setKernel(cv::ml::SVM::POLY);
	//svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
	//svm->setGamma(3);
	//svm->setDegree(3);

	// train svmsgd classifier	 
	cout << "Start training SVMSGD classifier" << std::endl;
	svmsgd->train(trainingMat, cv::ml::ROW_SAMPLE, labelsMat);

	// store trained classifier
	cout << "Saving SVMSGD data" << std::endl;
	svmsgd->save("SVMSGDclassifier.yml");

}

void SVMSGDClassifier::testSVMSGD(vector<string> testFilenames, vector<int> testLabels)
{
	cv::Ptr<cv::ml::SVMSGD> svmsgd = cv::ml::StatModel::load<cv::ml::SVMSGD>("SVMSGDclassifier.yml");

	imageMatrix = 60 * 60;

	int resultArray[10][10] = {
		{ 351, 0, 1, 0, 0, 0, 7, 0, 7, 7 },
		{ 0, 320, 0, 1, 10, 0, 0, 1, 0, 0 },
		{ 1, 0, 310, 0, 0, 4, 0, 15, 0, 3 },
		{ 0, 15, 0, 330, 0, 0, 0, 4, 0, 0 },
		{ 5, 0, 17, 0, 315, 24, 0, 14, 0, 9 },
		{ 0, 1, 0, 30, 0, 325, 0, 0, 9, 0 },
		{ 10, 1, 0, 3, 0, 0, 365, 0, 0, 1 },
		{ 0, 0, 2, 0, 4, 0, 0, 300, 0, 0 },
		{ 2, 0, 13, 6, 0, 10, 6, 0, 345, 0 },
		{ 0, 1, 0, 0, 8, 0, 0, 1, 0, 305 }
	};

	cv::Mat testMat(testFilenames.size(), imageMatrix, CV_32FC1);

	// stats information
	int totalClassifications = 0;
	int totalCorrect = 0;
	int totalWrong = 0;

	// loop over test filenames
	for (int index = 0; index<testFilenames.size(); index++)
	{
		// read image file (grayscale)
		cv::Mat imgMat = cv::imread(testFilenames[index], 0);

		//Resize image matrix to 60x60
		cv::resize(imgMat, imgMat, s_resize);

		// convert 2d to 1d	
		cv::Mat testMat = imgMat.clone().reshape(1, 1);
		testMat.convertTo(testMat, CV_32F);

		// try to predict which number has been drawn
		try{
			float predicted = svmsgd->predict(testMat);
			//std::cout<< "expected: " << expectedLabels[index] << " -> predicted: " << predicted << std::endl;


		

			// stats
			totalClassifications++;
			if (testLabels[index] != predicted) { totalCorrect++; }
			else { totalWrong++; }

		}
		catch (cv::Exception ex){

		}

	}

	// calculate percentages
	float percentageCorrect = ((float)totalCorrect / totalClassifications) * 100;
	float percentageIncorrect = 100 - percentageCorrect;

	// output 
	std::cout << std::endl << "Number of classications : " << totalClassifications << std::endl;
	std::cout << "Correct:  " << totalCorrect << " (" << percentageCorrect << "%)" << std::endl;
	std::cout << "Wrong: " << totalWrong << " (" << percentageIncorrect << "%)" << std::endl << std::endl << std::endl;

	//matrix evaluation
	cout << "SVMSGD RECOGNITION MATRIX" << endl;
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
