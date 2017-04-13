#include "SVMClassifier.h"



SVMClassifier::SVMClassifier()
{
	// For resizing
	s_resize.height = 60;
	s_resize.width = 60;
}


SVMClassifier::~SVMClassifier()
{
}


void SVMClassifier::trainSVM(vector<string> trainingFilenames, vector<int> labels)
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

	// train SVM 
	// Set up SVM's parameters
	cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
	svm->setType(cv::ml::SVM::C_SVC);
	svm->setKernel(cv::ml::SVM::POLY);
	svm->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 100, 1e-6));
	svm->setGamma(3);
	svm->setDegree(3);

	// train svm classifier	 
	cout << "Start training SVM classifier" << std::endl;
	svm->train(trainingMat, cv::ml::ROW_SAMPLE, labelsMat);

	// store trained classifier
	cout << "Saving SVM data" << std::endl;
	svm->save("classifier.yml");

}

void SVMClassifier::testSVM(vector<string> testFilenames, vector<int> testLabels)
{
	cv::Ptr<cv::ml::SVM> svm = cv::ml::StatModel::load<cv::ml::SVM>("classifier.yml");

	imageMatrix = 60 * 60;

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
		cv::Mat testMat = imgMat.clone().reshape(1,1);
		testMat.convertTo(testMat, CV_32F);

		// try to predict which number has been drawn
		try{
			float predicted = svm->predict(testMat);
			//std::cout<< "expected: " << expectedLabels[index] << " -> predicted: " << predicted << std::endl;


			int number = (int)floor(predicted + 0.5);
			switch (number)
			{
			case 0:
				resultArray[testLabels[index]][0]++;
				break;

			case 1:
				resultArray[testLabels[index]][1]++;
				break;

			case 2:
				resultArray[testLabels[index]][2]++;
				break;
			case 3:
				resultArray[testLabels[index]][3]++;
				break;

			case 4:
				resultArray[testLabels[index]][4]++;
				break;

			case 5:
				resultArray[testLabels[index]][5]++;
				break;
			case 6:
				resultArray[testLabels[index]][6]++;
				break;
			case 7:
				resultArray[testLabels[index]][7]++;
				break;
			case 8:
				resultArray[testLabels[index]][8]++;
				break;
			case 9:
				resultArray[testLabels[index]][9]++;
				break;
			}

			// stats
			totalClassifications++;
			if (testLabels[index] == predicted) { totalCorrect++; }
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
	cout << "SVM RECOGNITION MATRIX" << endl;
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
