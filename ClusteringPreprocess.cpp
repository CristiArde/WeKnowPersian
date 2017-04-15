#include "ClusteringPreprocess.h"

ClusteringPreprocess::ClusteringPreprocess()
{
}


ClusteringPreprocess::~ClusteringPreprocess()
{
}

void ClusteringPreprocess::setFiles(int imgFolder)
{
	//open the folder of Sample Digits
	string path = "Sample Digits Test/" + to_string(imgFolder);
	tinydir_dir test_number_dir;
	tinydir_open(&test_number_dir, path.c_str());

	//iterate inside folder
	while (test_number_dir.has_next)
	{
		//get the image file
		tinydir_file testImageFile;
		tinydir_readfile(&test_number_dir, &testImageFile);

		string testImageFileName = testImageFile.name;

		if (testImageFileName != "." && testImageFileName != "..")
		{
			// prepend full training_files directory
			testImageFileName.insert(0, path + "/");

			// store training filename and label
			testFileNames.push_back(testImageFileName);
			testLabels.push_back(0);
			//testLabels.push_back(currentNumberLabel);
		}
		//go into next directory folder
		tinydir_next(&test_number_dir);
	}
	tinydir_close(&test_number_dir);
}

vector<string> ClusteringPreprocess::getTestFileNames()
{
	return testFileNames;
}
vector<int> ClusteringPreprocess::getTestMatrixLabels()
{
	return testLabels;
}
