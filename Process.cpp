#include "Process.h"


Process::Process()
{
}


Process::~Process()
{
}


void Process::getTrainingFiles()
{
	//get the images ROOT folder
	tinydir_dir training_directory_root;
	tinydir_open(&training_directory_root, "Sample Digits");

	// go over all folder inside Sample Digits root folder
	while (training_directory_root.has_next)
	{
	/*
		tinydir_file file;
		tinydir_readfile(&training_directory_root, &file);

		string numbersDirName = file.name;

		cout << numbersDirName << endl;
		
	*/
		// get the subfolder of root
		tinydir_file file;
		tinydir_readfile(&training_directory_root, &file);

		// if it is a directory		
		if (file.is_dir)
		{

			string numbersDirName = file.name;

			// skip . / .. / 
			if (numbersDirName != "." && numbersDirName != "..")
			{
				//convert the subfolder name to integer
				int currentNumberLabel = atoi(file.name);

				// prepend full training_files directory
				numbersDirName.insert(0, "Sample Digits/");

				//open the subfolder of Sample Digits
				tinydir_dir training_number_subdir;
				tinydir_open(&training_number_subdir, numbersDirName.c_str());

				//iterate inside subfolders
				while (training_number_subdir.has_next)
				{

					//get the image file
					tinydir_file trainingImageFile;
					tinydir_readfile(&training_number_subdir, &trainingImageFile);

					string trainingImageFileName = trainingImageFile.name;
					
					//	// skip . / .. / 
					if (trainingImageFileName != "." && trainingImageFileName != "..")
					{

						// prepend full training_files directory
						trainingImageFileName.insert(0, numbersDirName + "/");

						// store training filename and label
						trainingFileNames.push_back(trainingImageFileName);
						matrixLabels.push_back(currentNumberLabel);
					}
					//go into next subdirectory folder
					tinydir_next(&training_number_subdir);
				}
				tinydir_close(&training_number_subdir);
			}	
		}
		// get next subfolder 0 , 1 etc
		tinydir_next(&training_directory_root);
	}
	// close directory	

	tinydir_close(&training_directory_root);

}

void Process::trainSVM()
{
	cout << "Im training now";
}

vector<string> Process::getTrainFileNames()
{
	return trainingFileNames;
}
vector<int> Process::getMatrixLabels()
{
	return matrixLabels;
}
