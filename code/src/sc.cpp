/**
* Manan Prajapati 
* COMP 6651: Project 2 Seam Carving
* REFERENCES: https://docs.opencv.org/3.4.3/da/d85/tutorial_js_gradients.html
* encs REMOTE VERSION
*/
/*
	USE OPENCV highgui, imgproc, core library
*/
#include <cmath>
#include <limits>
#include <vector>

#include "sc.h"

using namespace cv;
using namespace std;
enum Seam_Location { VERTICAL, HORIZONTAL };
Mat kernelX = (Mat_<double>(3,3) <<1,1,1,0,0,0,-1,-1,-1);
Mat kernelY = (Mat_<double>(3,3) <<-1,0,1,-1,0,1,-1,0,1);
bool printing = false;
//calculating energy map ----done
void computeEnergyMap(Mat& in_image, Mat& gradient_image) {
	Mat temp_in_image;
	Mat gradient_x, gradient_y;
	Mat gray_image;
	//prewitt
	cvtColor(in_image, temp_in_image, CV_RGB2GRAY);
	filter2D(temp_in_image, gradient_x, CV_32F , kernelX, Point(-1,-1));
	filter2D(temp_in_image, gradient_y, CV_32F , kernelY, Point(-1,-1));
	convertScaleAbs(gradient_x, gradient_x);
	convertScaleAbs( gradient_y, gradient_y );
	addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0, gradient_image);
	//end of

	//laplacian function ---
	//GaussianBlur(in_image, temp_in_image, Size(3, 3), 0, 0, BORDER_DEFAULT);
	//cvtColor(temp_in_image, temp_in_image, CV_RGB2GRAY);
	// You can try more different parameters
	//Laplacian(temp_in_image, gradient_image, CV_16S, 3, 1, 0, BORDER_DEFAULT);
	//convertScaleAbs(gradient_image, gradient_image);
	// end of laplacian 
	//reducing noise
	/*GaussianBlur(in_image, temp_in_image, Size(3, 3), 0, 0, BORDER_DEFAULT);

	cvtColor(temp_in_image, gray_image, COLOR_BGR2GRAY);
	
	Sobel(gray_image, gradient_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	//Scharr(gray_image, gradient_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	//Scharr(gray_image, gradient_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
	Sobel(gray_image, gradient_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(gradient_x, gradient_x);
	convertScaleAbs(gradient_y, gradient_y);
	
	addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0, gradient_image);
	*/
	//gradient_image.convertTo(gradient_image, CV_64F, 1.0/255.0);
	if(printing)
		imshow("gradient image", gradient_image);
	
}

vector<int> dynamicProgramming(Mat& energyImage, char direction /*Seam_Location seam_type*/) {

	vector<int> indexPath(energyImage.rows);
	int minimum = numeric_limits<int>::max();

	if (direction == 'V' /*seam_type == VERTICAL*/) {
		indexPath.resize(energyImage.rows);
		double** energyValues = new double*[energyImage.rows];
		for (int i = 0; i < energyImage.rows; ++i) {
			energyValues[i] = new double[energyImage.cols];
		}
			
	//	int pixelValue1, pixelValue2, pixelValue3;

		//filling the table with all pixel energy values -- add one more rows and column with infinity values.
		// -- change the pixel energy from int to double for acuracy.
		for (int i = 0; i < energyImage.rows; i++)
			for (int j = 0; j < energyImage.cols; j++)
				energyValues[i][j] = (double)energyImage.at<uchar>(i, j);

		/*	updating the table with minimum cost values --  remove extra conditions
			pixel1 = energyValues[i-1] [max((j-1),0)]
			pixel2 = energyValues[i-1] [j]
			pixel3 = energyValues[i-1] [min((j+1),j-1)]
		*/
		for (int i = 1; i < energyImage.rows; i++) {
			for (int j = 0; j < energyImage.cols; j++) {

				/* changed
				energyValues[i][j] = energyValues[i][j] + min(energyValues[i-1] [max((j-1),0)],
														min(energyValues[i-1] [j],energyValues[i-1] [min((j+1),j-1)]));
				*/
				
				if (j == 0) {
					energyValues[i][j] = energyValues[i][j] + min(energyValues[i - 1][j], energyValues[i - 1][j + 1]);
				}
				else if (j == energyImage.cols - 1) {
					energyValues[i][j] = energyValues[i][j] + min(energyValues[i - 1][j - 1], energyValues[i - 1][j]);
				}
				else {
					double pixelValue1 = energyValues[i - 1][j - 1];
					double pixelValue2 = energyValues[i - 1][j];
					double pixelValue3 = energyValues[i - 1][j + 1];
					energyValues[i][j] = energyValues[i][j] + min(pixelValue1, min(pixelValue2, pixelValue3));
				}
			}
		}

		//finding the minimum cost value index from last row (bottom) and putting the minimum cost index in vector.
		for (int j = 0; j < energyImage.cols; j++) {
			if (energyValues[energyImage.rows - 1][j] < minimum) {
				minimum = energyValues[energyImage.rows - 1][j];
				indexPath.at(energyImage.rows - 1) = j;
			}
		}

		// backtracking finding the minimum cost path or seam , storing it into indexPath vector
		// starting from second last row to first row.
		// change -- way to implement
		for (int i = energyImage.rows - 2; i >= 0; i--) { // reducing width

			if (indexPath.at(i + 1) == 0) {
				if (energyValues[i][indexPath.at(i + 1)] < energyValues[i][indexPath.at(i + 1) + 1])
					indexPath.at(i) = indexPath.at(i + 1);
				else
					indexPath.at(i) = indexPath.at(i + 1) + 1;
			}

			else if (indexPath.at(i + 1) == (energyImage.cols - 1)) {
				if (energyValues[i][indexPath.at(i + 1)] < energyValues[i][indexPath.at(i + 1) - 1])
					indexPath.at(i) = indexPath.at(i + 1);
				else
					indexPath.at(i) = indexPath.at(i + 1) - 1;
			}

			else {
				minimum = min(energyValues[i][indexPath.at(i + 1)], min(energyValues[i][indexPath.at(i + 1) - 1], energyValues[i][indexPath.at(i + 1) + 1]));
				if (minimum == energyValues[i][indexPath.at(i + 1)])
					indexPath.at(i) = indexPath.at(i + 1);
				else if (minimum == energyValues[i][indexPath.at(i + 1) - 1])
					indexPath.at(i) = indexPath.at(i + 1) - 1;
				else if (minimum == energyValues[i][indexPath.at(i + 1) + 1])
					indexPath.at(i) = indexPath.at(i + 1) + 1;
			}
		}
	}

	else if (direction == 'H' /*seam_type == HORIZONTAL*/) {

		indexPath.resize(energyImage.cols);
		//int energyValues[energyImage.rows][energyImage.cols];
		double** energyValues = new double*[energyImage.rows];
		for (int i = 0; i < energyImage.rows; ++i) {
			energyValues[i] = new double[energyImage.cols];
		}
//		int pixelValue1, pixelValue2, pixelValue3;

		for (int i = 0; i < energyImage.rows; i++)
			for (int j = 0; j < energyImage.cols; j++)
				energyValues[i][j] = (double)energyImage.at<uchar>(i, j);

		/*	updating the table with minimum cost values --  remove extra conditions
			pixel1 = energyValues[max((i-1),0)][j-1]
			pixel2 = energyValues[i] [j-1]
			pixel3 = energyValues [min((i+1),i-1)] [j-1]
		*/
		for (int j = 1; j < energyImage.cols; j++) {
			for (int i = 0; i < energyImage.rows; i++) {

				if (i == 0) {
					energyValues[i][j] = energyValues[i][j] + min(energyValues[i][j - 1], energyValues[i + 1][j - 1]);
				}
				else if (i == energyImage.rows - 1) {
					energyValues[i][j] = energyValues[i][j] + min(energyValues[i - 1][j - 1], energyValues[i][j - 1]);
				}
				else {
					double pixelValue1 = energyValues[i - 1][j - 1];
					double pixelValue2 = energyValues[i][j - 1];
					double pixelValue3 = energyValues[i + 1][j - 1];
					energyValues[i][j] = energyValues[i][j] + min(pixelValue1, min(pixelValue2, pixelValue3));
				}
			}
		}

		for (int j = 0; j < energyImage.rows; j++) {
			if (energyValues[j][energyImage.cols - 1] < minimum) {
				minimum = energyValues[j][energyImage.cols - 1];
				indexPath.at(energyImage.cols - 1) = j;
			}
		}

		for (int i = energyImage.cols - 2; i >= 0; i--) {

			if (indexPath.at(i + 1) == 0) {
				if (energyValues[indexPath.at(i + 1)][i] < energyValues[indexPath.at(i + 1) + 1][i])
					indexPath.at(i) = indexPath.at(i + 1);
				else
					indexPath.at(i) = indexPath.at(i + 1) + 1;
			}

			else if (indexPath.at(i + 1) == (energyImage.rows - 1)) {
				if (energyValues[indexPath.at(i + 1)][i] < energyValues[indexPath.at(i + 1) - 1][i])
					indexPath.at(i) = indexPath.at(i + 1);
				else
					indexPath.at(i) = indexPath.at(i + 1) - 1;
			}

			else {
				minimum = min(energyValues[indexPath.at(i + 1)][i], min(energyValues[indexPath.at(i + 1) - 1][i], energyValues[indexPath.at(i + 1) + 1][i]));
				if (minimum == energyValues[indexPath.at(i + 1)][i])
					indexPath.at(i) = indexPath.at(i + 1);
				else if (minimum == energyValues[indexPath.at(i + 1) - 1][i])
					indexPath.at(i) = indexPath.at(i + 1) - 1;
				else if (minimum == energyValues[indexPath.at(i + 1) + 1][i])
					indexPath.at(i) = indexPath.at(i + 1) + 1;
			}
		}
	}

	return indexPath;
}

//validating the image ---done new
bool sanityCheck(int new_width, int new_height, int orignal_width, int orignal_height) {

	if (new_width > orignal_width) {
		cout << "Invalid request!!! new_width has to be smaller than the current size!" << endl;
		return false;
	}
	if (new_height > orignal_height) {
		cout << "Invalid request!!! ne_height has to be smaller than the current size!" << endl;
		return false;
	}

	if (new_width <= 0) {
		cout << "Invalid request!!! new_width has to be positive!" << endl;
		return false;
	}

	if (new_height <= 0) {
		cout << "Invalid request!!! new_height has to be positive!" << endl;
		return false;
	}
	return true;
}

//default function ---done
bool seam_carving(Mat& in_image, int new_width, int new_height, Mat& out_image) {
	cout << "started" << endl;
	
	return seam_carving_trivial(in_image, new_width, new_height, out_image);
}


// seam carves by removing trivial seams
bool seam_carving_trivial(Mat& in_image, int new_width, int new_height, Mat& out_image) {

	Mat iimage = in_image.clone();
	Mat oimage = in_image.clone();
	Mat gradientImage;

	//1st step: Compute Energy Map
	printing = true;
	computeEnergyMap(in_image, gradientImage);
	printing = false;

	while (iimage.rows != new_height) {
		// horizontal seam if needed
		if (iimage.rows > new_height) {
			reduce_horizontal_seam_trivial(iimage, oimage, gradientImage);
			iimage = oimage.clone();
			computeEnergyMap(iimage, gradientImage);
		}
	}
	while (iimage.cols != new_width) {
		if (iimage.cols > new_width) {
			reduce_vertical_seam_trivial(iimage, oimage, gradientImage);
			iimage = oimage.clone();
			computeEnergyMap(iimage, gradientImage);
		}
	}

	// out_image = gradientImage.clone();
	out_image = oimage.clone();
	return true;
}

// horizontal trivial seam is a seam through the center of the image
bool reduce_horizontal_seam_trivial(Mat& in_image, Mat& out_image, Mat& energyImage) {

	// retrieve the dimensions of the new image
	int rows = in_image.rows - 1;
	int cols = in_image.cols;

	// create an image slighly smaller
	out_image = Mat(rows, cols, CV_8UC3);

	vector<int> rowIndexPerColumn;
	rowIndexPerColumn = dynamicProgramming(energyImage, 'H');

	//change way to implement -- use opencv imgproc highgui library
	for (int j = 0; j < cols; ++j) {
		for (int i = 0; i < rowIndexPerColumn.at(j); ++i) {

			Vec3b pixel = in_image.at<Vec3b>(i, j);
			out_image.at<Vec3b>(i, j) = pixel;
		}

		for (int i = rowIndexPerColumn.at(j); i < rows; ++i) {
			Vec3b pixel = in_image.at<Vec3b>(i + 1, j);
			out_image.at<Vec3b>(i, j) = pixel;
		}
	}

	return true;
}

// vertical trivial seam is a seam through the center of the image
bool reduce_vertical_seam_trivial(Mat& in_image, Mat& out_image, Mat& energyImage) {
	// retrieve the dimensions of the new image
	int rows = in_image.rows;
	int cols = in_image.cols - 1;

	// create an image slighly smaller
	out_image = Mat(rows, cols, CV_8UC3);

	vector<int> columnIndexPerRow;
	columnIndexPerRow = dynamicProgramming(energyImage, 'V');

	//change way to implement -- use opencv imgproc highgui library
	
		for(int i = 0; i < out_image.rows; i++ ) {
		for (int j = 0; j < out_image.cols; j++){
			if (j >= columnIndexPerRow.at(i))
				out_image.at<Vec3b>(i,j) = in_image.at<Vec3b>(i,j+1);
			else
				out_image.at<Vec3b>(i,j) = in_image.at<Vec3b>(i,j);
		}
	}
	

	

	/*for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < columnIndexPerRow.at(i); j++) {

			Vec3b pixel = in_image.at<Vec3b>(i, j);
			out_image.at<Vec3b>(i, j) = pixel;
		}

		for (int j = columnIndexPerRow.at(i); j < cols; j++) {

			Vec3b pixel = in_image.at<Vec3b>(i, j + 1);
			out_image.at<Vec3b>(i, j) = pixel;
		}
	}*/

	return true;
}
