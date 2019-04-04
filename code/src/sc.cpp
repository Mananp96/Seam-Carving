/**
* Manan Prajapati 
* COMP 6651: Project 2 Seam Carving
* REFERENCES: https://docs.opencv.org/3.4.3/da/d85/tutorial_js_gradients.html
* encs REMOTE VERSION
*/

#include <cmath>
#include <limits>
#include <vector>

#include "sc.h"
#define MINIMUM_OF(value1, value2) ((value1 < value2) ? value1 : value2)
#define MAXIMUM_OF(value1, value2) ((value1 > value2) ? value1 : value2) 
#define integer_max 2147483647
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
	//Prewitt function
	cvtColor(in_image, temp_in_image, CV_RGB2GRAY);
	filter2D(temp_in_image, gradient_x, CV_32F , kernelX, Point(-1,-1));
	filter2D(temp_in_image, gradient_y, CV_32F , kernelY, Point(-1,-1));
	convertScaleAbs(gradient_x, gradient_x);
	convertScaleAbs( gradient_y, gradient_y );
	addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0, gradient_image);
	//end of Prewitt

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


int find_minimumPoint(int imageSizeCol, int imageSizeRow, int minimumValue, vector<int>& indexPath, double** energyValues, int minValueIndex, char direction){
	//finding the minimum cost value index from last row and column and putting the minimum cost index in vector.
	if(direction == 'V'){
		for (int j = 0; j < imageSizeCol; j++) {
			if (energyValues[imageSizeRow - 1][j] < minimumValue) {
				minimumValue = energyValues[imageSizeRow - 1][j];
				indexPath.at(imageSizeRow - 1) = j;
				minValueIndex = j;
			}
		}
	}else{
		for (int j = 0; j < imageSizeRow; j++) {
			if (energyValues[j][imageSizeCol - 1]< minimumValue) {
				minimumValue = energyValues[j][imageSizeCol - 1];
				indexPath.at(imageSizeCol - 1) = j;
				minValueIndex = j;
			}
		}
	}
	return minValueIndex;
}

vector<int> dynamicProgramming(Mat& energyImage, char direction /*Seam_Location seam_type*/) {

	vector<int> indexPath;
	int minimumValue = 21400000;
	int pathIndex = 0;
	int minValueIndex = 21477777;
	//assign pixel value to table
	double** energyValues = new double*[energyImage.rows];
	for (int i = 0; i < energyImage.rows; ++i) {
		energyValues[i] = new double[energyImage.cols];
	}
	
	for (int i = 0; i < energyImage.rows; i++)
			for (int j = 0; j < energyImage.cols; j++)
				energyValues[i][j] = (double)energyImage.at<uchar>(i, j);
	
	/*for (int j = -1; j < energyImage.cols+1; j++){
		energyValues[-1][j] = integer_max;
		energyValues[energyImage.rows][j] = integer_max;
		
	}
	for (int i = -1; i < energyImage.rows+1; i++){
		energyValues[i][-1] = integer_max;
		energyValues[i][energyImage.cols] = integer_max;
	}
	*/

	if (direction == 'V' /*seam_type == VERTICAL*/) {
		indexPath.resize(energyImage.rows);

		//filling the table with all pixel energy values -- add one more rows and column with infinity values.
		// -- change the pixel energy from int to double for acuracy.
			//pixel1 = energyValues[i-1][max((j-1),0]
			//pixel2 = energyValues[i-1] [j]
			//pixel3 = energyValues [i-1] [min((j+1),energyImage.cols-1)]
		
		
		//updating the table with minimum cost values
		for (int i = 1; i < energyImage.rows; i++) {
			for (int j = 0; j < energyImage.cols; j++) {
				// int index1 = i-1;
				// int index2 = j - 1;
				// int index3 = j + 1;
				// int index4 = j;

				// double energyPixel1 = energyValues[index1][index2];
				// double energyPixel2 = energyValues[index1][index4];
				// double energyPixel3 = energyValues[index1][index3];

				// if (j == 0) {
					// energyValues[i][j] += MINIMUM_OF(energyPixel2, energyPixel3);
				// }
				// else if (j == energyImage.cols - 1) {
					// energyValues[i][j] += MINIMUM_OF(energyPixel1, energyPixel2);
				// }
				// else {
					// energyValues[i][j] += MINIMUM_OF(energyPixel1,min(energyPixel2,energyPixel3));		
				// }
				double pixel1 = energyValues[i-1][max((j-1),0)];
				double pixel2 = energyValues[i-1] [j];
				double pixel3 = energyValues [i-1] [min((j+1),energyImage.cols-1)];
				energyValues[i][j] += MINIMUM_OF(pixel1, MINIMUM_OF(pixel2, pixel3));
			}
		}

		minValueIndex = find_minimumPoint(energyImage.cols,energyImage.rows, minimumValue, indexPath, energyValues, minValueIndex, direction);
		// backtracking finding the minimum cost path or seam , storing it into indexPath vector
		// starting from second last row to first row.
		// change -- way to implement
		//find_Minimum_Value_Path(direction, energyValues, energyImage);
		for (int i = energyImage.rows - 2; i >= 0; i--) { // reducing width	
			int value1 = MAXIMUM_OF(minValueIndex - 1, 0);
			int value2 = minValueIndex;
			int value3 = MINIMUM_OF(minValueIndex + 1, energyImage.cols - 1);
			int pixelRate1 = energyValues[i][value1];
			int pixelRate2 = energyValues[i][value2];
			int pixelRate3 = energyValues[i][value3];
			
			if (MINIMUM_OF(pixelRate1, pixelRate2) > pixelRate3) {
               			pathIndex = 1;
            		}
           		else if (MINIMUM_OF(pixelRate1, pixelRate3) > pixelRate2) {
                		pathIndex = 0;
            		}
            		else if (MINIMUM_OF(pixelRate2, pixelRate3) > pixelRate1) {
               			pathIndex = -1;
           		}
			
			minValueIndex += pathIndex;
          		minValueIndex = MINIMUM_OF(MAXIMUM_OF(minValueIndex, 0), energyImage.cols - 1); // take care of edge cases
			indexPath.at(i) = minValueIndex;
		}
	}

	else if (direction == 'H' /*seam_type == HORIZONTAL*/) {

		indexPath.resize(energyImage.cols);
		//int energyValues[energyImage.rows][energyImage.cols];
		
		

		/*	updating the table with minimum cost values --  remove extra conditions
			pixel1 = energyValues[max((i-1),0)][j-1]
			pixel2 = energyValues[i] [j-1]
			pixel3 = energyValues [min((i+1),energyImage.rows-1)] [j-1]
		*/
		for (int j = 1; j < energyImage.cols; j++) {
			for (int i = 0; i < energyImage.rows; i++) {
				
				// if (i == 0) {
					// energyValues[i][j] = energyValues[i][j] + MINIMUM_OF(energyValues[i][j - 1], energyValues[i + 1][j - 1]);
				// }
				// else if (i == energyImage.rows - 1) {
					// energyValues[i][j] = energyValues[i][j] + MINIMUM_OF(energyValues[i - 1][j - 1], energyValues[i][j - 1]);
				// }
				// else {
					// double pixelValue1 = energyValues[i - 1][j - 1];
					// double pixelValue2 = energyValues[i][j - 1];
					// double pixelValue3 = energyValues[i + 1][j - 1];
					// energyValues[i][j] = energyValues[i][j] + MINIMUM_OF(pixelValue1, MINIMUM_OF(pixelValue2, pixelValue3));
				// }
				double pixel1 = energyValues[max((i-1),0)][j-1];
				double pixel2 = energyValues[i] [j-1];
				double pixel3 = energyValues [min((i+1),energyImage.rows-1)] [j-1];
				energyValues[i][j] += MINIMUM_OF(pixel1, MINIMUM_OF(pixel2, pixel3));
			}
		}

		minValueIndex = find_minimumPoint(energyImage.cols,energyImage.rows, minimumValue, indexPath, energyValues, minValueIndex, direction);
		//cout<<minValueIndex<<endl;
		for (int i = energyImage.cols - 2; i >= 0; i--) {
			
			int pixelRate1 = energyValues[MAXIMUM_OF(minValueIndex - 1, 0)][i];
			int pixelRate2 = energyValues[minValueIndex][i];
			int pixelRate3 = energyValues[MINIMUM_OF(minValueIndex + 1, energyImage.rows - 1)][i];
			
			if (MINIMUM_OF(pixelRate1, pixelRate2) > pixelRate3) {
               			pathIndex = 1;
            		}
           		else if (MINIMUM_OF(pixelRate1, pixelRate3) > pixelRate2) {
                		pathIndex = 0;
            		}
            		else if (MINIMUM_OF(pixelRate2, pixelRate3) > pixelRate1) {
               			pathIndex = -1;
           		}
			
			minValueIndex += pathIndex;
          		minValueIndex = MINIMUM_OF(MAXIMUM_OF(minValueIndex, 0), energyImage.rows - 1); // take care of edge cases
			indexPath.at(i) = minValueIndex;
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
		cout << "Invalid request!!! new_height has to be smaller than the current size!" << endl;
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
	

	return true;
}
