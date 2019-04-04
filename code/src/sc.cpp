/**
* Manan Prajapati 
* COMP 6651: Project 2 Seam Carving
* REFERENCES: https://docs.opencv.org/3.4.3/da/d85/tutorial_js_gradients.html
* 
*/

#include <vector>
#include "sc.h"

#define MINIMUM_OF(value1, value2) ((value1 < value2) ? value1 : value2)
#define MAXIMUM_OF(value1, value2) ((value1 > value2) ? value1 : value2) 
#define integer_max 2147483647
#define one 1
#define minusOne -1
#define zero 0

using namespace cv;
using namespace std;

Mat kernelX = (Mat_<double>(3,3) <<1,1,1,0,0,0,-1,-1,-1);
Mat kernelY = (Mat_<double>(3,3) <<-1,0,1,-1,0,1,-1,0,1);

bool printing = false;
//calculating energy map
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

	// if(printing)
		// imshow("gradient image", gradient_image);
	
}


int find_minimumPoint(int imageSizeCol, int imageSizeRow, int minimumValue, vector<int>& VH_seams, double** robust_table, int minValueIndex, String seam_type){
	//finding the minimum cost value index from last row and column and putting the minimum cost index in vector.
	if(seam_type == "VERTICAL"){
		for (int j = 0; j < imageSizeCol; j++) {
			if (robust_table[imageSizeRow - 1][j] < minimumValue) {
				minimumValue = robust_table[imageSizeRow - 1][j];
				VH_seams.at(imageSizeRow - 1) = j;
				minValueIndex = j;
			}
		}
	}else{
		for (int j = 0; j < imageSizeRow; j++) {
			if (robust_table[j][imageSizeCol - 1] < minimumValue) {
				minimumValue = robust_table[j][imageSizeCol - 1];
				VH_seams.at(imageSizeCol - 1) = j;
				minValueIndex = j;
			}
		}
	}
	return minValueIndex;
}

vector<int> find_seam(Mat& robustImage, String seam_type /*Seam_Location seam_type*/) {

	vector<int> VH_seams;
	int minimumValue = integer_max;
	int pathIndex = 0;
	int minValueIndex = integer_max;
	int image_rows = robustImage.rows;
	int image_columns = robustImage.cols;
	//create table
	double** robust_table = new double*[image_rows];
	for (int i = 0; i < image_rows; ++i) {
		robust_table[i] = new double[image_columns];
	}
	//assign enrgy_pixel value to table
	for (int i = 0; i < image_rows; i++)
			for (int j = 0; j < image_columns; j++)
				robust_table[i][j] = (double)robustImage.at<uchar>(i, j);
	

	if (seam_type == "VERTICAL") {
		VH_seams.resize(image_rows);

		//updating the table with minimum cost values
		for (int i = 1; i < image_rows; i++) {
			for (int j = 0; j < image_columns; j++) {
				double enrgy_pixel1 = robust_table[i-1][max((j-1),0)];
				double enrgy_pixel2 = robust_table[i-1] [j];
				double enrgy_pixel3 = robust_table [i-1] [min((j+1),image_columns-1)];
				robust_table[i][j] += MINIMUM_OF(enrgy_pixel1, MINIMUM_OF(enrgy_pixel2, enrgy_pixel3));
			}
		}

		minValueIndex = find_minimumPoint(image_columns,image_rows, minimumValue, VH_seams, robust_table, minValueIndex, seam_type);
		// backtracking finding the minimum cost path or seam , storing it into VH_seams vector
		// starting from second last row to first row.
		for (int i = image_rows - 2; i >= 0; i--) { // reducing width	
			int value1 = MAXIMUM_OF(minValueIndex - 1, 0);
			int value2 = minValueIndex;
			int value3 = MINIMUM_OF(minValueIndex + 1, image_columns - 1);
			double enrgy_pixelRate1 = robust_table[i][value1];
			double enrgy_pixelRate2 = robust_table[i][value2];
			double enrgy_pixelRate3 = robust_table[i][value3];
			
			if (MINIMUM_OF(enrgy_pixelRate1, enrgy_pixelRate2) > enrgy_pixelRate3) {
               	pathIndex = one;
            }
           	else if (MINIMUM_OF(enrgy_pixelRate1, enrgy_pixelRate3) > enrgy_pixelRate2) {
                pathIndex = zero;
            }
            else if (MINIMUM_OF(enrgy_pixelRate2, enrgy_pixelRate3) > enrgy_pixelRate1) {
               	pathIndex = minusOne;
           	}
			
			minValueIndex = minValueIndex + pathIndex;
			int indexvalue = MAXIMUM_OF(minValueIndex, 0);
          	minValueIndex = MINIMUM_OF(indexvalue, image_columns - 1);
			VH_seams[i] = minValueIndex;
		}
	}

	else if (seam_type == "HORIZONTAL") {

		VH_seams.resize(image_columns);
		
		for (int j = 1; j < image_columns; j++) {
			for (int i = 0; i < image_rows; i++) {
				double enrgy_pixel1 = robust_table[max((i-1),0)][j-1];
				double enrgy_pixel2 = robust_table[i] [j-1];
				double enrgy_pixel3 = robust_table [min((i+1),image_rows-1)] [j-1];
				robust_table[i][j] += MINIMUM_OF(enrgy_pixel1, MINIMUM_OF(enrgy_pixel2, enrgy_pixel3));
			}
		}

		minValueIndex = find_minimumPoint(image_columns,image_rows, minimumValue, VH_seams, robust_table, minValueIndex, seam_type);
		
		for (int j = image_columns - 2; j >= 0; j--) {
			double enrgy_pixelRate1 = robust_table[MAXIMUM_OF(minValueIndex - 1, 0)][j];
			double enrgy_pixelRate2 = robust_table[minValueIndex][j];
			double enrgy_pixelRate3 = robust_table[MINIMUM_OF(minValueIndex + 1, image_rows - 1)][j];
			
			if (MINIMUM_OF(enrgy_pixelRate1, enrgy_pixelRate2) > enrgy_pixelRate3) {
               	pathIndex = one;
            }
           	else if (MINIMUM_OF(enrgy_pixelRate1, enrgy_pixelRate3) > enrgy_pixelRate2) {
                pathIndex = zero;
            }
            else if (MINIMUM_OF(enrgy_pixelRate2, enrgy_pixelRate3) > enrgy_pixelRate1) {
               	pathIndex = minusOne;
           	}
			
			minValueIndex += pathIndex;
          	minValueIndex = MINIMUM_OF(MAXIMUM_OF(minValueIndex, 0), image_rows - 1);
			VH_seams.at(j) = minValueIndex;
		}
	}

	return VH_seams;
}

//validating the image
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

//default function
bool seam_carving(Mat& in_image, int new_width, int new_height, Mat& out_image) {
	return seam_carving_trivial(in_image, new_width, new_height, out_image);
}

// seam carves by removing trivial seams
bool seam_carving_trivial(Mat& in_image, int new_width, int new_height, Mat& out_image) {

	Mat input_Image = in_image.clone();
	Mat output_Image = in_image.clone();
	Mat gradientImage;

	//1st step: Compute Energy Map
	printing = true;
	computeEnergyMap(in_image, gradientImage);
	printing = false;

	while (input_Image.rows != new_height) {
		// horizontal seam if needed
		if (input_Image.rows > new_height) {
			reduce_horizontal_seam_trivial(input_Image, output_Image, gradientImage);
			input_Image = output_Image.clone();
			computeEnergyMap(input_Image, gradientImage);
		}
	}
	while (input_Image.cols != new_width) {
		if (input_Image.cols > new_width) {
			reduce_vertical_seam_trivial(input_Image, output_Image, gradientImage);
			input_Image = output_Image.clone();
			computeEnergyMap(input_Image, gradientImage);
		}
	}

	out_image = output_Image.clone();
	return true;
}

// horizontal trivial seam is a seam through the center of the image
bool reduce_horizontal_seam_trivial(Mat& in_image, Mat& output_Image, Mat& robustImage) {

	// retrieve the dimensions of the new image
	int rows = in_image.rows - 1;
	int cols = in_image.cols;

	// create an image slighly smaller
	output_Image = Mat(rows, cols, CV_8UC3);

	vector<int> seam_path;
	seam_path = find_seam(robustImage, "HORIZONTAL");

	for (int j = 0; j < cols; ++j) {
		for (int i = 0; i < seam_path.at(j); ++i) {

			Vec3b enrgy_pixel = in_image.at<Vec3b>(i, j);
			output_Image.at<Vec3b>(i, j) = enrgy_pixel;
		}

		for (int i = seam_path.at(j); i < rows; ++i) {
			Vec3b enrgy_pixel = in_image.at<Vec3b>(i + 1, j);
			output_Image.at<Vec3b>(i, j) = enrgy_pixel;
		}
	}

	return true;
}

// vertical trivial seam is a seam through the center of the image
bool reduce_vertical_seam_trivial(Mat& in_image, Mat& output_Image, Mat& robustImage) {
	// retrieve the dimensions of the new image
	int rows = in_image.rows;
	int cols = in_image.cols - 1;

	// create an image slighly smaller
	output_Image = Mat(rows, cols, CV_8UC3);

	vector<int> seam_path;
	seam_path = find_seam(robustImage, "VERTICAL");
	
	for(int out_rows = 0; out_rows < output_Image.rows; out_rows++ ) {
		for (int out_col = 0; out_col < output_Image.cols; out_col++){
			if (out_col >= seam_path.at(out_rows))
				output_Image.at<Vec3b>(out_rows,out_col) = in_image.at<Vec3b>(out_rows,out_col+1);
			else
				output_Image.at<Vec3b>(out_rows,out_col) = in_image.at<Vec3b>(out_rows,out_col);
		}
	}
	return true;
}
