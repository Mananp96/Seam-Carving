/**
* Manan Prajapati 
* COMP 6651: Project 2 Seam Carving
* REFERENCES: https://docs.opencv.org/3.4.3/da/d85/tutorial_js_gradients.html
* http://www.cs.cmu.edu/afs/andrew/scs/cs/15-463/f07/proj2/www/wwedler/
*/

#include <cmath>
#include <limits>
#include <vector>

#include "sc.h"

using namespace cv;
using namespace std;
enum SeamLocation { VER, HOR };

//calculating energy map ----done
void computeEnergyMap(Mat& in_image, Mat& gradient_image){
	Mat temp_in_image;
	Mat gradient_x, gradient_y;
	Mat gray_image;
	
	//reducing noise
	GaussianBlur(in_image,temp_in_image, Size(3, 3), 0, 0, BORDER_DEFAULT);
	
	cvtColor(temp_in_image, gray_image, CV_BGR2GRAY);
	Scharr(gray_image, gradient_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
	Scharr(gray_image, gradient_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
	
	convertScaleAbs(gradient_x, gradient_x);
	convertScaleAbs(gradient_y, gradient_y);
	addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0, gradient_image);
	//grad.convertTo(energy_image, CV_64F, 1.0/255.0);
}

vector<int> dynamicProgramming(Mat& energyImage, char direction){
	
	vector<int> indexPath;
	int minimum = numeric_limits<int>::max();

	if(direction=='V'){
		indexPath.resize(energyImage.rows);
		int energyValues[energyImage.rows][energyImage.cols];
		int pixelValue1, pixelValue2, pixelValue3;
		
		for(int i=0; i<energyImage.rows; i++)
			for(int j=0; j<energyImage.cols; j++)
				energyValues[i][j] = (int)energyImage.at<uchar>(i,j);
			
		for(int i=1; i<energyImage.rows; i++){
			for(int j=0; j<energyImage.cols; j++){
				
				if(j==0){	
					energyValues[i][j] = energyValues[i][j] + min(energyValues[i-1][j], energyValues[i-1][j+1]); 
				}
				else if(j==energyImage.cols-1){
					energyValues[i][j] = energyValues[i][j] + min(energyValues[i-1][j-1], energyValues[i-1][j]); 
				}
				else{
					int pixelValue1 = energyValues[i-1][j-1];
					int pixelValue2 = energyValues[i-1][j];
					int pixelValue3 = energyValues[i-1][j+1];
					energyValues[i][j] = energyValues[i][j] + min(pixelValue1, min(pixelValue2, pixelValue3)); 
				}
			}
		}
		
		for(int j=0; j<energyImage.cols; j++){
			if(energyValues[energyImage.rows-1][j]<minimum){
				minimum = energyValues[energyImage.rows-1][j];
				indexPath.at(energyImage.rows-1) = j;
			}
		}
		
		for(int i=energyImage.rows-2; i>=0; i--){
			
			if(indexPath.at(i+1)==0){
				if(energyValues[i][indexPath.at(i+1)]<energyValues[i][indexPath.at(i+1)+1])
					indexPath.at(i) = indexPath.at(i+1);
				else
					indexPath.at(i) = indexPath.at(i+1)+1;
			}
			
			else if(indexPath.at(i+1)==(energyImage.cols-1)){
				if(energyValues[i][indexPath.at(i+1)]<energyValues[i][indexPath.at(i+1)-1])
					indexPath.at(i) = indexPath.at(i+1);
				else
					indexPath.at(i) = indexPath.at(i+1)-1;
			}
			
			else{
				minimum = min(energyValues[i][indexPath.at(i+1)], min(energyValues[i][indexPath.at(i+1)-1], energyValues[i][indexPath.at(i+1)+1]));
				if(minimum == energyValues[i][indexPath.at(i+1)])
					indexPath.at(i) = indexPath.at(i+1);
				else if(minimum == energyValues[i][indexPath.at(i+1)-1])
					indexPath.at(i) = indexPath.at(i+1)-1;
				else if(minimum == energyValues[i][indexPath.at(i+1)+1])
					indexPath.at(i) = indexPath.at(i+1)+1;
			}
		}
	}
	
	else if(direction=='H'){
		
		indexPath.resize(energyImage.cols);
		int energyValues[energyImage.rows][energyImage.cols];
		int pixelValue1, pixelValue2, pixelValue3;
		
		for(int i=0; i<energyImage.rows; i++)
			for(int j=0; j<energyImage.cols; j++)
				energyValues[i][j] = (int)energyImage.at<uchar>(i,j);
			
		for(int j=1; j<energyImage.cols; j++){
			for(int i=0; i<energyImage.rows; i++){
				
				if(i==0){	
					energyValues[i][j] = energyValues[i][j] + min(energyValues[i][j-1], energyValues[i+1][j-1]); 
				}
				else if(i==energyImage.rows-1){
					energyValues[i][j] = energyValues[i][j] + min(energyValues[i-1][j-1], energyValues[i][j-1]); 
				}
				else{
					int pixelValue1 = energyValues[i-1][j-1];
					int pixelValue2 = energyValues[i][j-1];
					int pixelValue3 = energyValues[i+1][j-1];
					energyValues[i][j] = energyValues[i][j] + min(pixelValue1, min(pixelValue2, pixelValue3)); 
				}
			}
		}
		
		for(int j=0; j<energyImage.rows; j++){
			if(energyValues[j][energyImage.cols-1]<minimum){
				minimum = energyValues[j][energyImage.cols-1];
				indexPath.at(energyImage.cols-1) = j;
			}
		}
		
		for(int i=energyImage.cols-2; i>=0; i--){
			
			if(indexPath.at(i+1)==0){
				if(energyValues[indexPath.at(i+1)][i]<energyValues[indexPath.at(i+1)+1][i])
					indexPath.at(i) = indexPath.at(i+1);
				else
					indexPath.at(i) = indexPath.at(i+1)+1;
			}
			
			else if(indexPath.at(i+1)==(energyImage.rows-1)){
				if(energyValues[indexPath.at(i+1)][i]<energyValues[indexPath.at(i+1)-1][i])
					indexPath.at(i) = indexPath.at(i+1);
				else
					indexPath.at(i) = indexPath.at(i+1)-1;
			}
			
			else{
				minimum = min(energyValues[indexPath.at(i+1)][i], min(energyValues[indexPath.at(i+1)-1][i], energyValues[indexPath.at(i+1)+1][i]));
				if(minimum == energyValues[indexPath.at(i+1)][i])
					indexPath.at(i) = indexPath.at(i+1);
				else if(minimum == energyValues[indexPath.at(i+1)-1][i])
					indexPath.at(i) = indexPath.at(i+1)-1;
				else if(minimum == energyValues[indexPath.at(i+1)+1][i])
					indexPath.at(i) = indexPath.at(i+1)+1;
			}
		}
	}
	
	return indexPath;
}

//validating the image ---done new
bool sanityCheck(int new_width, int new_height, int orignal_width, int orignal_height){
	
    if(new_width>orignal_width){
        cout<<"Invalid request!!! new_width has to be smaller than the current size!"<<endl;
        return false;
    }
    if(new_height>orignal_height){
        cout<<"Invalid request!!! ne_height has to be smaller than the current size!"<<endl;
        return false;
    }
    
    if(new_width<=0){
        cout<<"Invalid request!!! new_width has to be positive!"<<endl;
        return false;
    }
    
    if(new_height<=0){
        cout<<"Invalid request!!! new_height has to be positive!"<<endl;
        return false;
    }
	return true;
}

//default function ---done
bool seam_carving(Mat& in_image, int new_width, int new_height, Mat& out_image){
    return seam_carving_trivial(in_image, new_width, new_height, out_image);
}


// seam carves by removing trivial seams
bool seam_carving_trivial(Mat& in_image, int new_width, int new_height, Mat& out_image){

    Mat iimage = in_image.clone();
    Mat oimage = in_image.clone();
	Mat gradientImage;
	
	//1st step: Compute Energy Map
	computeEnergyMap(in_image, gradientImage);
	
    while(iimage.rows!=new_height || iimage.cols!=new_width){
        // horizontal seam if needed
        if(iimage.rows>new_height){
            reduce_horizontal_seam_trivial(iimage, oimage, gradientImage);
            iimage = oimage.clone();
			computeEnergyMap(iimage, gradientImage);
        }
        
        if(iimage.cols>new_width){
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
bool reduce_horizontal_seam_trivial(Mat& in_image, Mat& out_image, Mat& energyImage){

    // retrieve the dimensions of the new image
    int rows = in_image.rows-1;
    int cols = in_image.cols;
    
    // create an image slighly smaller
    out_image = Mat(rows, cols, CV_8UC3);
	
	vector<int> rowIndexPerColumn;
	rowIndexPerColumn = dynamicProgramming(energyImage, 'H');
    
    for(int j=0;j<cols;++j){
        for(int i=0;i<rowIndexPerColumn.at(j);++i){
			
            Vec3b pixel = in_image.at<Vec3b>(i, j);
            out_image.at<Vec3b>(i,j) = pixel;
        }
    
        for(int i=rowIndexPerColumn.at(j);i<rows;++i){
            Vec3b pixel = in_image.at<Vec3b>(i+1, j);
            out_image.at<Vec3b>(i,j) = pixel;
        }
	}

    return true;
}

// vertical trivial seam is a seam through the center of the image
bool reduce_vertical_seam_trivial(Mat& in_image, Mat& out_image, Mat& energyImage){
    // retrieve the dimensions of the new image
    int rows = in_image.rows;
    int cols = in_image.cols-1;
    
    // create an image slighly smaller
    out_image = Mat(rows, cols, CV_8UC3);
	
	vector<int> columnIndexPerRow;
	columnIndexPerRow = dynamicProgramming(energyImage, 'V');
    
    for(int i=0;i<rows;++i){
		for(int j=0; j<columnIndexPerRow.at(i); j++){
			
			Vec3b pixel = in_image.at<Vec3b>(i, j);
            out_image.at<Vec3b>(i,j) = pixel;
		}

		for(int j=columnIndexPerRow.at(i); j<cols; j++){
			
			Vec3b pixel = in_image.at<Vec3b>(i, j+1);
            out_image.at<Vec3b>(i,j) = pixel;
		}	
    }
    
    return true;
}