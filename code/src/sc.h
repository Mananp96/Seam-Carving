#ifndef SEAMCARVINGCOMP665156
#define SEAMCARVINGCOMP665156

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// the function you need to implement - by defaiult it calls seam_carving_trivial
bool sanityCheck(int new_width, int new_height, int orignal_width, int orignal_height);
void computeEnergyMap(cv::Mat& in_image, cv::Mat& gradient_image);

//vector<int> dynamicProgramming(cv::Mat& energyImage, char direction);

bool seam_carving(cv::Mat& in_image, int new_width, int new_height, cv::Mat& out_image);


bool seam_carving_trivial(cv::Mat& in_image, int new_width, int new_height, cv::Mat& out_image);

bool reduce_horizontal_seam_trivial(cv::Mat& in_image, cv::Mat& out_image, cv::Mat& energyImage);


bool reduce_vertical_seam_trivial(cv::Mat& in_image, cv::Mat& out_image, cv::Mat& energyImage);



#endif

