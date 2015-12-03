// measureTools.cpp : 定义控制台应用程序的入口点。
//

#include "stdafx.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include  "opencv2/core/core.hpp"
#include <math.h>
#include <iostream>
#include <fstream>
#include <iostream>
#include <time.h>
#include <iomanip>
#define _IMG_DIR_ "H:\\labCode\\result\\measureresult\\"
using namespace cv;
using namespace std;

static void help()
{
	cout
		<< "\nThis program calculate F measure and Error rate in image segmentation\n"
		<< "this app read two images, one for ground truth ,one for your segmentation results\n"
		<< "Usage:\n"
		<< "./contours2\n"
		<< "\nA trackbar is put up which controls the contour level from -3 to 3\n"
		<< endl;
}
const double _beta = 0.3;
string _dir = _IMG_DIR_;
string _imageResultName = "152077.jpg";
string _imageGroundTruth = "152077.png";
static int countPixels(const Mat& img){

	//only accept uchar type!
	CV_Assert(img.depth() != sizeof(uchar));     

	int channels = img.channels();

	int nRows = img.rows * channels; 
	int nCols = img.cols;
	int countPix = 0;

	if (img.isContinuous())
	{
		nCols *= nRows;
		nRows = 1;         
	}

	int i,j;
	const uchar* p; 
	for( i = 0; i < nRows; ++i)
	{
		p = img.ptr<uchar>(i);
		for ( j = 0; j < nCols; ++j)
		{
			if(p[j]>0)
				++countPix;
		}
	}
	return countPix; 
}
static void logInfo(std::string filename, double fmeasure, double precision,double recall){
		std::string filepath(_IMG_DIR_);
		filepath = filepath + "\\" + "log.txt";
		std::fstream outfile(filepath,std::ios::app);
		/* format
		filepath    date    stage1time     stage2time    
		*/
		time_t curtime=time(0); 
		tm tim =*localtime(&curtime); 
		int day,mon,year; 
		day=tim.tm_mday;
		mon=tim.tm_mon;
		year=tim.tm_year;
		outfile<<year+1900<<"年"<<mon+1<<"月"<<day<<"日:  ";
		outfile<<filename<<std::setw(10);
		outfile<<"  fmeasure:"<< fmeasure <<"   ";
		outfile<<"  precision:"<< precision <<"   ";
		outfile<<"  recall:"<< recall <<"   ";
		outfile<<"\n";
		outfile.close();
		
}

int _tmain(int argc, _TCHAR* argv[])
{
	// init images path
	string resultPath = _dir + _imageResultName;
	string groundPath = _dir + _imageGroundTruth;


	Mat imgResult = imread(resultPath, CV_LOAD_IMAGE_GRAYSCALE);
	Mat imgGround = imread(groundPath, CV_LOAD_IMAGE_GRAYSCALE);

	if(imgResult.empty() || imgGround.empty()){
		fprintf(stderr, "Can not load image from %s and %s\n",resultPath, groundPath);
		return -1;
	}

	if(imgResult.size() != imgGround.size()){
		fprintf(stderr, "Can not calculate result because the input images do not have same size");
		return -1;
	}
	Mat resultMask = Mat::zeros(imgResult.size(),CV_8UC1);
	Mat groundMask = Mat::zeros(imgResult.size(),CV_8UC1);

	resultMask.setTo(1,imgResult);
	groundMask.setTo(1,imgGround);

	// calculate error rate according to the Ph.D Thesis of Han Shou Dong 
	// error = (fp + fn)/(fp + fn + tp + tn)
	int ammountPixs = countPixels(resultMask) + countPixels(groundMask);
	Mat temp;
	bitwise_xor(resultMask,groundMask,temp);
	int fpfn = countPixels(temp);
	double errorRate = (double)fpfn/ammountPixs;
	cout<<"The Error Rate is :"<< errorRate <<endl;
	

	//calculate f-measure = ((1+beta^2)*precision*recall)/(beta^2*precision+recall)
	
	// precision = tp/(tp+fp)
	
	Mat tpMat;
	bitwise_and(resultMask,groundMask,tpMat);
	int tpfp = countPixels(resultMask); int tp = countPixels(tpMat); int tpfn = countPixels(groundMask);
	double pres = (double)tp/tpfp;
	double recall = (double)tp/tpfn;
	
	double beta2 = _beta*_beta;

	double fmeasure = ((1+beta2)*pres*recall)/(beta2*pres+recall);
	cout <<"The fmeasure is: "<< fmeasure <<endl;

	cout <<"The precision is: "<< pres <<endl;

	cout <<"The recall is"<< recall<<endl;



	return 0;
}

