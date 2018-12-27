#include <vector>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "saliency.h"
#include <math.h>
using namespace std;
using namespace cv;
#include <iostream>
#include<time.h>
#include <ml.h>		  // opencv machine learning include file


int main()
{
	//strings of path
	string dem_path="/home2/nepal_national/dem_sub_resize_tif.png";//dem of the study area
	string b1_b3_path="/home2/nepal_national/2015_123_mosaic.tif";//extract clouds  b1_jpg.jpg is obtained from idl
	string b4_b6_path="/home2/nepal_national/2015_456_mosaic.tif";//mainly remove background bounding boxes
	string gt_path="/home2/nepal_national/gt_mosaic_prj_sub_resize_bi_evi_png.png";//ground truth image
	string evi_processed_path="/home2/nepal_national/evi_mosaic_tif.png";//normalized evi 
 
  string img123_txt_path="/home2/nepal_national/img123_test.txt";
  string img456_txt_path="/home2/nepal_national/img456_test.txt";
  string dem_txt_path="/home2/nepal_national/dem_test.txt";
  ofstream img123_txt_file;
  ofstream img456_txt_file;
  ofstream dem_txt_file;
  img123_txt_file.open(img123_txt_path.c_str());//,ios::app
  img456_txt_file.open(img456_txt_path.c_str());//,ios::app
  dem_txt_file.open(dem_txt_path.c_str());//,ios::app

  cv::Mat evi_norm_origin=imread(evi_processed_path.c_str(),cv::IMREAD_UNCHANGED);//
  cout<<"finish laoding evi_norm"<<endl;
  cout<<evi_norm_origin.rows<<" "<<evi_norm_origin.cols<<endl;
	cv::Mat b1_b3=imread(b1_b3_path.c_str(),cv::IMREAD_UNCHANGED);
  cout<<"finish loading image b1_b3"<<b1_b3.rows<<b1_b3.cols<<endl;
	cv::Mat b4_b6=imread(b4_b6_path.c_str(),cv::IMREAD_UNCHANGED);
  cout<<"finish loading image b4_b6"<<b4_b6.rows<<b4_b6.cols<<endl;
	cv::Mat dem=imread(dem_path.c_str(),0);
  cout<<"finish loading image dem"<<dem.rows<<dem.cols<<endl;
	
	//enhance each channel from b1-b6
  //std::vector<Mat> channels_123;
  //std::vector<Mat> channels_456;
  //split(b1_b3, channels_123);
  //split(b4_b6, channels_456);
  //std::vector<Mat> channels_123_enhanced;
  //std::vector<Mat> channels_456_enhanced;
 // for(int i=0;i<3;i++)
 // {
 //   equalizeHist(channels_123[i],channels_123_enhanced[i]);
 //   equalizeHist(channels_456[i],channels_456_enhanced[i]);
 // }
 // merge(channels_123_enhanced,b1_b3);
 // merge(channels_456_enhanced,b4_b6);
  cv::Mat evi_norm;
  equalizeHist(evi_norm_origin,evi_norm);
 
  //imwrite("/home2/nepal_national/2015_123_mosaic_enhanced.png",b1_b3);
//	generate_saliency_probability(evi_img_int_equalhist,"/home2/nepal_national/2015_evi_processed_enhanced.png",evi_enhance,200);//事实上，180在这里没用
//   cout<<"finish enhancing b1-b6"<<endl;
	
	//load in ground truth image
	cv::Mat gt_img_origin=imread(gt_path.c_str(),cv::IMREAD_UNCHANGED);
  cout<<"gt img:"<<gt_img_origin.rows<<" "<<gt_img_origin.cols<<endl;
 	cv::Mat gt_img(gt_img_origin.rows,gt_img_origin.cols,CV_8UC1);
  for(int i=0;i<gt_img.rows;i++)
  {
    for(int j=0;j<gt_img.cols;j++)
    {
       int gt_value=gt_img_origin.at<uchar>(i,j);
       if(gt_value>0)
       {
          gt_img.at<uchar>(i,j)=1;
       }
       else
       {
          gt_img.at<uchar>(i,j)=0;
       }
    }
  }
 //imwrite("/home2/nepal_national/2015_gt_mosaic_prj.png",gt_img);

	//threshold(gt_img_origin,gt_img,50.0,255,THRESH_BINARY);
	cv::Mat evi_enhance_img;
//	threshold(evi_enhance,evi_enhance_img,50,255,THRESH_BINARY);
  threshold(evi_norm,evi_enhance_img,180,255,THRESH_BINARY);
  //imwrite("/home2/nepal_national/evi_mosaic_prj_bi.png",evi_enhance_img);

	//contour_based urban extraction 
	std::cout << "start to calculate contour" << endl;
	vector<vector<Point>> contours_1;
	findContours(evi_enhance_img, contours_1, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	
	double area_threshold=250000;
	int img_index=0;
	cv::Mat evi_img_3c;
	cvtColor(evi_norm, evi_img_3c, CV_GRAY2BGR);
	for (size_t i = 0; i < contours_1.size(); i++)
	{
		double area = cv::contourArea(contours_1[i]);
		cv::Rect rect = cv::boundingRect(contours_1[i]);

		if((area<=area_threshold)&&(area>10))
		{
			Mat rec_img1_3;
			Mat rec_gt;
			Mat rec_img4_6;
			Mat rec_dem;
			if(area<90000)
			{
				cv::Rect rect_new=rect;
				if(rect.height<300)
				{
					if((evi_enhance_img.rows-rect.y)<300)
					{ rect_new.height=evi_enhance_img.rows-rect.y;}
					else
					{
						rect_new.height=300;
					}
				}
				if(rect.width<300)
				{
					if((evi_enhance_img.cols-rect.x)<300)
					{ rect_new.width=evi_enhance_img.cols-rect.x;}
					else
					{
						rect_new.width=300;
					}
				}

				rec_img1_3=b1_b3(rect_new);
				rec_img4_6=b4_b6(rect_new);
				rec_gt=gt_img(rect_new);
				rec_dem=dem(rect_new);
				cv::rectangle(evi_img_3c, rect_new, cv::Scalar(0, 0, 255), 1, 8);
			}
			else
			{
				rec_img1_3=b1_b3(rect);
				rec_img4_6=b4_b6(rect);
				rec_gt=gt_img(rect);
				rec_dem=dem(rect);
				cv::rectangle(evi_img_3c, rect, cv::Scalar(0, 255, 0), 1, 8);
			}

		
			img_index=img_index+1;
			stringstream stream;  
      stream<<img_index;  
      string ind_str=stream.str();  
			string dem_rec_path="/dem_sample_2015/dem_"+ind_str+".png";
			string img1_3_path="/img_sample_2015/img1_3_"+ind_str+".png";
			string img4_6_path="/img_sample_2015/img4_6_"+ind_str+".png";
			string gt_rec_path="/gt_sample_2015/gt_"+ind_str+".png";
      
  //    img123_txt_file<<img1_3_path<<" "<<gt_rec_path<<endl;
  //    img456_txt_file<<img4_6_path<<" "<<gt_rec_path<<endl;
  //    dem_txt_file<<dem_rec_path<<" "<<gt_rec_path<<endl;      
      img123_txt_file<<img1_3_path<<endl;
      img456_txt_file<<img4_6_path<<endl;
      dem_txt_file<<dem_rec_path<<endl;      
      
      
      string dem_rec_full_path="/home2/nepal_national"+dem_rec_path;
			string img1_3_full_path="/home2/nepal_national"+img1_3_path;
			string img4_6_full_path="/home2/nepal_national"+img4_6_path;
			string gt_rec_full_path="/home2/nepal_national"+gt_rec_path;
      
	//		imwrite(dem_rec_full_path.c_str(),rec_dem);
	//		imwrite(img1_3_full_path.c_str(),rec_img1_3);
	//		imwrite(img4_6_full_path.c_str(),rec_img4_6);
	//		imwrite(gt_rec_full_path.c_str(),rec_gt);
			
			
		}

	
	}
	img123_txt_file.close();
  img456_txt_file.close();
  dem_txt_file.close();
 // imwrite("/home2/nepal_national/evi_3c.png",evi_img_3c);
  cout<<"done"<<endl;
	
  return 1;
	

}
