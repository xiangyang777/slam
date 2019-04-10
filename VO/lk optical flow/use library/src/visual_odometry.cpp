/*
 * <one line to give the program's name and a brief idea of what it does.>
 * Copyright (C) 2016  <copyright holder> <email>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <boost/timer.hpp>
#include "myslam/config.h"
#include "myslam/visual_odometry.h"
#include "myslam/g2o_types.h"

namespace myslam
{

VisualOdometry::VisualOdometry() :
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 ), matcher_flann_( new cv::flann::LshIndexParams(5,10,2) )
{
    num_of_features_    = Config::get<int> ( "number_of_features" );
    scale_factor_       = Config::get<double> ( "scale_factor" );
    level_pyramid_      = Config::get<int> ( "level_pyramid" );
    match_ratio_        = Config::get<float> ( "match_ratio" );
    max_num_lost_       = Config::get<float> ( "max_num_lost" );
    min_inliers_        = Config::get<int> ( "min_inliers" );
    key_frame_min_rot   = Config::get<double> ( "keyframe_rotation" );
    key_frame_min_trans = Config::get<double> ( "keyframe_translation" );
    map_point_erase_ratio_ = Config::get<double> ("map_point_erase_ratio");
    fast_ = cv::FastFeatureDetector::create ();
}

VisualOdometry::~VisualOdometry()
{

}

bool VisualOdometry::addFrame ( Frame::Ptr frame )
{
    switch ( state_ )
    {
    case INITIALIZING:
    {
        state_ = OK;
        curr_ = ref_ = frame;
        map_->insertKeyFrame ( frame );
        // extract features from first frame and add them into map
        extractKeyPoints();
        
        break;
    }
    case OK:
    {
        curr_ = frame;
        extractKeyPoints();
        calcOpticalFlow();
        poseEstimationessential();
        if ( checkEstimatedPose() == true ) // a good estimation
        {
            curr_->T_c_w_ = T_c_r_estimated_ * ref_->T_c_w_;  // T_c_w = T_c_r*T_r_w 
            ref_ = curr_;
            setVectorZero();
            num_lost_ = 0;
            if ( checkKeyFrame() == true ) // is a key-frame
            {
                addKeyFrame();
            }
        }
        else // bad estimation due to various reasons
        {
            num_lost_++;
            if ( num_lost_ > max_num_lost_ )
            {
                state_ = LOST;
            }
            return false;
        }
        break;
    }
    case LOST:
    {
        cout<<"vo has lost."<<endl;
        break;
    }
    }

    return true;
}

void VisualOdometry::extractKeyPoints()
{
    boost::timer timer;
    fast_->detect ( curr_->color_, keypoints_curr_ );
    cout<<"extract keypoints cost time: "<<timer.elapsed()<<endl;
}

void VisualOdometry::calcOpticalFlow()
{
    vector<unsigned char> status;
    vector<float> error;
    vector<cv::Point2f> next_keypoints;
    vector<cv::Point2f> keypoints;
    vector<cv::Point2f> prev_keypoints;
    for ( auto kp: keypoints_curr_ ){
        prev_keypoints.push_back(kp.pt);
        keypoints.push_back(kp.pt);
        pts_2d_ref_.push_back(kp.pt);
    }
    cv::calcOpticalFlowPyrLK(ref_->color_,curr_->color_,prev_keypoints,next_keypoints,status,error);
    int i=0;
    for(auto iter=keypoints.begin();iter!=keypoints.end();i++){
        if(status[i]==0){
            iter=keypoints.erase(iter);
            continue;
        }
        *iter = next_keypoints[i];
        iter++;
    }
    for(auto kp:keypoints){
        pts_2d_curr_.push_back(kp);
    }
    
    
}


void VisualOdometry::poseEstimationessential()
{
    // construct the 3d 2d observations
    vector<cv::Point2f> pts2d1;
    vector<cv::Point2f> pts2d2;
    for(auto kp:pts_2d_ref_){
        pts2d1.push_back(kp);
    }
    for(auto kp:pts_2d_curr_){
        pts2d2.push_back(kp);
    }
    Mat K = ( cv::Mat_<double>(3,3)<<
        ref_->camera_->fx_, 0, ref_->camera_->cx_,
        0, ref_->camera_->fy_, ref_->camera_->cy_,
        0,0,1
    );
    

    cv::Point2d principal(ref_->camera_->cx_,ref_->camera_->cy_);
    Mat essential_matrix;
    essential_matrix=cv::findEssentialMat(pts2d1,pts2d2,ref_->camera_->fy_,principal,RANSAC);
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    cv::recoverPose(essential_matrix,pts2d1,pts2d2,R,t,ref_->camera_->fy_),principal;
    
    T_c_r_estimated_ = SE3(
        R, 
        t
    );
    
  
}

void setVectorZero(){
    pts_2d_ref_.clare();
    pts_2d_curr_.clare();
}



bool VisualOdometry::checkEstimatedPose()
{
    // check if the estimated pose is good
    if ( num_inliers_ < min_inliers_ )
    {
        cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
        return false;
    }
    // if the motion is too large, it is probably wrong
    Sophus::Vector6d d = T_c_r_estimated_.log();
    if ( d.norm() > 5.0 )
    {
        cout<<"reject because motion is too large: "<<d.norm()<<endl;
        return false;
    }
    return true;
}

bool VisualOdometry::checkKeyFrame()
{
    Sophus::Vector6d d = T_c_r_estimated_.log();
    Vector3d trans = d.head<3>();
    Vector3d rot = d.tail<3>();
    if ( rot.norm() >key_frame_min_rot || trans.norm() >key_frame_min_trans )
        return true;
    return false;
}

void VisualOdometry::addKeyFrame()
{
    cout<<"adding a key-frame"<<endl;
    map_->insertKeyFrame ( curr_ );
}

}
