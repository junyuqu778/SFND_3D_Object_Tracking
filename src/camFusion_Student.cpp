
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    std::vector<cv::DMatch> match_roi;  
    for (auto match:kptMatches)
    {
        cv::KeyPoint curr_kpt = kptsCurr[match.trainIdx];
        if (boundingBox.roi.contains(curr_kpt.pt)) 
            match_roi.push_back(match);         //get the keypoints match within roi and save them in match_roi
    }

    // calculate the mean distance
    double mean_dist = 0;
    for  (auto match:match_roi)  // find sum
        mean_dist += match.distance;
    
    if (match_roi.size() > 0) // to avoid division by 0
        mean_dist = mean_dist/match_roi.size();  
    else
        return;

    double dist_threshold = mean_dist * 1.2; 
    for  (auto match:match_roi)
    {
        if (match.distance < dist_threshold)    // check distance with threshold
            boundingBox.kptMatches.push_back(match);    // add if less than th
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    {
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        {
            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            {
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    std::sort(distRatios.begin(), distRatios.end());
    long medIndex = floor(distRatios.size() / 2.0);
    double medDistRatio = distRatios.size() % 2 == 0 ? (distRatios[medIndex - 1] + distRatios[medIndex]) / 2.0 : distRatios[medIndex]; // compute median dist. ratio to remove outlier influence
    
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medDistRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    
    vector<double> p_X, c_X;

    for (auto point:lidarPointsPrev)
        if (fabs(point.y) < 2.0)   //lane width is 4m
            p_X.push_back(point.x);  //long direction
    
    for (auto point:lidarPointsCurr)
    {
        if (fabs(point.y) < 2.0)
            c_X.push_back(point.x);
       
    }
    
    // sort
    sort(p_X.begin(), p_X.end()); 
    sort(c_X.begin(), c_X.end()); 
    
    double minPrev, minCurr;
    if (p_X[1] - p_X[0] > 0.1)
        minPrev = p_X[1];
    else
        minPrev = p_X[0];   //d0
    
    if (c_X[1] - c_X[0] > 0.1)
        minCurr = c_X[1];   //d1
    else
        minCurr = c_X[0];
    
    TTC =  minCurr / (minPrev - minCurr) / frameRate;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    vector<Box_P> box_pairs;
    for (auto match : matches)
    {
        // box in previous image
        cv::KeyPoint prev_kpt = prevFrame.keypoints[match.queryIdx];
        vector<vector<BoundingBox>::iterator> prev_boxes;
        for (auto it_p = prevFrame.boundingBoxes.begin(); it_p!= prevFrame.boundingBoxes.end(); it_p++)
        {
            if (it_p->roi.contains(prev_kpt.pt))
                prev_boxes.push_back(it_p);
        }

        // box in current image
        cv::KeyPoint curr_kpt = currFrame.keypoints[match.trainIdx];
        vector<vector<BoundingBox>::iterator> curr_boxes;
        for (auto it_c = currFrame.boundingBoxes.begin(); it_c!= currFrame.boundingBoxes.end(); it_c++)
        {
            if (it_c->roi.contains(curr_kpt.pt))
                curr_boxes.push_back(it_c);
        }

        // Store the matched boxes & counting
        if (prev_boxes.size() == 1 && curr_boxes.size() == 1) //whether the points belong to one object
        {
            bool matched = false;
            for (auto it : box_pairs)
            {
                if (prev_boxes[0]==it.prev_box && curr_boxes[0]==it.curr_box)
                {
                    it.match_count++;
                    matched = true;
                }
            }
            if (!matched)
            {
                Box_P box_p;
                box_p.prev_box = prev_boxes[0];
                box_p.curr_box = curr_boxes[0];
                box_p.match_count++;
                box_pairs.push_back(box_p);                
            }
        }
    }

    // Finding the match with the highest number of occurences
    for (auto pb_itr = prevFrame.boundingBoxes.begin(); pb_itr != prevFrame.boundingBoxes.end(); pb_itr++)
    {
        bool matched = false;
        vector<BoundingBox>::iterator enclos_box;
        int max_count = 0;
        for (auto bit : box_pairs)
        {
            if (bit.prev_box == pb_itr && bit.match_count > max_count)
            {
                enclos_box = bit.curr_box;
                max_count = bit.match_count;
                matched = true;
            }
        }

        // Store result in output
        if (matched)
            bbBestMatches.insert(make_pair(pb_itr->boxID, enclos_box->boxID));
    }
}
