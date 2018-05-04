#ifndef _OPENCVEMCLUSTERER_
#define _OPENCVEMCLUSTERER_


#include "OpenCVClusterer.h"

/**
 * Uses of opencv function to cluster RGB Image
 * Unfortunately OpenCV offers only one dimensional expectation maximization
 */
class OpenCVEMClusterer : public OpenCVClusterer
{
public:
    OpenCVEMClusterer(cv::Mat& inputImage, int clusterNo): OpenCVClusterer(inputImage, clusterNo) {}

private:
    /**
    * Given a RGB image it extracts the colors from it and clusters them into the desired number of classes with the expectation maximization algorithm.
    */
    bool clusterColors() override;
    
    bool extractColorsFromImage(cv::Mat& clusterInput) override;
    int transformColorToInt(const cv::Vec3f& color);
};

/**
 * Uses Expectation Maximization to perform the RGB color clustering and 
 * connected component labelling to perform image segmentation
 */


#endif