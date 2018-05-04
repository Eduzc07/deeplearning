#ifndef _OPENCVCLUSTERER_
#define _OPENCVCLUSTERER_

#include "RGBImageClusterer.h"
#include <map>
#include "ImageSegment.h"

/**
 * Uses of opencv function to cluster RGB Image
 */
class OpenCVClusterer : public RGBImageClusterer
{
protected:
    ///number of clusters
    unsigned int m_ClusterNo;
    ///calculated labels for the colors
    std::map<cv::Vec3b, int, LessVec3b> m_Labels;

public:
    OpenCVClusterer(cv::Mat& inputImage, int clusterNo): RGBImageClusterer(inputImage), m_ClusterNo(clusterNo) {}
    bool execute() override;

protected:
    /**
    * Prepare data 
    */
    virtual bool extractColorsFromImage(cv::Mat& clusterInput);

    /**
    * Given a RGB image it extracts the colors from it and clusters them 
    */
    virtual bool clusterColors() = 0;

    /**
     * builds the clustered image 
     */
    void buildClusteredImage();
};

#endif
