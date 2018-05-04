#ifndef _RGBIMAGECLUSTERINGSEGMENTER_
#define _RGBIMAGECLUSTERINGSEGMENTER_

#include "RGBImageClusterer.h"
#include "ConnectedComponentLabelling.h"
#include "ImageSegment.h"


/**
 * Segmentation in two steps 1. pixel clustering 2. connected component labelling 
 */

class RGBImageClusteringSegmenter {
public:
    /**
     * rgbImage should be the input image in the pixelclusterer used
     */
    RGBImageClusteringSegmenter(RGBImageClusterer* pixelClusterer) : m_InputImage(pixelClusterer->getInputImage()), m_PixelClusterer(pixelClusterer) {}

    /**
     * clusters the image with a clusterer
     * builds the segments with connected component labelling
     * @param[out]: true when an error occurred
     */
    bool execute(std::map <cv::Vec3b, ImageSegment, LessVec3b>& segmentsMap, cv::Mat& output);
    
private:
    cv::Mat& m_InputImage;
    cv::Mat m_SegmentedImage;
    RGBImageClusterer* m_PixelClusterer;

    ///complex data structure where the information about each segment after segmentation are saved.
    std::map<cv::Vec3b, ImageSegment, LessVec3b> m_Segments;
};

#endif 
