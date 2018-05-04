#ifndef _RGBIMAGECLUSTERER_
#define _RGBIMAGECLUSTERER_

#include <opencv2/core/core.hpp>

/**
 * abstract class for pixel clustering in an RGB image
 * the input is a RGB image and the output is also a clustered RGB Image
 */

class RGBImageClusterer {
public:
    RGBImageClusterer(cv::Mat& image) : m_InputImage(image) {}
    virtual ~RGBImageClusterer() {}

    /**
     *Cluster the image and save a clustered image where each cluster has a different color 
     * @param[out] - true when an error occurred
     */
    virtual bool execute() = 0;
    bool isRGBImage();
    inline cv::Mat& getClusteredImage() { return m_ClusteredImage; }
    cv::Mat& getInputImage() { return m_InputImage; }

protected:
    cv::Mat& m_InputImage;
    cv::Mat m_ClusteredImage;
};


#endif
