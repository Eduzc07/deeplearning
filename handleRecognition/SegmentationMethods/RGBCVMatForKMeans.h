#ifndef _RGBCVMATFORKMEANS_
#define _RGBCVMATFORKMEANS_

#include <map>
#include <opencv2/core/core.hpp>

#include "ImageSegment.h"

/**
 * @brief CVMat RGB image together with its histogram
 */

class RGBCVMatForKMeans {
public:
    RGBCVMatForKMeans(cv::Mat& mat) : m_Image(mat) { computeHisto(); }
    inline cv::Mat& getImage() { return m_Image; }
    std::map<cv::Vec3b, int, LessVec3b>&  getHisto() { return m_Histo; }

private:
    void computeHisto();
    
private:
    cv::Mat& m_Image;
    std::map<cv::Vec3b, int, LessVec3b> m_Histo;
};






#endif
