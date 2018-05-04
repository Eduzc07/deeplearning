#include "RGBImageClusteringSegmenter.h"

bool RGBImageClusteringSegmenter::execute(std::map< cv::Vec3b, ImageSegment, LessVec3b >& segmentsMap, cv::Mat& output)
{
    if (!m_PixelClusterer->execute())
        return true;
    ConnectedComponentLabelling ccl(m_PixelClusterer->getInputImage(), m_PixelClusterer->getClusteredImage());
    ccl.execute(m_Segments, m_SegmentedImage);
    segmentsMap = m_Segments;
    output = m_SegmentedImage;
//     output = m_PixelClusterer->getClusteredImage();
    return false;
}
