#ifndef _CONNECTEDCOMPONENTLABELLING_
#define _CONNECTEDCOMPONENTLABELLING_

#include "opencv2/core/core.hpp"
#include <set>
#include <map>
#include "ImageSegment.h"

/**
 * Performs connected component labelling on cv::Mat
 */
class ConnectedComponentLabelling
{
private:
    ///initialImage is the image that was initially clustered
    const cv::Mat& m_InitialImage;
    ///clusteredImage is the clustering of the initial image (kmeans clustered image, binary image etc.)
    const cv::Mat& m_ClusteredImage;
    ///computeAvgColor is a flag that says whether the average color of each segment should be calculated or not
    bool m_ComputeAvgColor;
    ///area from the image where the connected component labelling is performed
    QRect m_BoundingRect;

public:
    ConnectedComponentLabelling(const cv::Mat& initialImage, const cv::Mat& clusteredImage): m_InitialImage(initialImage), m_ClusteredImage(clusteredImage), m_ComputeAvgColor(false), m_BoundingRect(QRect(0, 0, 0, 0))  {}
    ~ConnectedComponentLabelling() {}

    inline void setBoundingRect(const QRect& rect) { m_BoundingRect = rect; }
    inline void setComputeAvgColor(bool flag) { m_ComputeAvgColor = flag; }

    /**
    * Performs connected component labeling from a clustered image.
    *@param[out]: segmentsMap the list of segments resulting after connect component labelling
    *@param[out]: the segmented image
    */
    void execute(std::map <cv::Vec3b, ImageSegment, LessVec3b>& segmentsMap, cv::Mat& output);

private:
    /**
     * Used in the first pass of the connected component region labeling algorithm
     * Gets the list of pixels around the current pixel that were already marked with a label and which are in the same kmeans cluster
     * @param[in]: mat matrix containing clustered data (kmeans in color space)
     * @param[in]: labels matrix of assigned labels
     * @param[in]: row, col - position of the current point
     * @param[in]: mask - connectivity mask giving the neighbours of a point in the image
     * @param[out]: the list of neighbours of the point
     */
    void getNeighbours(const cv::Mat& labels, int row, int col,  const cv::Vec3b& currentLabelCluster,
                       const std::vector< cv::Point >& mask, std::set<unsigned int>& neighbLabels, unsigned int& minNeighbLabel, bool repeat, int& counter);

    /**
     * Used in the first pass of the connected component region labeling algorithm
     * Joins a region marked with a given label with region marked with label_set
     * @param[in]: label is the label of the region to be joined
     * @param[in]: label_set are the labels of the regions to be joined with
     * @param[out]: parentLabels is a data structure keeping track of all regions and their neighbours
     */
    void setUnion(unsigned int label, const std::set< unsigned int >& label_set, std::vector< unsigned int >& parentLabels);
    /**
     * The same as the above only label set consists of a single label
     */
    void setUnion(unsigned int label1, unsigned int label2, std::vector< unsigned int >& parentLabels);
    /**
     * Finds the upmost parent of a region - the label of this found region will be the final label of the region after labelling
     * @param[in]: label is the label of the region to be described
     * @param[out]: parentLabels is a data structure keeping track of all regions and their neighbours
     */
    unsigned int setFind(unsigned int label, const std::vector< unsigned int >& parentLabels);
    /**
     * Debug function
     */
    void printPointInfo(const cv::Mat& labels, int row, int col, const std::vector< cv::Point >& mask);
};




#endif
