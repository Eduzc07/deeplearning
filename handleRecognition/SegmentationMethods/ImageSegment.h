#ifndef _IMAGESEGMENT_H
#define _IMAGESEGMENT_H

#include <opencv/cv.h>
#include <QRect>

/**
 * Description of a image segment
 * This data structure results from a component labelling operation
 */
class ImageSegment {
///@todo: to transform into private
public:
    ///leftmost, rightmost, topmost, bottommost coordinates of the segment
    int m_Left;
    int m_Right;
    int m_Top;
    int m_Bottom;
    ///average color in the segment
    cv::Vec3f m_AvgColor;
    ///colors of the points in the segmented image
    cv::Vec3b m_ColorSegImage;
    ///color of points in the clustered image
    cv::Vec3b m_ColorInitialImage;

    ///number of points in this segment
    int m_NoPoints;
    ///label from the segmentation
    int m_IndexLabelSegmentation;

public:
    ImageSegment(): m_Left(1000000), m_Right(0), m_Top(1000000), m_Bottom(0), m_NoPoints(0), m_IndexLabelSegmentation(-1)
    {}

    /**
    * Displays segment information
    */
    void print() const;

    /**
    * Computes the average height of the segment
    * @param[in]: mat the image containing the segment
    * @param[out]: returns the average height of the segment
    */
    double avgHeight(const cv::Mat& mat);

    inline bool intersects(const QRect& rect) const {
        QRect rect1(m_Left, m_Top, m_Right - m_Left, m_Bottom - m_Top);
        return rect1.intersects(rect);
    }

    inline bool intersects(const ImageSegment& seg) const {
        QRect rect1(m_Left, m_Top, m_Right - m_Left, m_Bottom - m_Top);
        QRect rect2(seg.m_Left, seg.m_Top, seg.m_Right - seg.m_Left, seg.m_Bottom - seg.m_Top);
        return rect1.intersects(rect2);
    }

    /**
    * Tests if the segment contains another segment
    * @param[in]: mat the image containing the segment
    * @param[in]: seg is the segment to be tested
    * @param[out]: return true when seg is contained in the current segment
    */
    bool contains(const cv::Mat& mat, const ImageSegment& seg) const;   

    /**
    * Marks the segment with a new color.
    * @param[in]: destMat the image containing the segment
    * @param[in]: the new color of the segment
    */
    void markWithNewColor(cv::Mat& destMat, const cv::Vec3b& color) const;


private:
    /**
    * Finds the top point of the segment lieing on a given column
    * @param[in]: mat the image containing the segment
    * @param[in]: col is the column
    * @param[out]: position of the top point or -1 when error
    */
    int findTopSegmentPoint(const cv::Mat& mat, int col) const;
    /**
     * Similar with findTopSegmentPoint but for bottom point
    * @param[in]: mat the image containing the segment
    * @param[in]: col is the column
    * @param[out]: position of the bottom point or -1 when error
    */

    int findBottomSegmentPoint(const cv::Mat& mat, int col) const;    
};

///functor used to compare 3d points
struct LessVec3b {
    bool operator()(const cv::Vec3b& val1, const cv::Vec3b& val2) const {
        return ((val1[0] < val2[0]) || (val1[0] == val2[0] && val1[1] < val2[1])
                || (val1[0] == val2[0] && val1[1] == val2[1] && val1[2] < val2[2]));
    }
};

struct LessCvPoint {
    bool operator()(const cv::Point& val1, const cv::Point& val2) const {
        return ((val1.x < val2.x) || (val1.x == val2.x && val2.y < val2.y));
    }
};

#endif // IMAGESEGMENT_H
