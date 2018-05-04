#ifndef _HANDLERECOGNITIONSEGMENT_
#define _HANDLERECOGNITIONSEGMENT_

#include "ImageSegment.h"

#include <QRect>

class HandleRecognitionSegment : public ImageSegment {
public:
    HandleRecognitionSegment(): ImageSegment() {}
    HandleRecognitionSegment(const ImageSegment& iseg) {
        m_Left = iseg.m_Left;
        m_Right = iseg.m_Right;
        m_Top = iseg.m_Top;
        m_Bottom = iseg.m_Bottom;
        m_AvgColor = iseg.m_AvgColor;
        m_ColorSegImage = iseg.m_ColorSegImage;
        m_ColorInitialImage = iseg.m_ColorInitialImage;
        m_NoPoints = iseg.m_NoPoints;
        m_IndexLabelSegmentation = iseg.m_NoPoints;        
    }

    /**
    * Generates the edge points of top, bottom, left, right edges
    * @param[in]: mat the image containing the segment
    * @param[in]: horizontal, first specify which edge points to return
    * @param[in]: verbose specify whether to display debug messages
    * @param[out]: returns the number of generated points
    */
    int generateEdge(const cv::Mat& mat, bool horizontal, bool first, std::vector< cv::Point >& edgePoints, bool verbose = false) const;
    /**
    * Generates all the contour points of the segment
    * @param[in]: mat the image containing the segment
    * @param[out]: contourPoints is the  vector containing all the contourPoints
    */
    void generateContourPoints(const cv::Mat& mat, std::vector< cv::Point >& contourPoints) const;

    /**
    * Calculates the center of mass of the segment
    * @param[in]: mat the image containing the segment
    */
    cv::Point centerMass(const cv::Mat& mat);
    /**
    * Calculates the center of mass of the two symmetric halves of the segment
    * @param[in]: mat the image containing the segment
    * @param[in]: center Mass the center of mass of the segment
    * @param[in]: the angle of the symmetry axis of the segment
    * @param[out]: centerMassLeft the center of mass of the left half of the segment
    * @param[out]: centerMassRight the center of mass of the right half of the segment
    */
    void centerMassHalf(const cv::Mat& mat, const cv::Point& centerMass, double angleVert, cv::Point& centerMassLeft, cv::Point& centerMassRight) const;
    /**
    * Scores symmetry based on the points in the segment. Calculates the symmetriy of each point in the segment and counts
    * how many from these symmetric points are found in the segment.
    * @param[in]: mat the image containing the segment
    * @param[in]: centerMass the center of mass of the segment
    * @param[in]: symmetryAngle the angle of the symmetry axis of the segment
    * @param[out]: returns the percentage of points whose symmetric is in the segment
    */
    double scoreSymmetry1(const cv::Mat& mat, const cv::Point& centerMass, double symmetryAngle) const;
    /**
    * Scores symmetry based on the position of the two lateral centers of mass. Calculates the difference
    * between the angle of the line between the two centers of mass with the horizontal and the given
    * symmetry angle.
    * how many from these symmetric points are found in the segment.
    * @param[in]: mat the image containing the segment
    * @param[in]: centerMass the center of mass of the segment
    * @param[in]: symmetryAngle the angle of the symmetry axis of the segment
    * @param[out]: returns the 1 - ratio of abs of diff angle and PI/2
    */
    double scoreSymmetry2(const cv::Mat& mat, cv::Point centerMass, double symmetryAngle) const;
    /**
    * Scores how many vertical lines from the segment contain holes (the small holes are ignored)
    * @param[in]: mat the image containing the segment
    * @param[out]: returns the 1 - average hole points per a tenth of the segment's area
    */
    double scoreHoles(const cv::Mat& mat);
    /**
    * Uses cv::findContours to get the contour of the segment. After it calculates the contour
    * it smoothens it by averaging over the neighbouring points.
    * @param[in]: mat the image containing the segment
    * @param[out]: smoothedContour contains the smoothed contour as std::vector<QPoint> (for using it in CrateRecogData)
    * @param[out]: returns how many points were found in the contour
    */
    double scoreContour(const cv::Mat& mat, std::vector< QPoint >& smoothedContour);





    /**
    * Finds where the segment touches the top side of its containing rectangle.
    * If this position is close to the middle of the segment returns 0.0, else
    * returns the minimum angle of the line between this touching point and a point
    * in the other extremity of the segment (if point touches top on the left side
    * then it generates all lines between this point and points on the right side
    * and takes the minimum angle). Function is used to check for abnormalities in their
    * crate handle segmentation.
    * @param[in]: mat the image containing the segment
    * @param[out]: 0.0 or the minimum angle as explained above
    */
    double checkRotationUp(const cv::Mat& mat);
    /**
     * Similar with checkRotationUp but for the bottom side of the segment
    * @param[in]: mat the image containing the segment
    * @param[out]: 0.0 or the minimum angle as explained above
    */
    double checkRotationDown(const cv::Mat& mat);
};


/**
* Calculate the symmetric of a point (x,y) relative to a line passing through (x0,y0) with the angle symmetryAngle with the vertical
* @param[in]: x,y position of the initial point
* @param[in]: x0,y0 position of the point on the reference line
* @param[in]: symmetryAngle the andle of the reference line with the vertical
* @param[out]: the position of the symmetric of (x,y)
*/
void calculateSymmetric(double x, double y, double x0, double y0, double symmetryAngle, double& x2, double& y2);

/**
* Calculate whether the point is in the lower plane compared to a (almost) horizontal line
* @param[in]: point, point to be analized
* @param[in]: lineRefPoint, reference point on an axis
* @param[in]: angleHoriz angle of the line with the horizontal
*/
bool pointInLowerPlane(const cv::Point& point, const cv::Point lineRefPoint, double angleHoriz);

/**
* Calculate whether the point is in the left plane compared to a (almost) vertical line
* @param[in]: point, point to be analized
* @param[in]: lineRefPoint, reference point on an axis
* @param[in]: angleVert angle of the line with the horizontal
*/
bool pointInLeftPlane(const cv::Point& point, const cv::Point lineRefPoint, double angleVert);


#endif 