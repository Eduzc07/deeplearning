#ifndef KMEANSCRATEHANDLESEGMENTER_H
#define KMEANSCRATEHANDLESEGMENTER_H

#include "CrateHandle.h"

class KMeansCrateHandleSegmenter: public CrateHandleSegmenter
{
    ///number of clusters in the KMeansClustering
    int m_ClusterNo;
    ///flags indicating what kind of postprocessing will be used
    int m_CompleteKMeans;

    ///How big can a segment which added to the initial handle segment be
    int m_ThreshMergeLabelsMaxAdditionSize = 300;
    ///For line interpolation stop conditions
    int m_ThreshMinUnalteredTopEdgePoints = 0;
    int m_ThreshMinUnalteredBottomEdgePoints = 0;
    double m_ThreshMaxLineFittingTopEdgeVariance = 1.5;
    double m_ThreshMaxLineFittingBottomEdgeVariance = 0.5;
    ///For curve interpolation
    int m_ThreshCurveTracking = 5;
    int m_ThreshCurveTrackingDelta = 5;
    int m_ThreshCurveTrackingStepAngle = 9;

public:
    KMeansCrateHandleSegmenter(const cv::Mat& inputImage, const cv::Mat& originalImage,  int clusterNo, bool shrink): CrateHandleSegmenter(inputImage, originalImage, shrink), m_ClusterNo(clusterNo), m_CompleteKMeans(false) { }
    ~KMeansCrateHandleSegmenter() {}

    void setClusterNo(int clusterNo) { m_ClusterNo = clusterNo; }
    int clusterNo() { return m_ClusterNo; }
    void setCompleteKMeans(bool completeKMeans) { m_CompleteKMeans = completeKMeans; }

    bool findHandles(std::vector< CrateHandle >& handles) override;

public:
    /**
     * The function performs kMeansClustering on the input image and then ConnectedComponentLabelling on the clustered image.
     * Returns false when the clustering function fails.
     * @param[out]: output is the result of the connected component labeling - a color segmentation image
     * @param[out]: a map defining for each of the colors in the segmentation image a Segment object
     */
    bool segment(cv::Mat& output, std::map< cv::Vec3b, ImageSegment, LessVec3b >& segments);


    /**
     * This is a post processing function.
     * Given a handle it marks the interior points (optionally) and optimizes the superior and inferior edges of the handle
     * @param[in]: handle is the found handle
     * @param[in]: segments is the list of segments resulted from the segmentation
     * @param[in]: markInterior is a flag that says whether the interior points of the segment should (not) be marked
     * @param[out]: handle - the function works direct of the handle structure
     */
    void optimizeEdges(CrateHandle& handle, bool markInterior);

    /**
     * Given a handle and a segmentation of the image (obtained with a higher cluster number as the initial handle) optimizes the
     * handle by adding segments in the additional segmentation. This is useful for completing the handle segment with neighbouring labels.
     * @param[in]: handle is the found handle
     * @param[in]: segments is the list of segments resulted from the segmentation and from which handle comes
     * @param[in]: extraSegmentedImage is an image resulted from another KMeans segmentation of the input image from which the handle comes
     * @param[in]: extraSegments are the segments corresponding to the extraSegmentedImage
     * @param[out]: handle - the function works direct of the handle structure
     */
    void mergeLabels(CrateHandle& handle, std::map< cv::Vec3b, ImageSegment, LessVec3b >& segments, const cv::Mat& extraSegmentedImage, const std::map< cv::Vec3b, ImageSegment, LessVec3b>& extraSegments);

    /**
     * The function performs thresholding on the input image and then ConnectedComponentLabelling on the resulting image.
     * Returns false when the clustering function fails.
     * @param[out]: output is the result of the connected component labeling - a color segmentation image
     * @param[out]: a map defining for each of the colors in the segmentation image a Segment object
     */
    bool threshold(const cv::Mat& input, int thresh, const QRect& targetRect, cv::Mat& output, std::map< cv::Vec3b, ImageSegment, LessVec3b >& segments);

    /**
     * Verifies if lateral labels are near the handle segment. Function works well for dark crates only.
     * It works by making a thresholding step of the image with a threshold taken as five times the average color in the handle segment.
     * A connencted component labeling is used to find the target regions for labels and when these regions are lieing lateral
     * of the handle returns false else returns true.
     * @param[in]: CrateHandle handle object
     * @param[out]: see in the description for the return value.
     */
    bool checkLateralLabels(const CrateHandle& handle);

private:
    /**
     *Optimizes the top or bottom edge of the handle
     @param[in]: specifies which edges is optimized
     @param[in]: seg - segment for wich the top edge is computed
     @param[out]: returns the points corresponding to the optimized edge
    */
    bool optimizeEdge(cv::Mat& handleImage, const HandleRecognitionSegment& seg, bool top, std::vector< cv::Point >& edgePoints);


    /**
     *Tests whether an edge is a line or not and returns the line parameters
    */
    bool isEdgeLineRegression(const std::vector< cv::Point >& edgePoints, bool top, double& vx, double& vy, double& x0, double& y0, double& variance);

    /**
     *Tests whether an edge is a line with Hough transformation
    */
    bool isEdgeLineHough(const std::vector< cv::Point >& edgePoints, bool top, double& vx, double& vy, double& x0, double& y0, double& variance);


    /**
     *Tests whether an edge is a curve (not a line). Function currently not implemented.
    */
    bool isEdgeCurve(const std::vector<cv::Point>& edgePoints);


    /**
     *Interpolates the edge of the handle as a line
    */
    void interpolateEdgeLine(cv::Mat& handleImage, const ImageSegment& seg, std::vector<cv::Point>& edgePoints, bool top, double vx, double vy, double x0, double y0, double variance);

    /**
     *Prepares data for interpolating the top edge as a curve.
    */
    ///@todo: to change this function to use approxPolyDP from OpenCV
    void interpolateEdgeCurve(cv::Mat& handleImage, const ImageSegment& seg, bool top, std::vector< cv::Point >& edgePoints);

    /**
     *Interpolates the edge of the handle as a line.
     * Starting from imin (a position in the middle of the top edge) travels to the left and to the right.
     * Where he finds a discontinuity, presumably due to the presence of a label, makes the following algorithm
     * rotates a line around the discontinuity point starting in the vertical position until the line touches
     * the top edge. The part of the edge between the discontinuity and the place where the line touches the edge
     * is then newly constructor based on the found line.
    */
    void interpolateTopEdgeCurve(cv::Mat& handleImage, const ImageSegment& seg, const std::vector< cv::Point >& edgePoints, int imin);



    /**
     *Marks the interior points of a segment in the same color as the segment
     *Does not update the pixel count for the segment (should it ?).
     * The function is currently not used and probably will be deleted in the future.
     @param[in]: seg - segment to be transformed
     */
    void markInteriorPoints(cv::Mat& handleImage, const HandleRecognitionSegment& seg);

    /**
     *For each vertical/horizontal line in the segment's bounding box marks the points not belonging to the segment
     *and saves in a custom data format
     @param[in]: seg - segment to be processed
     @param[in]: horizontal - search along vertical or horizontal lines
     @param[out]: for each vertical/horizontal coordinate save the list of "line segments" (start,end)
     containing only points not belonging to the processed segment (a segment is a collection of points with similar features
     extract with a segmentation algorithm from an image)
     By hole is meant here a point not belonging to a segment.
     */
    void findHolePositions(const cv::Mat& handleImage, const ImageSegment& seg, bool horizontal, std::map< int, std::vector< std::pair< int, int > > >& holesMap);
    /**
     * Print holes positions
     *
     */
    void printHolePositions(std::map< int, std::vector< std::pair< int, int > > >& holesMap);

private:
    struct ColorCompareVertical {
        bool operator()(const cv::Mat& mat, int i, int j, const cv::Vec3b& color) const {
            return (mat.at<cv::Vec3b>(j, i) != color);
        }
    };


    struct ColorCompareHorizontal {
        bool operator()(const cv::Mat& mat, int i, int j, const cv::Vec3b& color) const {
            return (mat.at<cv::Vec3b>(i, j) != color);
        }
    };
};


#endif // KMEANSCRATEHANDLESEGMENTER_H
