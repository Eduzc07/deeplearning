#ifndef FLOODFILLCRATEHANDLESEGMENTER_H
#define FLOODFILLCRATEHANDLESEGMENTER_H

#include <list>

#include "CrateHandle.h"

struct PointToFlood;

class FloodFillCrateHandleSegmenter: public CrateHandleSegmenter
{
private:
    bool m_SegmentationOK;
    
private:
    ///functors used when flooding the logo images
    struct PointToFlood {
        virtual bool operator()(const cv::Vec3b& ptColor) {
            return true;
        }
        virtual ~PointToFlood() {}
    };


    struct WhitePointToFlood : public PointToFlood {
        bool operator()(const cv::Vec3b& ptColor) {
            return (ptColor[0] >= 200 && ptColor[1] >= 200 && ptColor[2] >= 200);
        }
    };


    struct BlackPointToFlood : public PointToFlood {
        bool operator()(const cv::Vec3b& ptColor) {
            return (ptColor[0] < 50 && ptColor[1] < 50  && ptColor[2] < 50);
        }
    };


public:
    FloodFillCrateHandleSegmenter(const cv::Mat& inputImage, const cv::Mat& originalImage, bool shrink, bool cut = true): CrateHandleSegmenter(inputImage, originalImage, shrink, cut), m_SegmentationOK(false) {}
    ~FloodFillCrateHandleSegmenter() {}

    bool segmentationOK() { return m_SegmentationOK; }
    bool findHandles(std::vector<CrateHandle>& handles);

public:
    /**
    * Each logo segment image (obtained from the logo recognition with 22 images) contains the crate logo surrounded by a black background
    * and by a white background. This function removes the black and white background from a logo segment image.
    * @param[in]: halfImage is a flag that says whether the image it is actually the superior half of a logo segment image
    * @param[out]: mask is a binary image showing where the crate is in the input image
    * @param[out]: returns false when either the white or the black backgroud removal do not work
    */
    bool removeBlackAndWhiteBackground(cv::Mat& mask, bool halfImage = false);
    /**
      * Each logo segment image (obtained from the logo recognition with 22 images) contains the crate logo surrounded by a black background
      * and by a white background. This function removes the white background from a logo segment image. It uses a flood fill function from
      * Opencv to flood all points from the edges of the image until it finds no more white points.
      * @param[in]: halfImage is a flag that says whether the image it is actually the superior half of a logo segment image
      * @param[out]: mask is a binary image showing where the white background is in the input image
      * @param[out]: returns false when an error occurs in the preprocessing of the image: see function preprocessingWhiteBackgroundRemoval
      */
    bool removeWhiteBackground(cv::Mat& mask, bool halfImage = false);

    /**
      * Each logo segment image (obtained from the logo recognition with 22 images) contains the crate logo surrounded by a black background
      * and by a white background. This function removes the black background from a logo segment image. It uses the same flood fill strategy
      * as RemoveWhiteBackground but instead of flooding from the edges of the image it floods from the edges of the mask resulted from
      * the ResultWhiteBackground function.
      * @param[in]: halfImage is a flag that says whether the image it is actually the superior half of a logo segment image
      * @param[in]: threshHigh is a flag that expands the flood fill range when true
      * @param[in]: mask showing the position of the white background in the image
      * @param[out]: mask shows at the end where is the background (white and black) of the image
      */
    bool removeBlackBackground(cv::Mat& mask, bool bigThresh, bool halfImage);

    /**
      * Removes the black background and performs a connected component labeling of the mask to eliminate small segments
      * and to test the quality of the background removal.
      * @param[in]: halfImage is a flag that says whether the image it is actually the superior half of a logo segment image
      * @param[in]: threshHigh is a flag that expands the flood fill range when true
      * @param[in]: mask showing the position of the white background in the image
      * @param[out]: mask shows at the end where the crate is in the image
      */
    bool removeBlackBackgroundAndSegment(cv::Mat& mask, HandleRecognitionSegment& crateSegment, bool thresHigh, bool halfImage = false);
    bool removeBlackBackgroundAndSegment(cv::Mat& mask, bool thresHigh, bool halfImage = false);

    /**
      * This function removes the background and the crate from the logo segment image - the purpose is to obtain the handle area.
      * It uses the same flood fill strategy as RemoveBlackBackground but instead of flooding from the edges of the image it
      * floods from the edges of the mask resulted from the ResultBlackBackgroundAndSegment function.
      * @param[in]: halfImage is a flag that says whether the image it is actually the superior half of a logo segment image
      * @param[in]: thresh is a parameter used by the flood fill function
      * @param[in]: mask showing the position of the background in the image
      * @param[out]: mask shows at the end where the empty part of the crate side is
      */
    bool removeCrate(cv::Mat& mask, const cv::Scalar& thresh);

    /**
      * Performs postprocessing of the found handle segment. Removes the pixels belonging to the segment whose color differs
      * substantially from the average color of the segment.
      * @param[in]: halfImage is a flag that says whether the image it is actually the superior half of a logo segment image
      * @param[in]: thresh is a parameter used by the flood fill function
      * @param[in]: mask showing the position of the background in the image
      * @param[out]: mask shows at the end where the empty part of the crate side is
      */
    bool removeOutlierPixels(const cv::Mat& mask, std::map< cv::Vec3b, ImageSegment, LessVec3b >& segments, cv::Mat& segmentedImage);
    /**
      * Erodes the handle.
      * @param[in]: handle is the handle to be processed
      * @param[out]: segments are the results of the handle segmentation
      */
    bool erodeHandles(const cv::Mat& mask, std::map< cv::Vec3b, ImageSegment, LessVec3b >& segments, cv::Mat& segmentedImage);

    inline int getLeft()  { return m_Left; }
    inline int getRight() { return m_Right; }
    inline int getTop() { return m_Top; }
    inline int getBottom() { return m_Bottom; }
    
private:
    /**
     * Draws a line where the white border finishes, thus CHANGING m_InputImage, this is necessary to prevent false flooding of white crates.
     */
    bool preprocessingWhiteBackgroundRemoval();

    /**
     * Produces a list of points from the border of the image
     */
    void prepareBorderPointsWhiteRemoval(std::list<cv::Point>& borderPoints, bool halfImage = false);

    /**
     * Produces a list of points at the edge of white black backgrounds of the image
     */
    void prepareBorderPointsBlackRemoval(std::list< cv::Point >& borderPoints, cv::Mat& whiteRemovedMask, int maxDistThreshold, bool halfImage = false);

    /**
     * Produces a list of points from the border of the image
     */
    void prepareBorderPointsCrateRemoval(std::list< cv::Point >& borderPoints, cv::Mat& blackAndWhiteRemovedMask, int maxDistThreshold, int topSpace, bool halfImage = false);


    /**
     * Floods m_inputImage with a given threshold and starting from the given borderpoints
     */
    bool floodWithBorderPoints(cv::Mat& mask, std::list< cv::Point >& borderPoints, FloodFillCrateHandleSegmenter::PointToFlood* pointToFlood, const cv::Scalar& thresh);

    /**
     * Computes the distance between two vectors
     */
    double dist(const cv::Vec3f& f1, const cv::Vec3f& f2);

private:
    int m_Left = 0;
    int m_Right = 0;
    int m_Top = 0;
    int m_Bottom = 0;
};


#endif // FLOODFILLCRATEHANDLESEGMENTER_H
