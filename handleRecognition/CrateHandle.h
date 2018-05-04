#ifndef CRATEHANDLE_H
#define CRATEHANDLE_H

#include <map>

#include "HandleRecognitionSegment.h"
#include "logohandle.h"


///Handle features
struct CrateHandleFeatures {
    ///Handle features
    double m_Width = 0.0;
    double m_Height = 0.0;
    double m_WidthHeight = 0.0;
    double m_FillFactor = 0.0;
    double m_ImageWidth = 0.0;
    double m_ImageHeight = 0.0;
    double m_WidthRatio = 0.0;
    double m_HeightRatio = 0.0;
    double m_TopCornerYHeight = 0.0;

    double m_SymmetryAxis1 = 0.0;
    double m_SymmetryAxis2 = 0.0;
    double m_SymmetryAxis3 = 0.0;
    double m_SymmetryAxis4 = 0.0;
    double m_SymmetryAxis5 = 0.0;
    cv::Point m_CenterMass = cv::Point(0, 0);
    cv::Point m_CenterMassLeft = cv::Point(0, 0);
    cv::Point m_CenterMassRight = cv::Point(0, 0);
    double m_CenterMassHalfWidth = 0.0;
    double m_CenterMassSection = 0.0;
    double m_CenterMassSectionHeight = 0.0;
    double m_CenterMassLatSectionHeight = 0.0;
    double m_CenterMassBottomSection = 0.0;
    double m_CenterMassLatBottomSection = 0.0;
    double m_TopDistance = 0.0;
    double m_BottomDistance = 0.0;
    double m_TopBottomDistance = 0.0;
    double m_BottomHeight = 0.0;
    double m_LeftDistance = 0.0;
    double m_RightDistance = 0.0;

    double m_BottomCenter = 0.0;
    double m_BottomCenterLeft = 0.0;
    double m_BottomCenterRight = 0.0;
    QPoint m_BottomCenterHalfLeft = QPoint(0, 0);
    QPoint m_BottomCenterHalfRight = QPoint(0, 0);

    ///the contour of the handle
    std::vector<QPoint> m_Contour;

    ///segmentation scores
    double m_SegContourScore = 0.0;
    double m_SegMedianScore = 0.0;
    double m_SegFormScore = 0.0;

public:
    CrateHandleFeatures() {}
    ~CrateHandleFeatures() {}
};

struct CrateHandle {
    ///image containing the handles: there can be more than 1 detected handle in the handle image (since the handles are chosen by geometrical criteria only)
    cv::Mat m_HandleImage;
    ///map containing the segments found by segmentation in m_HandleImage
    std::map<cv::Vec3b, HandleRecognitionSegment, LessVec3b> m_SegmentsMap;
    ///colors corresponding to the handle segments in m_HandleImage
    std::vector<cv::Vec3b> m_HandleColors;

    ///the calculate features for each handle in m_HandleImage
    std::vector<CrateHandleFeatures> m_HandleFeatures;

    ///constants
    double m_ScaleFactor = 2.0;
    ///How many angles are tested for the symmetry axis of the handle
    int m_ThreshNoAngleTests1 = 40;
    ///Range of the symmetry angle that is tested for symmetry
    double m_ThreshTargetSymmetryAngle1 = 10;
    int m_ThreshNoAngleTest2 = 10;
    double m_ThreshTargetSymmetryAngle2 = 2.5;
    ///Maximum score at which we reject a handle
    static double m_ThresMaxHandleReject;/* = 0.8;*/
    
    ///constants handle recognition
    int m_ThreshHandleColor = 255;
    double m_ThreshLateralAssymetry = 0.3;
    double m_ThresPointNo = 0;
    double m_ThreshHandleMinDistanceToTop = 0;
    double m_ThreshHandleMinDistanceLateral = 0;
    double m_ThreshHandleMinHeight = 0;
    double m_ThreshHandleMinWidth = 0;

    CrateHandle() {}
    ///used for training and tests
    CrateHandle(const cv::Mat& handleImage);
    ~CrateHandle() {}

    inline CrateHandle clone() {
        CrateHandle retval;
        retval.m_HandleImage = m_HandleImage.clone();
        retval.m_HandleColors = m_HandleColors;
        retval.m_SegmentsMap = m_SegmentsMap;
        retval.m_HandleFeatures = m_HandleFeatures;
        return retval;
    }

    /**
    * Calculates the 3 scores for the segmentation:  m_SegContourScore, m_SegMedianScore = 0.0, m_SegFormScore
    * Calculates also the features for each handle in m_HandleImage
    * @param[in]: number of handle (used only for debug purposes)
    * @param[out]: return false when an error had occured
    */
    bool score(int i);
    /**
    * Constructs a crate handle from a segmented image, a map of segments and a list of colors
    * Function has almost constructor semantics
    * @param[in]: segmentedImage is a segmented image
    * @param[in]: segments resulted from the segmentation of the image
    * @param[in]: handleSegmentColors the colors in which the handles are represented in the segmentedImage
    */
    void build(const cv::Mat& segmentedImage, const std::map< cv::Vec3b, ImageSegment, LessVec3b >& segments, const std::vector< cv::Vec3b >& handleSegmentColors);
    /**
    * Given a list of segments it extracts the handles based on their dimension and position, and completes the CrateHandle structure
    * @param[in]: segmentedImage is a segmented image
    * @param[in]: segments resulted from the segmentation of the image
    * @param[out]: returns false when CrateHandle is not constructed (build function is not called)
    */
    bool detectAndCreate(const cv::Mat& segmentedImage, const std::map< cv::Vec3b, ImageSegment, LessVec3b >& segments);
    /**
    * Eliminates the handles that have a too poor segmentation score from the CrateHandleStructure.
    * @param[out]: returns false when all the handles are poorly segmented and true otherwise
    */
    bool badScore();
    bool tooManyHandles() const { return m_HandleFeatures.size() != 1;}
    /**
    * Eliminates the handles that have a too poor symmetry score from the CrateHandle structure.
    * Function is similar with badScore but uses different criteria.
    * @param[out]: returns false when all the handles are poorly segmented and true otherwise
    */
    bool badSegmentation();
    /**
    * Keeps only one handle in the CrateHandle structure
    * @param[in]: index for the handle that will be kept
    */
    void keepUniqueHandle(int idx);
    std::vector<CrateHandleFeatures> getFeatures() const { return m_HandleFeatures; }

    /**
    * Extracts from the crate handle structure the features of the handle and saves the in the logohandledata structure
    * @param[out]: CLogoHandleData that needs to be filled
    * @param[out]: true if all the data has been succesfully saved, false otherwise
    */
    bool computeDBEntry(CLogoHandleData& lhd);

    /**
    * From the info given by the user in the handle editor builds the handle features.
    * @param[out]: CLogoHandleData that needs to be filled
    * @param[out]: true if all the data has been succesfully saved, false otherwise
    */
    bool extractFeatures(CLogoHandleData& lhd);

    /**
    * Interpolates the handle contour
    * @param[in]: polygon as edited in the handle editor
    * @param[in]: list of points in the contour 
    */
    std::vector<std::pair<int, int>> interpolateHandleContour(const std::vector<int>& poly);

private:
    /**
    * Calculates the features of all handles in the CrateHandle
    */
    bool calculateDistances();
    /**
    * Calculates the median line and the median line score
    */
    bool calculateMedianLine();
    /**
    * Deletes handles from the CrateHandle
    * @param[in]: keepList indices of handles that will be kept
    * @param[in]: eraseList indices of handles that will be erased
    */
    void eraseHandles(const std::vector<int>& keepList, const std::vector<int>& eraseList);

    /**
    * Help functions for handle contour interpolation
    */    
    std::vector<std::pair<int, int>> findInterpolating(const std::vector<int>& x, const std::vector<int>& y);
    std::vector<std::pair<int, int>> interpolateSecondOrder(const std::vector<int>& x, const std::vector<int>& y, bool onX);
    /**
    * Restore colors in binary image after resize
    */
    void restoreBinaryColours(cv::Mat& mat);
};

///base class for the handle segmenters
class CrateHandleSegmenter
{
protected:
    cv::Mat m_InputImage;
    cv::Mat m_InputImageOriginal;

    bool m_Shrink = false;
    bool m_Cut = false;

public:
    CrateHandleSegmenter() {}
    CrateHandleSegmenter(const cv::Mat& inputImage, const cv::Mat& originalImage, bool shrink, bool cut = true);

    ~CrateHandleSegmenter() {}

    /**
     * Algorithm for handle segmentation.
     @param[out]: handles the data structures containing the segmented handles
     */
    virtual bool findHandles(std::vector<CrateHandle>& handles) = 0;
};

#endif // CRATEHANDLE_H
