#ifndef _BACKGROUND_CRATEHANDLE_SEGMENTER_
#define _BACKGROUND_CRATEHANDLE_SEGMENTER_

/**
 * Class uses the full logosegment and computes the crate surface out of the image.
 * The handle is what remains in the superior part of the image.
 */ 

#include "CrateHandle.h"
#include "RGBImageClusterer.h"

class BackgroundCrateHandleSegmenter : public CrateHandleSegmenter {
public:
    BackgroundCrateHandleSegmenter() {}
    BackgroundCrateHandleSegmenter(const cv::Mat& inputImage, const cv::Mat& originalImage);

    /**
    * @brief calculate a list of segmentations (kmeans methods)
    * decide which segment is the background 
    * extract it
    * and infer the position of the handle
    */
    bool findHandles(std::vector<CrateHandle>& handles) override;

    int estimateBackgroundImage();
    inline cv::Mat getBackgroundCompositionImage() { return m_BackgroundComposition; }
    
private:    
    /**
     * @brief create the used image clusterers 
     * initializes the member variable m_Clusterers
     */
    void createClusterers();
    
    /**
     * @brief: finds the segment corresponding to the background segment 
     * @return: true if background was found, false otherwise
     * @param: segments, list of segments obtained from an image segmentations
     * @param: backId, color of the background in the given segmentation which corresponds to the background
     */
    bool findBackgroundSegment(const std::map<cv::Vec3b, ImageSegment, LessVec3b>& createBackgroundImage, cv::Vec3b& backId);
    
    void buildCompositionImage(int greyGrades);
    
private:
    int m_MinClusterNo = 2;
    int m_MaxClusterNo = 6;
    std::vector<RGBImageClusterer*> m_Clusterers;
    cv::Mat m_BackgroundComposition;
    std::vector<std::pair<cv::Vec3b, cv::Mat>> m_FoundBackgrounds;  
};



#endif

