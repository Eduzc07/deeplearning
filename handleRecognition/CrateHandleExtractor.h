#ifndef CRATEHANDLEEXTRACTOR_H
#define CRATEHANDLEEXTRACTOR_H

#include "FloodFillCrateHandleSegmenter.h"
#include "KMeansCrateHandleSegmenter.h"

///class that uses Kmeans and flood fill based segmenters to extract the crate handle
class CrateHandleExtractor
{
    enum CRATE_BRIGHTNESS {DARK = 0, BRIGHT = 1, NORMAL = 2};
    
    cv::Mat m_InputImage;
    cv::Mat m_InputImageCopy;
    CrateHandle m_Handle;
    std::vector<CrateHandle> m_Handles;
    
    ///constants
    int m_ThresMaxDark = 100;
    int m_ThresMinLight = 150;
    int m_ThresFloodDimTol = 5;

    
public:
    CrateHandleExtractor(const cv::Mat& inputImage): m_InputImage(inputImage) {  }
    ~CrateHandleExtractor() {}
    
    /**
    * Combines two segmentation methods to extracts the optimal handle if it is consistent.
    * @param[out]: returns false when the extracted handles are inconsistent
    */       
    bool execute();
    CrateHandle getFoundHandle();
    std::vector<CrateHandle>& getAllHandles() { return m_Handles; }
    
private:
    /**
    * Calculate the average color of the image
    */   
    int imageBrightness();
    /**
    * Handle detection with the flood fill method
    * @param[out]: handle the obtained CrateHandle object
    * @param[out]: found returns whether a handle has been found or not
    * @param[out]: returns whether the handle segmentation was succesful or not; when not it means that the crate is probably a dark crate  
    */     
    bool handleDetectionFloodFill(CrateHandle& handle, bool& found);
    /**
    * Handle detection with the kmeans method
    * @param[out]: handles the obtained CrateHandle objects
    * @param[in]: minCluster, maxCluster the number of clusters that are to be tested
    * @param[in]: completeKMeans, whether the algorithm tries to join labels with the found handle
    */   
    void handleDetectionKMeans(std::vector<CrateHandle>& handles, int minCluster, int maxCluster, bool completeKMeans = true);

    /**
    * Adaptive histogram equalization
    */       
    void equalizeColor();
    
    /**
    * Tests if the flood fill handle matches with the kmeans handles.
    * @param[in]: handlesKMeans handles obtained with the kmeans segmentation method
    * @param[in]: handleFloodFill handle obtained with the flood fill method
    * @param[out]: returns whether the floodFillHandle dimensions match the KMeans handles dimensions
    */       
    bool removeFloodFillHandle(const std::vector<CrateHandle>& handlesKMeans, const CrateHandle& handleFloodFill);
    /**
    * Tests if the center of mass of the floodFillHandle is in the same position with the average center of mass of the KMeans handles
    * @param[in]: handles KMeans handles and floodFill handles taken together
    */       
    bool differentHandles(const std::vector<CrateHandle>& handles) const;

    /**
    * Sorts the handles based on the contour and median scores and choses the handle with the optimal combination of those scores.
    * @param[in]: handles KMeans handles and floodFill handles taken together
    * @param[out]: returns the optimal handle
    */   
    CrateHandle findOptimalHandle(const std::vector<CrateHandle> &handles);
};

#endif // CRATEHANDLEEXTRACTOR_H
