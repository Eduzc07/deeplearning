#ifndef __HANDLEFEATURE_
#define __HANDLEFEATURE_

#include <vector>
#include <map>
#include <QString>
#include <opencv2/core/core.hpp>


class VertHorizHistograms {
   
public:
    VertHorizHistograms(const cv::Mat& image);
    
    cv::Mat getVertHistAsImage(int index);
    cv::Mat getHorizHistAsImage(int index);
    inline cv::Mat getGrayImage(int index) { return m_SingleChannelImages[index];  }
    inline cv::Mat getBinaryImage(int index) { return m_BinarisedImages[index]; }
    inline std::vector<std::vector<int>> getVertHistograms() { return m_VertHistogram; }
    inline std::vector<std::vector<int>> getHorizHistograms() { return m_HorizHistogram; }
    inline int getForegroundPointCount(int index) { return m_ForegroundPointsCount[index]; }
    
private:
    virtual void generateOtsuImages();
    void generateHistograms();
    void generateVertHistograms(const cv::Mat& binImg, std::vector<int>& histo, bool backgroundBright);
    cv::Mat getHistAsImage(const std::vector<int>& histo);
    void computeHistoPointCount(int index);    
    
private:
    std::vector<cv::Mat> m_BinarisedImages;
    std::vector<cv::Mat> m_SingleChannelImages;
    std::vector<bool> m_BackgroundBright;
    std::vector<int> m_ForegroundPointsCount;
    std::vector<std::vector<int>> m_VertHistogram;
    std::vector<std::vector<int>> m_HorizHistogram;
    const int m_MinVal = 0;
    int m_MaxValX = 200;
    int m_MaxValY = 200;
    int m_Channels = 3;
};

///KNN like class
class VertHorizHistoDataBase {
    struct VertHorizHistoDBEntry {
        int m_ID;
        std::vector<std::vector<int>> m_VertHist;
        std::vector<std::vector<int>> m_HorizHist;
    };
    
public:
    VertHorizHistoDataBase() {}
    void train(const QString& path, int maxClassId);
    ///returns pairs crateID, score sorted decreasing on the score
    std::vector<std::pair<int, double>> predictClass(const std::vector<std::vector<int>>& vertHist, const std::vector<std::vector<int>>& horizHist);

private:
    double compareHist(const std::vector<int>& hist1, const std::vector<int>& hist2);
    
private:
    ///pair histogram with number of points in the histogram for faster searching
    std::multimap<int, VertHorizHistoDBEntry> m_DB;
    int m_Channels = 3;
};


#endif


