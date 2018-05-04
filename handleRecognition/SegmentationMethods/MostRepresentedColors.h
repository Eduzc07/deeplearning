#ifndef _MOSTREPRESENTEDCOLORS_
#define _MOSTREPRESENTEDCOLORS_

#include "GenericKMeansImageClusterer.h"
#include "VectorDouble.h"
#include "ImageSegment.h"

class MostRepresentedColors : public GenericKMeansImageClusterer<VectorDouble> {
public:
    MostRepresentedColors(cv::Mat& mat, int histoThresh, double distClustering) : 
        GenericKMeansImageClusterer<VectorDouble>(mat, 5, 1.0), m_HistoThresh(histoThresh), m_DistClustering(distClustering) {}

    virtual void initClusterCenters();

private:
    void calculateImageHistogram(); 
    void calculateSeeds();
    ///use agglomerative clustering to cluster the seeds
    void clusterSeeds();
    bool colorBelongsToCluster(const cv::Vec3b& col, const std::vector<cv::Vec3b>& cluster);
    void mergeClusters(unsigned int i, const std::vector<unsigned int>& toMerge);

private:
    ///threshold to define representative color
    unsigned int m_HistoThresh = 50;
    ///maximum distance between representative colors in a cluster
    double m_DistClustering = 10.0;
    ///color histogram
    std::map<cv::Vec3b, unsigned int, LessVec3b> m_Histo;
    ///most represented color seeds
    std::vector<cv::Vec3b> m_Seeds;
    ///seeds grouping
    std::vector<std::vector<cv::Vec3b>> m_SeedClusters;
};

#endif

