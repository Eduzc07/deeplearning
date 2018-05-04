#ifndef _CLUSTERMOMENTS_
#define _CLUSTERMOMENTS_

#include <QString>
#include <map>
#include <vector>
#include <valarray>

class ClusterMoments {
public:    
    ClusterMoments(const QString& momentsFile): m_MomentsFile(momentsFile) {}
    void execute();
    
private:
    void readFeatures();
    void generateClusters();
    
    std::vector<std::vector<double>> extractBasicFeatures(const std::vector<std::vector<double>>& fullFeature);
    void createNewCluster(const std::vector<std::vector<double>>& f, const QString& file);
    void computeClusterCenters();
    void computeClustersMeansVariances();
    void replaceSumWithMeanClusters();
    void displayClusters(bool withMeansVars);
    void displayFeature(const std::vector<std::vector<double>>& f);
    void displayClusterCenters();
    
private:
    QString m_MomentsFile;
    std::map<QString, std::vector<std::vector<double>>> m_FeatureMap;
    
    double m_AreaTol = 0.1;
    double m_PosTol = 0.1;
    double m_AreaThresh = 0.01;
    
    std::vector<std::pair<std::vector<std::vector<double>>, QStringList>> m_Clusters;
    std::vector<std::vector<std::vector<double>>> m_ClusterCenters;
    std::vector<std::pair<std::vector<std::valarray<double>>, std::vector<std::valarray<double>>>> m_ClustersMeanVariance;
};












#endif
