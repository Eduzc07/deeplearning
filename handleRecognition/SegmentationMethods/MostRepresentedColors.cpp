#include "MostRepresentedColors.h"

#include <vector>

void MostRepresentedColors::initClusterCenters()
{
    ///compute image color histogram
    calculateImageHistogram();
    ///calculate most represented color seeds
    calculateSeeds();
    ///cluster the seeds into groups
    clusterSeeds();

    ///for each seed cluster generate a cluster center
    ///@todo test for when no clusters were found
    m_ClusterNo = m_SeedClusters.size();
//     qDebug() << "Cluster number : " << m_ClusterNo;
    for (auto cluster : m_SeedClusters) {
//         qDebug() << "Cluster ";
        QString clustString;
        VectorDouble center(0.0, 3);
        for (auto col : cluster) {
            clustString += "(" + QString::number(col[0]) + "," + QString::number(col[1]) + "," + QString::number(col[2]) + ")";
            center = center + VectorDouble(col, std::make_pair(0, 0));
        }
        center = center / cluster.size();
        m_ClusterCenters.push_back(center);
//         qDebug() << "Cluster : " << clustString;
//         qDebug() << "Cluster : " << center.getComp(0) << " " << center.getComp(1) << " " << center.getComp(2);
    }
}

void MostRepresentedColors::calculateImageHistogram()
{
    for (int i = 50; i < m_InputImage.rows - 50; i++)
        for (int j = 50; j < m_InputImage.cols - 50; j++) {
            cv::Vec3b color = m_InputImage.at<cv::Vec3b>(i, j);
            if (m_Histo.find(color) != m_Histo.end())
                m_Histo[color]++;
            else
                m_Histo[color] = 1;
        }
}

void MostRepresentedColors::calculateSeeds()
{
    for (std::pair<cv::Vec3b, unsigned int> color : m_Histo) {
        if (color.second > m_HistoThresh) {
//             qDebug() << "histo " << color.first[0] << " " << color.first[1] << " " << color.first[2] << " count " << color.second;
            m_Seeds.push_back(color.first);
//             std::vector<cv::Vec3b> vect;
//             vect.push_back(color.first);
//             m_SeedClusters.push_back(vect);
        }
    }
}

void MostRepresentedColors::clusterSeeds()
{
    if (m_Seeds.empty())
        return;

    for (auto color : m_Seeds) {
        bool firstCluster = false;
        unsigned int firstClusterId = 0;
        std::vector<unsigned int> toMerge;
        for (unsigned int i = 0; i < m_SeedClusters.size(); i++) {
            if (colorBelongsToCluster(color, m_SeedClusters[i])) {
                if (!firstCluster) {
                    m_SeedClusters[i].push_back(color);
                    firstClusterId = i;
                    firstCluster = true;
                } else {
                    toMerge.push_back(i);
                }
            }
        }
        if (!toMerge.empty()) {
            mergeClusters(firstClusterId, toMerge);
        }
        if (toMerge.empty() && !firstCluster) {
            std::vector<cv::Vec3b> vect;
            vect.push_back(color);
            m_SeedClusters.push_back(vect);
        }
    }
}

bool MostRepresentedColors::colorBelongsToCluster(const cv::Vec3b& col, const std::vector<cv::Vec3b>& cluster)
{
    for (auto c : cluster) {
        double dist = sqrt((c[0] - col[0]) * (c[0] - col[0]) + (c[1] - col[1]) * (c[1] - col[1]) + (c[2] - col[2]) * (c[2] - col[2]));
        if (dist < m_DistClustering)
            return true;
    }
    return false;
}

void MostRepresentedColors::mergeClusters(unsigned int i, const std::vector<unsigned int>& toMerge)
{
    std::vector<cv::Vec3b> mergedCluster = m_SeedClusters[i];
    for (auto idx : toMerge) {
        for (auto col : m_SeedClusters[idx]) {
            mergedCluster.push_back(col);
        }
    }
    std::vector<std::vector<cv::Vec3b>> newSeedClusters;
    std::vector<unsigned int> allIndices;
    for (unsigned int i = 0; i < m_SeedClusters.size(); i++) {
        allIndices.push_back(i);
    }

    for (unsigned int i : allIndices) {
        if (std::find(toMerge.begin(), toMerge.end(), i) == toMerge.end()) {
            newSeedClusters.push_back(m_SeedClusters[i]);
        }
    }
    newSeedClusters.push_back(mergedCluster);
    m_SeedClusters = newSeedClusters;
}



/*
cv::Vec3b BackgroundSubstraction::findMostRepresentedColor()
{
    unsigned int countMaxCols = 0;
    cv::Vec3b retVal;

//     double count = m_InputImage.rows * m_InputImage.cols;
    for (auto color : m_Histo) {
        if (color.second > 50) {
            qDebug() << "histo " << color.first[0] << " " << color.first[1] << " " << color.first[2] << " count " << color.second;
        }
        if (color.second > countMaxCols) {
            retVal = color.first;
            countMaxCols = color.second;
        }
    }
    qDebug() << " Most represented color : " << retVal[0] << " " << retVal[1] << " " << retVal[2] << " " << countMaxCols;
    return retVal;
}
*/
