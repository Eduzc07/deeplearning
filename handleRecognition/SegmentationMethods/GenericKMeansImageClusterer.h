#ifndef _GENERICKMEANSIMAGECLUSTERING_
#define _GENERICKMEANSIMAGECLUSTERING_

#include <vector>
#include <map>
#include <unordered_map>
#include <QDebug>

/**
 * KMeans clustering of elements associated to indices in an image
 * Uses the brute force method for now (Lloyd's algorithm)
 * @todo: optimize the center initialization method when starting
 * @todo: give the possibility that Kmeans uses input from mean shift 
 * or is the input for expectation maximization
 */

#include "RGBImageClusterer.h"

template <class T> class GenericKMeansImageClusterer : public RGBImageClusterer {
protected:
    ///the elements to be clustered
    ///the index is the position in the image
    std::map<std::pair<unsigned int, unsigned int>, T> m_Elements;
    ///number of elements
    unsigned int m_ElementNo;
    ///number of clusters to be computed
    unsigned int m_ClusterNo;
    ///assignments element index -> clusterNo
    std::map<std::pair<unsigned int, unsigned int>, unsigned int> m_Result;
    ///e.g. only 100*m_StopCondition% elements are reassigned in between iterations
    double m_StopCondition = 0.1;
    ///how many times the clusters are recomputed when the method does not converge
    int m_MaxIterations = 10;
    ///position of the cluster centers
    std::vector<T> m_ClusterCenters;
    ///how much the cluster centers have moved in the last iteration
    double m_CenterMovement = 0.0;

public:
    GenericKMeansImageClusterer(cv::Mat& mat, unsigned int clusterNo, double stopCondition);
    /**
     * Run execute Iteration until stop condition is met or too many iterations
     */
    bool execute();

protected:
    /**
     * Get the values at the cluster centers
     */
    virtual void initClusterCenters();

    /**
     * Get the values at the cluster centers
     */
    bool updateClusterCenters();

    /**
     * Computes to which cluster belongs each point
     */
    void calculateClusters();
    /**
     * builds the clustered image 
     */
    void buildClusteredImage();
};

template <class T> GenericKMeansImageClusterer<T>::GenericKMeansImageClusterer(cv::Mat& mat, unsigned int clusterNo, double stopCondition) : RGBImageClusterer(mat), m_ClusterNo(clusterNo), m_StopCondition(stopCondition)
{
    ///such that the execute loop does not exit directly
    m_CenterMovement = m_StopCondition + 1.0;
    m_ElementNo = m_InputImage.rows * m_InputImage.cols;

    for (int i = 0; i < m_InputImage.cols; i++)
        for (int j = 0; j < m_InputImage.rows; j++) {
            m_Elements[std::make_pair(i, j)] = T(m_InputImage.at<cv::Vec3b>(j, i), std::make_pair(i, j));
        }
}

template <class T> bool GenericKMeansImageClusterer<T>::execute() 
{
    qDebug() << "KMeans start ";
    initClusterCenters();
    if (m_ClusterNo < 2) 
        return false;
    int count = 0;
    while (count < m_MaxIterations && m_CenterMovement > m_StopCondition) {
        calculateClusters();
        if (!updateClusterCenters())
            return false;
        count++;
        qDebug() << "KMeans iteration " << count;
    }
    buildClusteredImage();
    return true;
}

/**
 * First attempt: cluster centers are initialized at egal distances from one another
 * @todo: sort the elements before chosing the cluster centers
 * @todo: must make sure that the centers do not repeat
 */

template <class T> void GenericKMeansImageClusterer<T>::initClusterCenters() 
{
    if (!m_ClusterNo)
        return;
    qDebug() << "Init cluster centers. Image rows : " <<  m_InputImage.rows  << " cols : " << m_InputImage.cols;
    
    int step = ceil(sqrt(m_ClusterNo));
    int offset = step * step - m_ClusterNo;
    int stepY = m_InputImage.rows / step;
    int stepX = m_InputImage.cols / step;
    
    qDebug() << "Step : " << step << " Offset : " << offset << " stepX " << stepX << " stepY : "<< stepY;
    
    for (unsigned int i = 0; i < m_ClusterNo; i++) {
        int id = offset/2 + i;
        int idx = id % step;
        int idy = id / step;
        int indexI = stepX/2 + idx * stepX;
        int indexJ = stepY/2 + idy * stepY;
        m_ClusterCenters.push_back(m_Elements[std::make_pair(indexI, indexJ)]);
        qDebug() << "Cluster " << i << " i: " << indexI << " j " << indexJ << " val " << m_Elements[std::make_pair(indexI, indexJ)].toString();
    }
}

/**
 * Update the cluster center positions and calculate the distances between them
 * @todo: must make sure that the centers do not repeat
 */
template <class T> bool GenericKMeansImageClusterer<T>::updateClusterCenters() 
{
    if (!m_ClusterNo)
        return true;

    qDebug() << "Updated cluster centers";
    
    unsigned int dataDimensions = m_Elements[std::make_pair(0, 0)].getDim();
    double scaleVal = double(m_ElementNo) / double(m_ClusterNo);
    ///count how many members are inside a class
    std::vector<unsigned int> counters(m_ClusterNo, 0);
    ///new center positions
    std::vector<T> values(m_ClusterNo, T(0, dataDimensions));

    for (auto it : m_Elements) {
        unsigned int cluster = m_Result[it.first];
        if (cluster < m_ClusterNo)
            counters[cluster]++;
        values[cluster] = values[cluster] + it.second / scaleVal;  //scaling so that we do not compute with values which are too big
    }

    for (unsigned int i = 0; i < m_ClusterNo; i++) {
        if (!counters[i])
            return false;
        values[i] = values[i] / double(counters[i]);
        values[i] = values[i] * scaleVal;
    }

    ///calculates how much the centers have moved
    ///and updates the center positions
    m_CenterMovement = 0.0;
    for (unsigned int i = 0; i < m_ClusterNo; i++) {
        double dist = values[i].distTo(m_ClusterCenters[i]);
        m_CenterMovement += dist / double(m_ClusterNo);
        m_ClusterCenters[i] = values[i];
    }
    
    
    
    qDebug() << "Center movement " << m_CenterMovement << " Stop condition " << m_StopCondition;
    qDebug() << "updateclustercenters ends";
    
    return true;
}

template <class T> void GenericKMeansImageClusterer<T>::calculateClusters() 
{
    if (!m_ClusterNo)
        return;

    qDebug() << "Calculate clusters";
    std::unordered_map<T, int> hashTable;
    
    for (auto it : m_Elements) {
        auto h = hashTable.find(it.second);
        if (h != hashTable.end()) {
            m_Result[it.first] = h->second;
            continue;
        }
    
        double refVal = it.second.distTo(m_ClusterCenters[0]);
        int minCluster = 0;
        for (unsigned int j = 1; j < m_ClusterNo; j++) {
            double testVal = it.second.distTo(m_ClusterCenters[j]);
            if (testVal < refVal) {
                refVal = testVal;
                minCluster = j;
            }
        }
        m_Result[it.first] = minCluster;
        hashTable[it.second] = minCluster;
    }
    
    qDebug() << "Calculate clusters ends";
}

template <class T> void GenericKMeansImageClusterer<T>::buildClusteredImage()
{
    //save the clustered image
    m_ClusteredImage = cv::Mat::zeros(m_InputImage.size(), m_InputImage.type());

    for (int i = 0; i < m_ClusteredImage.rows; i++)
        for (int j = 0; j < m_ClusteredImage.cols; j++) {
            unsigned int cluster = m_Result[std::make_pair(j, i)];
            int temp = 255 - 255 / m_ClusterNo * cluster;
//             qDebug() << "Clustered image " << i << " " << j << " " << temp;
            m_ClusteredImage.at<cv::Vec3b>(i, j) = cv::Vec3b(temp, temp, temp);
        }
}

#endif
