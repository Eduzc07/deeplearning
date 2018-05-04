#include "ClusterMoments.h"

#include <cstdio>
#include <QDebug>
#include <QStringList>
#include <QFileInfo>

void ClusterMoments::execute() {
    readFeatures();
    generateClusters();
//     displayClusters();
}

void ClusterMoments::displayClusters(bool withMeansVars)
{
    int count = 1;
    for (unsigned int i = 0; i < m_Clusters.size(); i++) {
        auto c = m_Clusters[i];
        printf("Cluster %d size %u\n", count, c.second.size());
        for (unsigned int i = 0; i < c.first.size(); i++) {
            for (unsigned int j = 0; j < c.first[i].size(); j++)
                printf("%f ", c.first[i][j]);
            printf("\n");
        }
//         if (c.second.size() == 1)
        for (QString file : c.second)
            printf("%s\n", qPrintable(file));            
        printf("\n");
        
        if (withMeansVars) {
            printf("Means \n");
            auto cm = m_ClustersMeanVariance[i].first;
            for (unsigned int i = 0; i < cm.size(); i++) {
                for (unsigned int j = 0; j < cm[i].size(); j++)
                    printf("%f ", cm[i][j]);
                printf("\n");
            }
            printf("\n");
            printf("Variance \n");
            auto cv = m_ClustersMeanVariance[i].second;
            for (unsigned int i = 0; i < cv.size(); i++) {
                for (unsigned int j = 0; j < cv[i].size(); j++)
                    printf("%f ", cv[i][j]);
                printf("\n");
            }
            printf("\n");
        }
        
        count++;
    }
}

void ClusterMoments::displayClusterCenters()
{
    int count = 1;
    for (auto c: m_ClusterCenters) {
        printf("Cluster center %d \n", count);
        for (unsigned int i = 0; i < c.size(); i++) {
            for (unsigned int j = 0; j < c[i].size(); j++)
                printf("%f ", c[i][j]);
            printf("\n");
        }
        printf("\n");
        count++;
    }
}
void ClusterMoments::displayFeature(const std::vector<std::vector<double>>& f)
{
    printf("Feature\n");
    for (unsigned int i = 0; i < f.size(); i++) {
        for (unsigned int j = 0; j < f[i].size(); j++)
            printf("%f ", f[i][j]);
        printf("\n");
    }
}


void ClusterMoments::readFeatures() {
    if (!QFileInfo(m_MomentsFile).exists()) {
        qDebug() << "File " << m_MomentsFile << " does not exist! ";
        exit(1);
    }
    
    QFile* momFile = new QFile(m_MomentsFile);
    
    if (momFile->open(QIODevice::ReadOnly))
    {
        std::vector<std::vector<double>> currentFeature;
        QString currentFile;
        QTextStream in(momFile);
        while (!in.atEnd())
        {
            QString line = in.readLine();
            if (line.contains("cucu")) { ///line with the path of the image
                if (currentFeature.empty()) {
                    currentFile = line.trimmed();
                } else {
                    m_FeatureMap[currentFile] = currentFeature;
                    currentFeature.clear();
                    currentFile = line.trimmed();
                }
            } else {
                QStringList tokens = line.trimmed().split(' ', QString::SkipEmptyParts);
                if (tokens.size() != 10)
                    continue;
                std::vector<double> f;
                for (auto t : tokens) {
                    f.push_back(t.toDouble());
                }
                currentFeature.push_back(f);
            }
        }
        momFile->close();
    }
    delete momFile;
}

void ClusterMoments::generateClusters()
{
    for (auto d : m_FeatureMap) {
        
//         displayClusters();
        
        ///extract only area and center position from the the feature
        std::vector<std::vector<double>> f = extractBasicFeatures(d.second);
//         displayFeature(f);
        
        if (m_Clusters.empty()) {
            ///if no clusters create one
            createNewCluster(f, d.first);
            continue;
        } 
        
        ///search through all clusters where to add the new element
        
        ///compute the cluster centers
        computeClusterCenters();    
//         displayClusterCenters();

        
        ///find the closest cluster center
        bool found = false;
        int minIdx = -1;
        double minDist = 1000000;
        printf("F size %zu Clusters %zu Cluster centers %zu\n", f.size(), m_Clusters.size(), m_ClusterCenters.size());
        for (unsigned int j = 0; j < m_ClusterCenters.size(); j++) {
            if (f.size() != m_ClusterCenters[j].size())
                continue;
            
            double dist = 0.0;
            for (unsigned int i = 0; i < f.size(); i++) {
                for (unsigned int k = 0; k < f[i].size(); k++)
                    dist += std::abs(m_ClusterCenters[j][i][k] - f[i][k]);
            }
            printf("Dist %d = %f \n", j, dist);
            if (dist < minDist) {
                minDist = dist;
                minIdx = j;
                found = true;
            }
        }

        ///if not found start new cluster
        if (!found) {
            createNewCluster(f, d.first);
            continue;
        }

        printf("Closest cluster center %d\n", minIdx);
//         displayClusters();
//         displayClusterCenters();
        
        ///if found check that the distance to the cluster centers respects the required tolerances
        bool withinTolerances = true;
        for (unsigned int i = 0; i < f.size(); i++) {
            if (std::abs(m_ClusterCenters[minIdx][i][0] - f[i][0]) > m_AreaTol)
                withinTolerances = false;
            if (std::abs(m_ClusterCenters[minIdx][i][1] - f[i][1]) > m_PosTol) 
                withinTolerances = false;
            if (std::abs(m_ClusterCenters[minIdx][i][2] - f[i][2]) > m_PosTol) 
                withinTolerances = false;                        
            if (!withinTolerances)
                break;
        }
        
        printf("Within tolerances %s\n", withinTolerances ? "true" : "false");
//         getchar();
        
        //if not within tolerances start new cluster
        if (!withinTolerances) {
            createNewCluster(f, d.first);
            continue;                
        }
        
        //if cluster center closest to element found - add the new element to the respective cluster
        //update the sum in the the first element of the cluster center
        printf("Adding to cluster:\n");
//         displayFeature(m_Clusters[minIdx].first);
        auto& sum = m_Clusters[minIdx].first;
        for(unsigned int i = 0; i < sum.size(); i++) {
            for (unsigned int j = 0; j < sum[i].size(); j++) {
                sum[i][j] += f[i][j];
            }
        }
        //add the file to the list of files in the second element of the cluster center
        m_Clusters[minIdx].second.push_back(d.first);
    }
    
    replaceSumWithMeanClusters();
    std::sort(m_Clusters.begin(), m_Clusters.end(), [](const std::pair<std::vector<std::vector<double>>, QStringList>& a, const std::pair<std::vector<std::vector<double>>, QStringList>& b) -> bool { return a.second.size() > b.second.size(); });
    computeClustersMeansVariances();
    displayClusters(true);
    printf("Clustered %zu elements\n", m_FeatureMap.size());
}


std::vector<std::vector<double> > ClusterMoments::extractBasicFeatures(const std::vector<std::vector<double> >& fullFeature)
{
    std::vector<std::vector<double>> ret;
    for (auto v : fullFeature) {
        if (v[0] < m_AreaThresh)
            continue;
        std::vector<double> ff;
        ff.push_back(v[0]);
        ff.push_back(v[1]);
        ff.push_back(v[2]);
        ret.push_back(ff);
    }
    return ret;
}

void ClusterMoments::createNewCluster(const std::vector<std::vector<double> >& f, const QString& file)
{
    QStringList list;
    list.push_back(file);
    m_Clusters.push_back(std::make_pair(f, list));
}

void ClusterMoments::computeClusterCenters()
{
    m_ClusterCenters.clear();
    for (unsigned i = 0; i < m_Clusters.size(); i++) {
        std::vector<std::vector<double>> center = m_Clusters[i].first; 
        int clusterSize = m_Clusters[i].second.size();
        if (!clusterSize)
            clusterSize = 1;
        for (std::vector<double>& centeri : center) {
            for (double& centerij : centeri)
                centerij /= clusterSize;
        }
    m_ClusterCenters.push_back(center);
    }
}

void ClusterMoments::replaceSumWithMeanClusters()
{
    for (auto& c : m_Clusters) {
        std::vector<std::vector<double>> center = c.first;
        int clusterSize = c.second.size();
        if (!clusterSize)
            clusterSize = 1;
        for (std::vector<double>& centeri : center) {
            for (double& centerij : centeri)
                centerij /= clusterSize;
        }
        c.first = center;
    }    
}

void ClusterMoments::computeClustersMeansVariances()
{
    for (auto& c : m_Clusters) {
        unsigned int fNo = c.first.size();
        
        ///Firstly compute the means
        std::vector<std::valarray<double>> sums;
        ///initialization        
        for (unsigned int i = 0; i < fNo; i++) {
            std::valarray<double> v(0.0, 10);
            sums.push_back(v);
        }
        
        ///compute sum
        for (QString file : c.second) {
            std::vector<std::vector<double>> feature = m_FeatureMap[file];
            for (unsigned int i = 0; i < fNo; i++) {
                std::valarray<double> v(0.0, 10);
                v[0] = feature[i][0];
                v[1] = feature[i][1];
                v[2] = feature[i][2];
                v[3] = feature[i][3];
                v[4] = feature[i][4];
                v[5] = feature[i][5];
                v[6] = feature[i][6];
                v[7] = feature[i][7];
                v[8] = feature[i][8];
                v[9] = feature[i][9];
                sums[i] += v;
            }
        }
        
        ///compute mean
        int clusterSize = c.second.size();
        if (clusterSize == 0)
            clusterSize = 1;
        
        std::vector<std::valarray<double>> means;
        for (unsigned int i = 0; i < sums.size(); i++) {
            std::valarray<double> v = sums[i]/double(clusterSize);
            means.push_back(v);
        }

        ///Second compute the variances
        sums.clear();
        ///initialization        
        for (unsigned int i = 0; i < fNo; i++) {
            std::valarray<double> v(0.0, 10);
            sums.push_back(v);
        }
    
        ///compute sum
        for (QString file : c.second) {
            std::vector<std::vector<double>> feature = m_FeatureMap[file];
            for (unsigned int i = 0; i < fNo; i++) {
                std::valarray<double> v(0.0, 10);
                v[0] = feature[i][0];
                v[1] = feature[i][1];
                v[2] = feature[i][2];
                v[3] = feature[i][3];
                v[4] = feature[i][4];
                v[5] = feature[i][5];
                v[6] = feature[i][6];
                v[7] = feature[i][7];
                v[8] = feature[i][8];
                v[9] = feature[i][9];
                sums[i] += (v - means[i])*(v - means[i]);
            }
        }

        std::vector<std::valarray<double>> vars;
        for (unsigned int i = 0; i < sums.size(); i++) {
            std::valarray<double> v = sqrt(sums[i]/double(clusterSize));
            vars.push_back(v);
        }

        m_ClustersMeanVariance.push_back(std::make_pair(means, vars));
    }
    
}


