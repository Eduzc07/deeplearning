#include "OpenCVClusterer.h"


bool OpenCVClusterer::execute()
{
    if (!clusterColors())
        return false;
    
    buildClusteredImage();
    return true;
}

bool OpenCVClusterer::extractColorsFromImage(cv::Mat& clusterInput)
{
    std::map<cv::Vec3b, int, LessVec3b> colorMap;

    for (int i = 0; i < m_InputImage.rows; i++)
        for (int j = 0; j < m_InputImage.cols; j++) {
            colorMap[m_InputImage.at<cv::Vec3b>(i, j)] = 1;
        }

    unsigned int noColors = colorMap.size();

    if (noColors < m_ClusterNo) {
        return false;
    }

    //prepare for kmeans clustering
    clusterInput = cv::Mat(noColors , 1, CV_32FC3);

    std::map<cv::Vec3b, int, LessVec3b>::iterator it = colorMap.begin();
    int count = 0;

    for (; it != colorMap.end(); ++it) {
        clusterInput.at<cv::Vec3f>(count, 0) = it->first;
        count++;
    }

    return true;
}

void OpenCVClusterer::buildClusteredImage()
{
    //save the clustered image
    m_ClusteredImage = cv::Mat::zeros(m_InputImage.size(), m_InputImage.type());

    for (int i = 0; i < m_ClusteredImage.rows; i++)
        for (int j = 0; j < m_ClusteredImage.cols; j++) {
            unsigned int cluster = m_Labels[m_InputImage.at<cv::Vec3b>(i, j)];
            int temp = 255 - 255 / m_ClusterNo * cluster;
//             qDebug() << "Clustered image " << i << " " << j << " " << temp;
            m_ClusteredImage.at<cv::Vec3b>(i, j) = cv::Vec3b(temp, temp, temp);
        }
}
