#include "OpenCVEMClusterer.h"

#include <QTime>
#include <QDebug>
#include <set>
#include <opencv2/ml/ml.hpp>

bool OpenCVEMClusterer::clusterColors()
{
    m_Labels.clear();
    try {
        QTime* t1 = new QTime();
        t1->start();

        cv::Mat emOutput;
        cv::Mat emInput1;

        if (!extractColorsFromImage(emInput1)) {
            printf("Error color extraction\n");
            return false;
        }
        
        ///for the moment calculate expectation maximization with grey colors
        std::set<unsigned int> greyColors;
        
        for (int i = 0; i < emInput1.rows; i++) {
            cv::Vec3f value = emInput1.at<cv::Vec3f>(i, 0);
            greyColors.insert(transformColorToInt(value));
        }
        cv::Mat emInput = cv::Mat(greyColors.size() , 1, CV_64F);

        int count = 0;
        for (unsigned color1 : greyColors) {
            emInput.at<double>(count, 0) = color1;
            count++;
        }
        
        ///run expectation maximization
        cv::Ptr<cv::ml::EM>  emModel = cv::ml::EM::create();
        emModel->setClustersNumber(m_ClusterNo);
        emModel->setCovarianceMatrixType(cv::ml::EM::COV_MAT_DIAGONAL);
        emModel->setTermCriteria(cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100, 1.0));
        emModel->trainEM(emInput);

        ///find which label corresponds to each of the colors in the original image
        count = 0;
        for (int i = 0; i < emInput1.rows; i++) {            
            cv::Vec3f color = emInput1.at<cv::Vec3f>(i, 0);
            cv::Mat probs;
            int greyColor = transformColorToInt(color);
            cv::Vec2d clasif = emModel->predict(greyColor, probs);
            QString probString;
            
            int maxIdx = 0;
            for (unsigned int i = 0; i < m_ClusterNo; i++) {
                if (probs.at<double>(0, i) > probs.at<double>(0, maxIdx))
                    maxIdx = i;
                probString += QString::number(probs.at<double>(0, i)) + " ";
            }
//             qDebug() << "color : " << greyColor  << "probs: " << probString << "clasif: " << clasif[0] << " - " << clasif[1];
            m_Labels[static_cast<cv::Vec3b>(color)] = int(clasif[1]);

            count++;
        }

        int duration = t1->elapsed();
        qDebug() << "Total expectation maximization" << duration << "ms";

        return true;
    } catch(cv::Exception& e) {
        return false;
    }
}

bool OpenCVEMClusterer::extractColorsFromImage(cv::Mat& clusterInput)
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

int OpenCVEMClusterer::transformColorToInt(const cv::Vec3f& color)
{
    return (color[0] + color[1] + color[2]) / 3;
}
