#include "OpenCVKMeansClusterer.h"

#include <QTime>
#include <QDebug>


bool OpenCVKMeansClusterer::clusterColors()
{
    m_Labels.clear();
    try {
        QTime* t1 = new QTime();
        t1->start();

        cv::Mat centers(m_ClusterNo, 1, CV_32FC3);
        cv::Mat kmeansOutput;
        cv::Mat kmeansInput;

        if (!extractColorsFromImage(kmeansInput)) {
            printf("Error color extraction\n");
            return false;
        }

        ///run kmeans clustering
        cv::kmeans(kmeansInput, m_ClusterNo, kmeansOutput, cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0), 3, cv::KMEANS_PP_CENTERS, centers);

        ///find which label corresponds to each of the colors in the original image
        int count = 0;
        for (int i = 0; i < kmeansInput.rows; i++) {
            m_Labels[static_cast<cv::Vec3b>(kmeansInput.at<cv::Vec3f>(i, 0))] = kmeansOutput.at<int>(count);
            count++;
        }

        int duration = t1->elapsed();
        qDebug() << "Total time kmeans clustering" << duration << "ms";

        return true;
    } catch(cv::Exception& e) {
        return false;
    }
}