#include "BackgroundExtractionGraphicsItem.h"
#include "BackgroundCrateHandleSegmenter.h"
#include "CVMatQImageConversion.h"

QImage BackgroundExtractionGraphicsItem::imageTransform(const QImage& img)
{
    cv::Mat cvImg = CVMatQImageConversion::QImage2Mat(img);

    if (m_Method == 0) {
        BackgroundCrateHandleSegmenter bchs(cvImg, cvImg);
        bchs.estimateBackgroundImage();
        QImage retImg = CVMatQImageConversion::Mat2QImage(bchs.getBackgroundCompositionImage());
        return retImg;
    } else if (m_Method == 1) {
        cv::Mat grayImg;
        cv::cvtColor(cvImg, grayImg, CV_BGR2GRAY);
        cv::Mat otsuImg;
        cv::threshold(grayImg, otsuImg, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        cv::Mat output;
        cv::cvtColor(otsuImg, output, CV_GRAY2BGR);
        QImage retImg = CVMatQImageConversion::Mat2QImage(output);
        return retImg;        
    } else {
        ///watersheding transformation with 2 marker areas
        int w = cvImg.cols;
        int h = cvImg.rows;
        cv::Mat mask = cv::Mat::zeros(cvImg.size(), CV_32SC1);
        cv::rectangle(mask, cv::Point(w / 4, h / 4), cv::Point(3 * w / 4, 3 * h / 4),  64, CV_FILLED);
//         cv::rectangle(mask, cv::Point(0, 0), cv::Point(w, 10), 128, CV_FILLED);
        cv::rectangle(mask, cv::Point(0, h - 10), cv::Point(w, h), 128, CV_FILLED);
        cv::rectangle(mask, cv::Point(0, 0), cv::Point(10, h), 128, CV_FILLED);
        cv::rectangle(mask, cv::Point(w - 10, 0), cv::Point(w, h), 128, CV_FILLED);
        cv::watershed(cvImg, mask);        
        cv::Mat output1, output2;
        mask.convertTo(output1, CV_8UC1);
        cv::cvtColor(output1, output2, CV_GRAY2BGR);
        QImage retImg = CVMatQImageConversion::Mat2QImage(output2);
        return retImg; 
    }
}
