#include "VertHorizHistoGraphicsItem.h"
#include "VertHorizHistoDB.h"
#include "CVMatQImageConversion.h"

QImage VertHorizHistoGraphicsItem::imageTransform(const QImage& img)
{
    cv::Mat cvImg = CVMatQImageConversion::QImage2Mat(img);    
    VertHorizHistograms vhh(cvImg);
    
    cv::Mat retImg;
    cv::cvtColor(vhh.getBinaryImage(m_Type), retImg, CV_GRAY2BGR);
    
    QImage retImage = CVMatQImageConversion::Mat2QImage(retImg);
    return retImage; 
}
