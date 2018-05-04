#include "KMeansIconGraphicsItem.h"
#include "OpenCVKMeansClusterer.h"
#include "OpenCVEMClusterer.h"
#include "RGBImageClusteringSegmenter.h"
#include "GenericKMeansImageClusterer.h"
#include "VectorDouble.h"
#include "MostRepresentedColors.h"
#include "CVMatQImageConversion.h"

QImage KMeansIconGraphicsItem::imageTransform(const QImage& img)
{
    cv::Mat cvImg = CVMatQImageConversion::QImage2Mat(img);
    RGBImageClusterer* clusterer = nullptr;

    switch (m_SegMethod) {
        case 0: 
            clusterer = new OpenCVKMeansClusterer(cvImg, m_NoClusters);
            break;
        case 1:
            clusterer = new OpenCVEMClusterer(cvImg, m_NoClusters);
            break;
        case 2:
            clusterer = new MostRepresentedColors(cvImg, 30, 5.0 * (m_NoClusters - 1));
            break;
        case 3:
            clusterer = new GenericKMeansImageClusterer<VectorDouble>(cvImg, m_NoClusters, 1.0);
            break;
        default:
            break;
    }

//     GenericKMeansImageClusterer<VectorDouble> gkic(cvImg, 5, 0.2);    
//     OpenCVKMeansClusterer omkc(cvImg, 5);
//     OpenCVEMClusterer oemc(cvImg, 5);

//     RGBImageClusteringSegmenter rics(&oemc);
//     RGBImageClusteringSegmenter rics(&omkc);
//     RGBImageClusteringSegmenter rics(&gkic);    

    RGBImageClusteringSegmenter rics(clusterer);

    std::map< cv::Vec3b, ImageSegment, LessVec3b > segments;
    ///@todo: should test if the segmentation was successfull
    rics.execute(segments, cvImg);
    QImage retImg = CVMatQImageConversion::Mat2QImage(cvImg);

    return retImg;
}


