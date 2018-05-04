#ifndef _CVMATQIMAGECONVERSION_
#define _CVMATQIMAGECONVERSION_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QImage>

class CVMatQImageConversion {
public:

    CVMatQImageConversion() {}
    static QImage Mat2QImage(cv::Mat const& src);
    static cv::Mat QImage2Mat(QImage const& src);
};

#endif
