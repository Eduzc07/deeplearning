#include "CVMatQImageConversion.h"

///adapted from
///http://stackoverflow.com/questions/17127762/cvmat-to-qimage-and-back
QImage CVMatQImageConversion::Mat2QImage(cv::Mat const& src)
{
     cv::Mat temp = src; // make the same cv::Mat
     cv::cvtColor(src, temp, CV_BGR2RGB); // cvtColor Makes a copt, that what i need
     QImage dest((const uchar *) temp.data, temp.cols, temp.rows, temp.step, QImage::Format_RGB888);
     dest.bits(); // enforce deep copy, see documentation
     // of QImage::QImage ( const uchar * data, int width, int height, Format format )
     return dest;
}

cv::Mat CVMatQImageConversion::QImage2Mat(QImage const& src)
{
     QImage src1 = src.convertToFormat(QImage::Format_RGB888);
     cv::Mat tmp(src1.height(), src1.width(), CV_8UC3, const_cast<uchar*>(src1.bits()), src1.bytesPerLine());
     cv::Mat result; // deep copy just in case (my lack of knowledge with open cv)
     cv::cvtColor(tmp, result, CV_BGR2RGB);
     return result;
}
