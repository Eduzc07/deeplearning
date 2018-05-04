#include "ImageSegment.h"
#include <QDebug>

void ImageSegment::print() const
{
    printf("Segment \n");
    printf("--------------------\n");
    printf("Average color %f %f %f\n", m_AvgColor[0], m_AvgColor[1], m_AvgColor[2]);
    printf("Label from segmentation %d\n", m_IndexLabelSegmentation);
    printf("Initial color: %d %d %d \n", m_ColorInitialImage[0], m_ColorInitialImage[1], m_ColorInitialImage[2]);
    printf("Color from segmentation %d %d %d \n", m_ColorSegImage[0], m_ColorSegImage[1], m_ColorSegImage[2]);
    printf("Position: left %d right %d top %d bottom %d \n", m_Left, m_Right, m_Top, m_Bottom);
    printf("Number of points: %d \n", m_NoPoints);


    qDebug() << "Segment";
    qDebug() << "--------------------";
    qDebug() << "Average color" << m_AvgColor[0] << m_AvgColor[1] << m_AvgColor[2];
    qDebug() << "Label from segmentation " << m_IndexLabelSegmentation;
    qDebug() << "Initial color: " << m_ColorInitialImage[0] << m_ColorInitialImage[1] << m_ColorInitialImage[2];
    qDebug() << "Color from segmentation " << m_ColorSegImage[0] << m_ColorSegImage[1] << m_ColorSegImage[2];
    qDebug() << "Position: left" <<  m_Left << " right " <<  m_Right << "top" << m_Top << "bottom" << m_Bottom;
    qDebug() << "Number of points: " << m_NoPoints;
}

double ImageSegment::avgHeight(const cv::Mat& mat)
{
    double avgHeight = 0.0;
    int count = 0;

    for (int i = m_Left; i <= m_Right; i++) {
        if (i % 10 == 0) {
            int top = 0;
            int bottom = 0;

            for (int j = m_Top; j <= m_Bottom; j++)
                if (mat.at<cv::Vec3b>(j, i) == m_ColorSegImage) {
                    top = j;
                    break;
                }

            for (int j = m_Bottom; j >= m_Top; j--)
                if (mat.at<cv::Vec3b>(j, i) == m_ColorSegImage) {
                    bottom = j;
                    break;
                }

            avgHeight += (bottom - top);
            count++;
        }
    }

    avgHeight /= (double)count;
    return avgHeight;
}

bool ImageSegment::contains(const cv::Mat& mat, const ImageSegment& seg) const
{
    if (seg.m_Left <= m_Left)
        return false;

    if (seg.m_Top <= m_Top)
        return false;

    if (seg.m_Right >= m_Right)
        return false;

    if (seg.m_Bottom >= m_Bottom)
        return false;

    for (int i = seg.m_Left; i <= seg.m_Right; i++) {
        int topCol = findTopSegmentPoint(mat, i);
        int bottomCol = findBottomSegmentPoint(mat, i);

        int topColSeg = seg.findTopSegmentPoint(mat, i);
        int bottomColSeg = seg.findBottomSegmentPoint(mat, i);

        if (topColSeg < topCol || bottomColSeg > bottomCol)
            return false;
    }

    return true;
}

int ImageSegment::findTopSegmentPoint(const cv::Mat& mat, int col) const
{
    int index = m_Top;

    while (mat.at<cv::Vec3b>(index, col) != m_ColorSegImage && index <= m_Bottom)
        index++;

    if (index == m_Bottom + 1)
        return -1; //should not happen
    else
        return index;
}

int ImageSegment::findBottomSegmentPoint(const cv::Mat& mat, int col) const
{
    int index = m_Bottom;

    while (mat.at<cv::Vec3b>(index, col) != m_ColorSegImage && index >= m_Top)
        index--;

    if (index == m_Top - 1)
        return -1;
    else
        return index;
}

void ImageSegment::markWithNewColor(cv::Mat& destMat, const cv::Vec3b& color) const
{
    int starti = m_Top;
    int endi = m_Bottom;
    int startj = m_Left;
    int endj = m_Right;

    for (int i = starti; i <= endi; i++)
        for (int j = startj; j <= endj; j++) {
            if (destMat.at<cv::Vec3b>(i, j) == m_ColorSegImage)
                destMat.at<cv::Vec3b>(i, j) = color;
        }
}
