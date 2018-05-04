#include "HandleRecognitionSegment.h"

#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <opencv2/core.hpp>
#include <QDebug>
#include <vector>

int HandleRecognitionSegment::generateEdge(const cv::Mat& mat, bool horizontal, bool first, std::vector< cv::Point >& edgePoints, bool verbose) const
{
    //edgePoints.clear();
    if (verbose)
        qDebug() << "generateEdge" << m_Left << m_Right << m_Top << m_Bottom;

    int count = 0;
    int count1 = 0;

    if (horizontal) {
        if (first) { ///top edge
            int starti = m_Left;
            int endi = m_Right;
            int startj = m_Top;
            int endj = m_Bottom;

            for (int i = starti; i <= endi; i++)
                for (int j = startj; j <= endj; j++)   {
                    if (mat.at<cv::Vec3b>(j, i) == m_ColorSegImage) {
                        edgePoints.push_back(cv::Point(i, j));
                        count++;
                        break;
                    } else if (mat.at<cv::Vec3b>(j, i) != cv::Vec3b(220, 220, 220) && verbose) {
                        qDebug() << "Suspicious colour in image " << i << j  << mat.at<cv::Vec3b>(j, i)[0] << mat.at<cv::Vec3b>(j, i)[1] << mat.at<cv::Vec3b>(j, i)[2];
                    }

                    count1++;
                }
        } else { ///bottom edge
            int starti = m_Left;
            int endi = m_Right;
            int startj = m_Bottom;
            int endj = m_Top;

            for (int i = starti; i <= endi; i++)
                for (int j = startj; j >= endj; j--) {
                    if (mat.at<cv::Vec3b>(j, i) == m_ColorSegImage) {
                        edgePoints.push_back(cv::Point(i, j));
                        count++;
                        break;
                    }  else if (mat.at<cv::Vec3b>(j, i) != cv::Vec3b(220, 220, 220) && verbose) {
                        qDebug() << "Suspicious colour in image " << i << j << mat.at<cv::Vec3b>(j, i)[0] << mat.at<cv::Vec3b>(j, i)[1] << mat.at<cv::Vec3b>(j, i)[2];
                    }

                    count1++;
                }
        }
    } else {
        if (first) { ///left edge
            int starti = m_Top;
            int endi = m_Bottom;
            int startj = m_Left;
            int endj = m_Right;

            for (int i = starti; i <= endi; i++)
                for (int j = startj; j <= endj; j++) {
                    if (mat.at<cv::Vec3b>(i, j) == m_ColorSegImage) {
                        edgePoints.push_back(cv::Point(j, i));
                        count++;
                        break;
                    } else if (mat.at<cv::Vec3b>(i, j) != cv::Vec3b(220, 220, 220) && verbose) {
                        qDebug() << "Suspicious colour in image " << i << j << mat.at<cv::Vec3b>(j, i)[0] << mat.at<cv::Vec3b>(j, i)[1] << mat.at<cv::Vec3b>(j, i)[2];
                    }

                    count1++;
                }
        } else {  ///right edge
            int starti = m_Top;
            int endi = m_Bottom;
            int startj = m_Right;
            int endj = m_Left;

            for (int i = starti; i <= endi; i++)
                for (int j = startj; j >= endj; j--) {
                    if (mat.at<cv::Vec3b>(i, j) == m_ColorSegImage) {
                        edgePoints.push_back(cv::Point(j, i));
                        count++;
                        break;
                    } else if (mat.at<cv::Vec3b>(i, j) != cv::Vec3b(220, 220, 220) && verbose) {
                        qDebug() << "Suspicious colour in image " << i << j << mat.at<cv::Vec3b>(j, i)[0] << mat.at<cv::Vec3b>(j, i)[1] << mat.at<cv::Vec3b>(j, i)[2];
                    }

                    count1++;
                }
        }
    }

    return count1;
}

void HandleRecognitionSegment::generateContourPoints(const cv::Mat& mat, std::vector< cv::Point >& contourPoints) const
{
    std::vector<cv::Point> topEdge;
    generateEdge(mat, true, true, contourPoints);
    std::vector<cv::Point> bottomEdge;
    generateEdge(mat, true, false, contourPoints);
    std::vector<cv::Point> leftEdge;
    generateEdge(mat, false, true, contourPoints);
    std::vector<cv::Point> rightEdge;
    generateEdge(mat, false, false, contourPoints);

    contourPoints.erase(std::unique(contourPoints.begin(), contourPoints.end()), contourPoints.end());
}

cv::Point HandleRecognitionSegment::centerMass(const cv::Mat& mat)
{
    int starti = m_Left;
    int endi = m_Right;
    int startj = m_Top;
    int endj = m_Bottom;

    long int sumX = 0;
    long int sumY = 0;
    int noPoints = 0;

    for (int i = starti; i <= endi; i++)
        for (int j = startj; j <= endj; j++) {
            if (mat.at<cv::Vec3b>(j, i) == m_ColorSegImage) {
                sumX += i;
                sumY += j;

                noPoints++;
            }
        }

    double centerX = (double)sumX / (double)noPoints;
    double centerY = (double)sumY / (double)noPoints;

    m_NoPoints = noPoints;
    return cv::Point((int)centerX, (int)centerY);
}


void HandleRecognitionSegment::centerMassHalf(const cv::Mat& mat, const cv::Point& centerMass, double angleVert, cv::Point& centerMassLeft, cv::Point& centerMassRight) const
{
    int starti = m_Left;
    int endi = m_Right;
    int startj = m_Top;
    int endj = m_Bottom;

    long int sumXLeft = 0;
    long int sumYLeft = 0;
    int noPointsLeft = 0;

    long int sumXRight = 0;
    long int sumYRight = 0;
    int noPointsRight = 0;

    for (int i = starti; i <= endi; i++)
        for (int j = startj; j <= endj; j++) {
            if (mat.at<cv::Vec3b>(j, i) == m_ColorSegImage) {
                if (pointInLeftPlane(cv::Point(i, j), centerMass, angleVert)) {
                    sumXLeft += i;
                    sumYLeft += j;
                    noPointsLeft++;
                } else {
                    sumXRight += i;
                    sumYRight += j;
                    noPointsRight++;
                }
            }
        }

    double centerXLeft = (double)sumXLeft / (double)noPointsLeft;
    double centerYLeft = (double)sumYLeft / (double)noPointsLeft;

    double centerXRight = (double)sumXRight / (double)noPointsRight;
    double centerYRight = (double)sumYRight / (double)noPointsRight;

    centerMassLeft = cv::Point(centerXLeft, centerYLeft);
    centerMassRight = cv::Point(centerXRight, centerYRight);
}

double HandleRecognitionSegment::scoreSymmetry1(const cv::Mat& mat, const cv::Point& centerMass, double symmetryAngle) const
{
    int starti = m_Left;
    int endi = m_Right;
    int startj = m_Top;
    int endj = m_Bottom;

    int countSymmetric = 0;
    int noPoints = 0;

    for (int i = starti; i <= endi; i++)
        for (int j = startj; j <= endj; j++) {
            if (mat.at<cv::Vec3b>(j, i) == m_ColorSegImage) {
                noPoints++;
                double x2 = 0.0, y2 = 0.0;

                calculateSymmetric(i, j, centerMass.x, centerMass.y, symmetryAngle, x2, y2);

                int i1 = (int)x2;
                int j1 = (int)y2;

                if (i1 <= starti || i1 >= endi || j1 <= startj || j1 >= endj)
                    continue;

//             if (i1 <= 0 || i1 >= mat.cols || j1 <= 0 || j1 >= mat.rows)
//                 continue;

//             mat.at<Vec3b>(j1, i1) = m_ColorSegImage;

                if (mat.at<cv::Vec3b>(j1, i1) == m_ColorSegImage)
                    countSymmetric++;
            }
        }

    return (double)countSymmetric / (double)noPoints;
}


double HandleRecognitionSegment::scoreSymmetry2(const cv::Mat& mat, cv::Point centerMass, double symmetryAngle) const
{
    cv::Point centerMassLeft(0, 0);
    cv::Point centerMassRight(0, 0);
    centerMassHalf(mat, centerMass, symmetryAngle, centerMassLeft, centerMassRight);

    if (fabs(double(centerMassRight.x - centerMassLeft.x)) < 1)
        return 0.0;

    double horizAngle = atan((double)(centerMassRight.y - centerMassLeft.y) / fabs(double(centerMassRight.x - centerMassLeft.x)));

    return (M_PI / 2.0 - fabs(horizAngle - symmetryAngle)) / M_PI * 2.0;
}


double HandleRecognitionSegment::checkRotationUp(const cv::Mat& mat)
{
    int posTouch = 0;

    for (int i = m_Left; i <= m_Right; i++) {
        if (mat.at<cv::Vec3b>(m_Top, i) == m_ColorSegImage) {
            posTouch = i;
        }
    }

    if ((posTouch > (m_Right - m_Left) / 4) + m_Left && (posTouch < m_Left + (m_Right - m_Left) * 3 / 4))
        return 0.0;

    double minAngle = M_PI;

    if (posTouch <= m_Left + (m_Right - m_Left) / 4) {
        for (int i = m_Left + (m_Right - m_Left) * 3 / 4; i < m_Right; i++) {
            int top = 0;

            for (int j = m_Top; j <= m_Bottom; j++) {
                if (mat.at<cv::Vec3b>(j, i) == m_ColorSegImage) {
                    top = j;
                    break;
                }
            }

            double angle = atan((double)(top - m_Top) / (double)(i - posTouch));

            if (angle < minAngle)
                minAngle = angle;
        }
    } else {
        for (int i = m_Left; i <= m_Left +  + (m_Right - m_Left) / 4; i++) {
            int top = 0;

            for (int j = m_Top; j <= m_Bottom; j++) {
                if (mat.at<cv::Vec3b>(j, i) == m_ColorSegImage) {
                    top = j;
                    break;
                }
            }

            double angle = atan((double)(top - m_Top) / (double)(posTouch - i));

            if (angle < minAngle)
                minAngle = angle;
        }

        minAngle = -minAngle;
    }

    return minAngle;
}

double HandleRecognitionSegment::checkRotationDown(const cv::Mat& mat)
{
    int posTouch = 0;

    for (int i = m_Left; i <= m_Right; i++) {
        if (mat.at<cv::Vec3b>(m_Bottom, i) == m_ColorSegImage) {
            posTouch = i;
        }
    }

    if ((posTouch > (m_Right - m_Left) / 4) + m_Left && (posTouch < m_Left + (m_Right - m_Left) * 3 / 4))
        return 0.0;

    double minAngle = M_PI;

    if (posTouch <= m_Left + (m_Right - m_Left) / 4) {
        for (int i = m_Left + (m_Right - m_Left) * 3 / 4; i < m_Right; i++) {
            int bottom = 0;

            for (int j = m_Bottom; j >= m_Top; j--) {
                if (mat.at<cv::Vec3b>(j, i) == m_ColorSegImage) {
                    bottom = j;
                    break;
                }
            }

            double angle = atan((double)(m_Bottom - bottom) / (double)(i - posTouch));

            if (angle < minAngle)
                minAngle = angle;
        }

        minAngle = - minAngle;
    } else {
        for (int i = m_Left; i <= m_Left +  + (m_Right - m_Left) / 4; i++) {
            int bottom = 0;

            for (int j = m_Bottom; j >= m_Top; j--) {
                if (mat.at<cv::Vec3b>(j, i) == m_ColorSegImage) {
                    bottom = j;
                    break;
                }
            }

            double angle = atan((double)(m_Bottom - bottom) / (double)(posTouch - i));

            if (angle < minAngle)
                minAngle = angle;
        }
    }

    return minAngle;
}

double HandleRecognitionSegment::scoreHoles(const cv::Mat& mat)
{
    int starti = m_Left;
    int endi = m_Right;
    int startj = m_Top;
    int endj = m_Bottom;

    int noCountedLines = 0;
    double sumScores = 0.0;

    for (int i = starti; i <= endi; i++) {
        int index = startj;

        while (mat.at<cv::Vec3b>(index, i) != m_ColorSegImage && index < endj)
            index++;

        if (index == endj)
            continue;

        int startj = index;
        index = endj;

        while (mat.at<cv::Vec3b>(index, i) != m_ColorSegImage && index > startj)
            index--;

        if (index == startj)
            continue;

        int endj = index;
        noCountedLines++;
        int countSpaces = 0;
        bool startHole = false;
        bool holeCounted = 0;

        for (int j = startj; j <= endj; j++) {
            if (mat.at<cv::Vec3b>(j, i) != m_ColorSegImage) {
                if (startHole)
                    holeCounted++;

                startHole = true;
            } else {
                if (startHole && holeCounted)
                    countSpaces += holeCounted + 1;

                startHole = false;
                holeCounted = 0;
            }
        }

        if (countSpaces > (endj - startj + 1) / 10)
            sumScores += 1.0;
        else
            sumScores += 1.0 - (double)countSpaces * 10.0 / (double)(endj - startj + 1);
    }

    return sumScores / (double)noCountedLines;
}

double HandleRecognitionSegment::scoreContour(const cv::Mat& mat, std::vector< QPoint >& smoothedContour)
{
//     int code = rand()%100;

    int starti = m_Left;
    int endi = m_Right;
    int startj = m_Top;
    int endj = m_Bottom;

    cv::Mat contourMat = cv::Mat::zeros(mat.size(), CV_8UC1);

    for (int i = starti; i <= endi; i++)
        for (int j = startj; j <= endj; j++) {
            if (mat.at<cv::Vec3b>(j, i) == m_ColorSegImage) {
                contourMat.at<uchar>(j, i) = 255;
            }
        }



    std::vector<std::vector<cv::Point>> contourParts;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(contourMat, contourParts, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    std::vector<std::vector<cv::Point>> contourPartsStl;

    for (int i = 0; i < (int)contourParts.size(); i++)
        contourPartsStl.push_back(contourParts[i]);

    std::vector<std::vector<cv::Point>>::iterator it = contourPartsStl.begin();

    while (it != contourPartsStl.end()) {
        if (cv::contourArea(*it) < 100) {
            it = contourPartsStl.erase(it);
            continue;
        }

        ++it;
    }

    int smoothLevel = 4;

    std::vector<cv::Point> mainContour;

    if (contourPartsStl.size())
        mainContour = contourPartsStl[0];

    if (mainContour.size() < 30)
        return -1.0;
//     cv::Mat contourMat1 = cv::Mat::zeros(mat.size(), CV_8UC1);

    double smoothSumX = 0.0;
    double smoothSumY = 0.0;
    double smoothX = 0.0;
    double smoothY = 0.0;

    for (int j = 1; j <= smoothLevel; j++) {
        smoothSumX += (double)mainContour[j % mainContour.size()].x;
        smoothSumY += (double)mainContour[j % mainContour.size()].y;
        smoothSumX += (double)mainContour[(-j + mainContour.size()) % mainContour.size()].x;
        smoothSumY += (double)mainContour[(-j + mainContour.size()) % mainContour.size()].y;
    }

    smoothSumX += (double)mainContour[0].x;
    smoothSumY += (double)mainContour[0].y;
    smoothX = smoothSumX / (2.0 * (double)smoothLevel + 1.0);
    smoothY = smoothSumY / (2.0 * (double)smoothLevel + 1.0);
    smoothedContour.push_back(QPoint(floor(smoothX), floor(smoothY)));
//     contourMat1.at<char>(floor(smoothY), floor(smoothX)) = 127;

    for (int i = 1; i < (int)mainContour.size(); i++) {
        smoothSumX += (double)mainContour[(i + smoothLevel) % mainContour.size()].x;
        smoothSumY += (double)mainContour[(i + smoothLevel) % mainContour.size()].y;
        smoothSumX -= (double)mainContour[(i - smoothLevel - 1 + mainContour.size()) % mainContour.size()].x;
        smoothSumY -= (double)mainContour[(i - smoothLevel - 1 + mainContour.size()) % mainContour.size()].y;
        smoothX = smoothSumX / (2.0 * (double)smoothLevel + 1.0);
        smoothY = smoothSumY / (2.0 * (double)smoothLevel + 1.0);
        smoothedContour.push_back(QPoint(floor(smoothX), floor(smoothY)));
//         contourMat1.at<char>(floor(smoothY), floor(smoothX)) = 127;
    }

//         qDebug() << "-------------------------------------";
//     cv::namedWindow(qPrintable(QString("Contour")+QString::number(code)), CV_WINDOW_KEEPRATIO);
//     imshow(qPrintable(QString("Contour")+QString::number(code)), contourMat1);

    return smoothedContour.size();
}

void calculateSymmetric(double x, double y, double x0, double y0, double symmetryAngle, double& x2, double& y2)
{
    double x1 = x0, y1 = y;

    if (fabs(tan(symmetryAngle)) > 0.01) {
        x1 = ((y - y0) + x * tan(symmetryAngle) + x0 / tan(symmetryAngle)) / (tan(symmetryAngle) + 1 / tan(symmetryAngle));
        y1 = ((x - x0) + y / tan(symmetryAngle) + y0 * tan(symmetryAngle)) / (tan(symmetryAngle) + 1 / tan(symmetryAngle));
    }

    x2 = 2 * x1 - x;
    y2 = 2 * y1 - y;
}


bool pointInLowerPlane(const cv::Point& point, const cv::Point lineRefPoint, double angleHoriz)
{
    double val1 = (double)point.y - (double)point.x * tan(angleHoriz);
    double val2 = (double)lineRefPoint.y - (double)lineRefPoint.x * tan(angleHoriz);

    return val1 > val2;
}


bool pointInLeftPlane(const cv::Point& point, const cv::Point lineRefPoint, double angleVert)
{
    double val1 = (double)point.x - (double)point.y * tan(angleVert);
    double val2 = (double)lineRefPoint.x - (double)lineRefPoint.y * tan(angleVert);

    return val1 < val2;
}

