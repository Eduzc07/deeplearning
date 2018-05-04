#include "CrateHandle.h"

#include <algorithm>
#include <math.h>

#include <QTime>
#include <QDebug>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

double CrateHandle::m_ThresMaxHandleReject = 0.8;

CrateHandle::CrateHandle(const cv::Mat& handleImage)
{
    m_HandleImage = handleImage;
    std::map<cv::Vec3b, int, LessVec3b> colors;

    for (int i = 0; i < handleImage.rows; i++) {
        for (int j = 0; j < handleImage.cols; j++) {
            cv::Vec3b color = handleImage.at<cv::Vec3b>(i, j);

            if (colors.find(color) != colors.end()) {
                colors[color]++;
            } else {
                colors[color] = 0;
//                 printf("New color found %d %d %d- %d %d\n", color[0], color[1], color[2], i , j);
                
                if (colors.size() > 2) {
                    qDebug() << "More than 2 colors in image";
                    break;
                }
            }
        }
        if (colors.size() > 2) {
            break;
        }
    }

    if (colors.size() == 2) {
        std::map<cv::Vec3b, int, LessVec3b>::iterator it = colors.begin();
        int maxNo = 0;
        std::map<cv::Vec3b, int, LessVec3b>::iterator itMax;

        while (it != colors.end()) {
            if (it->second > maxNo) {
                maxNo = it->second;
                itMax = it;
            }
            ++it;
        }

        colors.erase(itMax);
        m_HandleColors.push_back(colors.begin()->first);
        ImageSegment seg;
        seg.m_AvgColor = (cv::Vec3f)m_HandleColors[0];
        seg.m_ColorSegImage = seg.m_ColorInitialImage = m_HandleColors[0];
        seg.m_NoPoints = colors.begin()->second;

        for (int i = 0; i < handleImage.rows; i++)
            for (int j = 0; j < handleImage.cols; j++) {
                cv::Vec3b color = handleImage.at<cv::Vec3b>(i, j);

                if (color == m_HandleColors[0]) {
                    if (j < seg.m_Left)
                        seg.m_Left = j;

                    if (j > seg.m_Right)
                        seg.m_Right = j;

                    if (i < seg.m_Top)
                        seg.m_Top = i;

                    if (i > seg.m_Bottom)
                        seg.m_Bottom = i;
                }
            }

        m_SegmentsMap[m_HandleColors[0]] = static_cast<HandleRecognitionSegment>(seg);
        m_HandleFeatures.push_back(CrateHandleFeatures());
        score(0);
    }
}

bool CrateHandle::score(int handleNo)
{
    if (!m_HandleImage.rows || !m_HandleImage.cols)
        return false;

//     qDebug() << "No handles " << m_HandleColors.size();
    for (int i = 0; i < (int)m_HandleColors.size(); i++) {
        HandleRecognitionSegment seg = m_SegmentsMap.at(m_HandleColors[i]);

        ///calculate center of mass for segment
        cv::Point centerMass = seg.centerMass(m_HandleImage);
        m_HandleFeatures[i].m_CenterMass = centerMass;

        ///calculate the position of points in the segment in polar coordinates
        std::vector<double> anglesVector;
        double refSymmetryAngle = 0.0;

        ///score the symmetry
        double handleMarginSymmetryTest = m_ThreshNoAngleTests1;
        double targetValue = m_ThreshTargetSymmetryAngle1;

        double symmetryScore1 = 0.0;
        double symmetryAngle1 = refSymmetryAngle;
        double symmetryScore2 = 0.0;
        double symmetryAngle2 = refSymmetryAngle;

        for (int j = 0; j < handleMarginSymmetryTest; j++) {
            double angle = (-targetValue / 2 + targetValue / handleMarginSymmetryTest * (double)j) * M_PI / 180.0 + refSymmetryAngle;
            double score1 = seg.scoreSymmetry1(m_HandleImage, centerMass, angle);

            //qDebug() << "Score 1: " << angle << score1;
            if (score1 > symmetryScore1) {
                symmetryScore1 = score1;
                symmetryAngle1 = angle;
            }

            double score2 = seg.scoreSymmetry2(m_HandleImage, centerMass, angle);

            //qDebug() << "Score 2: " << angle << score2;
            if (score2 > symmetryScore2) {
                symmetryScore2 = score2;
                symmetryAngle2 = angle;
            }
        }

        std::vector<cv::Point> contourPoints;
        seg.generateContourPoints(m_HandleImage, contourPoints);
        cv::RotatedRect rRect = cv::minAreaRect(contourPoints);

        m_HandleFeatures[i].m_SymmetryAxis3 = rRect.angle * M_PI / 180.0 < -1 ? rRect.angle * M_PI / 180.0 + M_PI / 2.0 : rRect.angle * M_PI / 180.0;
        m_HandleFeatures[i].m_SymmetryAxis1 = -symmetryAngle1;
        m_HandleFeatures[i].m_SymmetryAxis2 = symmetryAngle2;

        m_HandleFeatures[i].m_SymmetryAxis4 = seg.checkRotationUp(m_HandleImage);
        m_HandleFeatures[i].m_SymmetryAxis5 = seg.checkRotationDown(m_HandleImage);

        ///score the holes
        m_HandleFeatures[i].m_SegFormScore = seg.scoreHoles(m_HandleImage) * symmetryScore1;
        m_HandleFeatures[i].m_SegContourScore = seg.scoreContour(m_HandleImage, m_HandleFeatures[i].m_Contour);
        ///score the contour
    }

    if (!calculateDistances())
        return false;

    if (!calculateMedianLine())
        return false;

    return true;
}

void CrateHandle::build(const cv::Mat& segmentedImage, const std::map< cv::Vec3b, ImageSegment, LessVec3b >& segments, const std::vector< cv::Vec3b >& handleSegmentColors)
{
    m_HandleImage = cv::Mat::zeros(segmentedImage.size(), segmentedImage.type());

    for (int i = 0; i < segmentedImage.rows; i++)
        for (int j = 0; j < segmentedImage.cols; j++) {
            cv::Vec3b temp = segmentedImage.at<cv::Vec3b>(i, j);
            bool handleColorFound = false;

            for (int k = 0; k < (int)handleSegmentColors.size(); k++) {
                if (handleSegmentColors[k] == temp) {
                    m_HandleImage.at<cv::Vec3b>(i, j) = temp;
                    handleColorFound = true;
                    break;
                }
            }

            ///@todo: it is possible that (220,220,220) to be one of the colors of the handle
            if (!handleColorFound)
                m_HandleImage.at<cv::Vec3b>(i, j) = cv::Vec3b(220, 220, 220);
        }

    m_SegmentsMap.clear();
    for (auto it : segments) {
        m_SegmentsMap[it.first] = HandleRecognitionSegment(it.second);
    }
    m_HandleColors = handleSegmentColors;

    for (int i = 0; i < (int)m_HandleColors.size(); i++) {
        if (m_SegmentsMap.find(m_HandleColors[i]) == m_SegmentsMap.end())
            qDebug() << "Error when building cratehandle - colors";
    }

    m_HandleFeatures.clear();

    for (int i = 0; i < (int)m_HandleColors.size(); i++) {
        m_HandleFeatures.push_back(CrateHandleFeatures());
    }

//     qDebug() << "Handle features " << m_HandleFeatures.size();
}

void CrateHandle::keepUniqueHandle(int idx)
{
    if (!m_HandleImage.rows || !m_HandleImage.cols)
        return;

    if (idx < 0 || idx >= (int)m_HandleColors.size())
        return;


    cv::Vec3b color(127, 127, 127);
    ImageSegment goodSegment = m_SegmentsMap.at(m_HandleColors[idx]);

    for (int i = 0; i < m_HandleImage.cols; i++)
        for (int j = 0; j < m_HandleImage.rows; j++) {
            if (m_HandleImage.at<cv::Vec3b>(j, i) == m_HandleColors[idx])
                m_HandleImage.at<cv::Vec3b>(j, i) = color;
            else
                m_HandleImage.at<cv::Vec3b>(j, i) = cv::Vec3b(220, 220, 220);
        }

    m_HandleColors.clear();
    m_HandleColors.push_back(color);
    CrateHandleFeatures chf = m_HandleFeatures[idx];
    m_HandleFeatures.clear();
    m_HandleFeatures.push_back(chf);
    m_SegmentsMap.clear();
    m_SegmentsMap[color] = goodSegment;
}

bool CrateHandle::detectAndCreate(const cv::Mat& segmentedImage, const std::map< cv::Vec3b, ImageSegment, LessVec3b >& segments)
{
    QTime* t1 = new QTime();
    t1->start();

    ///@TODO: measure the average thickness of the handle and impose restrictions on that

    m_ThresPointNo = segmentedImage.rows * segmentedImage.cols / 20;
    m_ThreshHandleMinDistanceToTop = segmentedImage.rows / 10;
    m_ThreshHandleMinDistanceLateral = segmentedImage.cols / 10;
    m_ThreshHandleMinHeight = segmentedImage.rows / 10;
    m_ThreshHandleMinWidth = segmentedImage.cols / 4;

    std::vector<cv::Vec3b> handleSegmentColors;

    std::map<cv::Vec3b, ImageSegment, LessVec3b>::const_iterator it = segments.begin();

    for (; it != segments.end(); ++it) {
        ImageSegment seg = it->second;
        int distLeft = seg.m_Left;
        int distRight = segmentedImage.cols - seg.m_Right;
        int maxDist = std::max(distLeft, distRight);

        if (maxDist < m_ThreshHandleMinDistanceLateral)
            continue;

        int minDist = std::min(distLeft, distRight);
        float diffDistFact = (float)(maxDist - minDist) / (float)maxDist;

        if (diffDistFact > m_ThreshLateralAssymetry)
            continue;

        if (seg.m_Top < m_ThreshHandleMinDistanceToTop)
            continue;

        if (seg.m_Bottom > segmentedImage.rows - m_ThreshHandleMinDistanceToTop)
            continue;

        if (seg.m_NoPoints < m_ThresPointNo)
            continue;

        double height = seg.avgHeight(segmentedImage);

        if (height < m_ThreshHandleMinHeight)
            continue;

        if (seg.m_Right - seg.m_Left < m_ThreshHandleMinWidth)
            continue;

        //printf("Possible handle has been found\n");
        //printSegment(seg);
        handleSegmentColors.push_back(seg.m_ColorSegImage);
    }

    if (!handleSegmentColors.size()) {
        int duration = t1->elapsed();
        qDebug() << "Time handle recognition" << duration << "ms";
        printf("No grip/handle/butt has been found \n");
        return false;
    }

    build(segmentedImage, segments, handleSegmentColors);

    int duration = t1->elapsed();
    qDebug() << "Time handle recognition" << duration << "ms";

    return true;
}

bool CrateHandle::badScore()
{
    int count = 0;
    std::vector<int> eraseList;
    std::vector<int> keepList;


    for (int i = 0; i < (int)m_HandleFeatures.size(); i++) {
        if (m_HandleFeatures[i].m_SegFormScore >= m_ThresMaxHandleReject) {
            count++;
            keepList.push_back(i);
        } else {
            eraseList.push_back(i);
        }
    }

    if (keepList.size() && eraseList.size()) {
        eraseHandles(keepList, eraseList);
    }

    return (count == 0);
}

bool CrateHandle::badSegmentation()
{
    int count = 0;
    std::vector<int> eraseList;
    std::vector<int> keepList;

    for (int i = 0; i < (int)m_HandleFeatures.size(); i++) {
        if ((fabs((m_HandleFeatures[i].m_SymmetryAxis1 + m_HandleFeatures[i].m_SymmetryAxis2) / 2.0 - m_HandleFeatures[i].m_SymmetryAxis4) < 0.1)
                && (fabs((m_HandleFeatures[i].m_SymmetryAxis1 + m_HandleFeatures[i].m_SymmetryAxis2) / 2.0 - m_HandleFeatures[i].m_SymmetryAxis5) < 0.1)
           && (m_HandleFeatures[i].m_SegContourScore > 0)) {
            count++;
            keepList.push_back(i);
        } else {
            eraseList.push_back(i);
        }
    }


    if (keepList.size() && eraseList.size()) {
        eraseHandles(keepList, eraseList);
    }

    return (count == 0);
}

CrateHandleSegmenter::CrateHandleSegmenter(const cv::Mat& inputImage, const cv::Mat& originalImage, bool shrink, bool cut)
{
    m_Shrink = shrink;
    m_Cut = cut;
    
    cv::Mat supHalfMat;
    cv::Mat supHalfMat1;
    cv::Mat shrinkedInput;
    cv::Mat shrinkedInput1;
    
    
    if (cut) {
        supHalfMat = cv::Mat(inputImage, cv::Range(0, inputImage.rows / 2), cv::Range(0, inputImage.cols));
        supHalfMat1 = cv::Mat(originalImage, cv::Range(0, originalImage.rows / 2), cv::Range(0, originalImage.cols));
    } else {
        supHalfMat = inputImage.clone();
        supHalfMat1 = originalImage.clone();
    }

    if (shrink) {
        cv::resize(supHalfMat, shrinkedInput, cv::Size(0, 0), 0.5, 0.5, CV_INTER_AREA);
        m_InputImage = shrinkedInput;
        cv::resize(supHalfMat1, shrinkedInput1, cv::Size(0, 0), 0.5, 0.5, CV_INTER_AREA);
        m_InputImageOriginal = shrinkedInput1;
    } else {
        m_InputImage = supHalfMat;
        m_InputImageOriginal = supHalfMat1;
    }
}

///assume that center of mass and symmetry angle have been computed
bool CrateHandle::calculateDistances()
{
    if (m_HandleFeatures.size() != m_HandleColors.size()) {
        qDebug() << "Different number of features and colors!!";
        return false;
    }

    for (int i = 0; i < (int)m_HandleFeatures.size(); i++) {
        HandleRecognitionSegment seg = m_SegmentsMap.at(m_HandleColors[i]);

        double angleHoriz = m_HandleFeatures[i].m_SymmetryAxis3;

        ///can not happen
        if (fabs(tan(angleHoriz)) > 100) {
            qDebug() << "Error calculate distances " << angleHoriz << tan(angleHoriz);
            return false;
        }

        ///distance to horizontal axis
        double maxTopDistance = m_HandleFeatures[i].m_CenterMass.y - seg.m_Top;
        double maxBottomDistance = seg.m_Bottom - m_HandleFeatures[i].m_CenterMass.y;
        double maxLeftDistance = m_HandleFeatures[i].m_CenterMass.x - seg.m_Left;
        double maxRightDistance = seg.m_Right - m_HandleFeatures[i].m_CenterMass.x;

        int topCenter = 0;
        int bottomCenter = 0;

        for (int j = seg.m_Top; j <= seg.m_Bottom; j++)
            if (m_HandleImage.at<cv::Vec3b>(j, m_HandleFeatures[i].m_CenterMass.x) == seg.m_ColorSegImage) {
                topCenter = j;
                break;
            }

        for (int j = seg.m_Bottom; j >= seg.m_Top; j--)
            if (m_HandleImage.at<cv::Vec3b>(j, m_HandleFeatures[i].m_CenterMass.x) == seg.m_ColorSegImage) {
                bottomCenter = j;
                break;
            }

        m_HandleFeatures[i].m_CenterMassSection = bottomCenter - topCenter;
        m_HandleFeatures[i].m_CenterMassSectionHeight = (double)(bottomCenter - topCenter) / (double)(seg.m_Bottom - seg.m_Top);
        m_HandleFeatures[i].m_CenterMassBottomSection = (double)(-m_HandleFeatures[i].m_CenterMass.y + bottomCenter) / (double)(bottomCenter - topCenter);
        m_HandleFeatures[i].m_BottomCenter = (double)bottomCenter;

        cv::Point centerMassLeft(0, 0);
        cv::Point centerMassRight(0, 0);
        seg.centerMassHalf(m_HandleImage, m_HandleFeatures[i].m_CenterMass, angleHoriz, centerMassLeft, centerMassRight);

        m_HandleFeatures[i].m_CenterMassLeft = centerMassLeft;
        m_HandleFeatures[i].m_CenterMassRight = centerMassRight;

        topCenter = 0;
        bottomCenter = 0;

        for (int j = seg.m_Top; j <= seg.m_Bottom; j++)
            if (m_HandleImage.at<cv::Vec3b>(j, m_HandleFeatures[i].m_CenterMassLeft.x) == seg.m_ColorSegImage) {
                topCenter = j;
                break;
            }

        for (int j = seg.m_Bottom; j >= seg.m_Top; j--)
            if (m_HandleImage.at<cv::Vec3b>(j, m_HandleFeatures[i].m_CenterMassLeft.x) == seg.m_ColorSegImage) {
                bottomCenter = j;
                break;
            }

        m_HandleFeatures[i].m_CenterMassLatSectionHeight = (double)(bottomCenter - topCenter) / (double)(seg.m_Bottom - seg.m_Top);
        m_HandleFeatures[i].m_CenterMassLatBottomSection = (double)(-m_HandleFeatures[i].m_CenterMassLeft.y + bottomCenter) / (double)(bottomCenter - topCenter);
        m_HandleFeatures[i].m_BottomCenterLeft = (double)bottomCenter;

        topCenter = 0;
        bottomCenter = 0;

        for (int j = seg.m_Top; j <= seg.m_Bottom; j++)
            if (m_HandleImage.at<cv::Vec3b>(j, m_HandleFeatures[i].m_CenterMassRight.x) == seg.m_ColorSegImage) {
                topCenter = j;
                break;
            }

        for (int j = seg.m_Bottom; j >= seg.m_Top; j--)
            if (m_HandleImage.at<cv::Vec3b>(j, m_HandleFeatures[i].m_CenterMassRight.x) == seg.m_ColorSegImage) {
                bottomCenter = j;
                break;
            }

        m_HandleFeatures[i].m_CenterMassLatSectionHeight += (double)(bottomCenter - topCenter) / (double)(seg.m_Bottom - seg.m_Top);
        m_HandleFeatures[i].m_CenterMassLatBottomSection += (double)(-m_HandleFeatures[i].m_CenterMassRight.y + bottomCenter) / (double)(bottomCenter - topCenter);
        m_HandleFeatures[i].m_CenterMassLatSectionHeight /= 2.0;
        m_HandleFeatures[i].m_CenterMassLatBottomSection /= 2.0;
        m_HandleFeatures[i].m_BottomCenterRight = (double)bottomCenter;

        bottomCenter = 0;

        for (int j = seg.m_Bottom; j >= seg.m_Top; j--)
            if (m_HandleImage.at<cv::Vec3b>(j, (m_HandleFeatures[i].m_CenterMassLeft.x + seg.m_Left) / 2) == seg.m_ColorSegImage) {
                bottomCenter = j;
                break;
            }

        m_HandleFeatures[i].m_BottomCenterHalfLeft = QPoint((m_HandleFeatures[i].m_CenterMassLeft.x + seg.m_Left) / 2, bottomCenter);
        bottomCenter = 0;

        for (int j = seg.m_Bottom; j >= seg.m_Top; j--)
            if (m_HandleImage.at<cv::Vec3b>(j, (m_HandleFeatures[i].m_CenterMassRight.x + seg.m_Right) / 2) == seg.m_ColorSegImage) {
                bottomCenter = j;
                break;
            }

        m_HandleFeatures[i].m_BottomCenterHalfRight = QPoint((m_HandleFeatures[i].m_CenterMassRight.x + seg.m_Right) / 2, bottomCenter);


//         qDebug() << "CenterMassLatSectionHeight" << m_HandleFeatures[i].m_CenterMassLatSectionHeight;
//         qDebug() << "CenterMassLatBottomSection" << m_HandleFeatures[i].m_CenterMassLatBottomSection;

        m_HandleFeatures[i].m_CenterMassLeft = centerMassLeft;
        m_HandleFeatures[i].m_CenterMassRight = centerMassRight;
        m_HandleFeatures[i].m_TopDistance = maxTopDistance;
        m_HandleFeatures[i].m_BottomDistance = maxBottomDistance;
        m_HandleFeatures[i].m_LeftDistance = maxLeftDistance;
        m_HandleFeatures[i].m_RightDistance = maxRightDistance;
        m_HandleFeatures[i].m_Height = maxTopDistance + maxBottomDistance;
        m_HandleFeatures[i].m_Width = maxLeftDistance + maxRightDistance;
        m_HandleFeatures[i].m_ImageHeight = m_HandleImage.rows;
        m_HandleFeatures[i].m_TopCornerYHeight = (double)seg.m_Top / (double)m_HandleImage.rows;
        m_HandleFeatures[i].m_ImageWidth = m_HandleImage.cols;
        m_HandleFeatures[i].m_HeightRatio = m_HandleFeatures[i].m_Width / m_HandleFeatures[i].m_ImageWidth;
        m_HandleFeatures[i].m_WidthRatio = m_HandleFeatures[i].m_Height / m_HandleFeatures[i].m_ImageHeight;

        m_HandleFeatures[i].m_WidthHeight = m_HandleFeatures[i].m_Height / m_HandleFeatures[i].m_Width;
        m_HandleFeatures[i].m_FillFactor = (double)seg.m_NoPoints / (m_HandleFeatures[i].m_Height + 1.0) / (m_HandleFeatures[i].m_Width + 1.0);

        if (maxBottomDistance < 0.01)
            m_HandleFeatures[i].m_TopBottomDistance = -1.0;
        else
            m_HandleFeatures[i].m_TopBottomDistance = maxTopDistance / maxBottomDistance;

        m_HandleFeatures[i].m_BottomHeight = maxBottomDistance / m_HandleFeatures[i].m_Height;

        m_HandleFeatures[i].m_CenterMassHalfWidth = (double)std::abs(centerMassLeft.x - centerMassRight.x) / m_HandleFeatures[i].m_Width;
    }

    return true;
}

bool CrateHandle::calculateMedianLine()
{
    if (m_HandleFeatures.size() != m_HandleColors.size()) {
        qDebug() << "Different number of features and colors!!";
        return false;
    }

    for (int i = 0; i < (int)m_HandleFeatures.size(); i++) {
        HandleRecognitionSegment seg = m_SegmentsMap.at(m_HandleColors[i]);

        std::vector<cv::Point> topEdge;
        std::vector<cv::Point> bottomEdge;

        seg.generateEdge(m_HandleImage, true, true, topEdge);
        seg.generateEdge(m_HandleImage, true, false, bottomEdge);

        std::vector<cv::Point> smoothTopEdge = topEdge;
        std::vector<cv::Point> smoothBottomEdge = bottomEdge;

        if (topEdge.size() != bottomEdge.size())
            return false;

        int smoothLevel = 5;
        int sumTop = 0, sumBottom = 0;

        for (int i = 0; i < (int)topEdge.size(); i++) {
            if (i > smoothLevel && i < (int)topEdge.size() - smoothLevel) {
                sumTop -= topEdge[i - smoothLevel].y;
                sumTop += topEdge[i + smoothLevel].y;
                smoothTopEdge[i].y = sumTop / (2 * smoothLevel + 1);
                sumBottom -= bottomEdge[i - smoothLevel].y;
                sumBottom += bottomEdge[i + smoothLevel].y;
                smoothBottomEdge[i].y = sumBottom / (2 * smoothLevel + 1);
            } else if (i < smoothLevel) {
                sumTop += topEdge[2 * i].y;
                sumTop += topEdge[2 * i + 1].y;
                sumBottom += bottomEdge[2 * i].y;
                sumBottom += bottomEdge[2 * i + 1].y;
            } else if (i == smoothLevel) {
                sumTop += topEdge[2 * i].y;
                sumBottom += bottomEdge[2 * i].y;
            }
        }

        std::vector<cv::Point> medianLine;
        double diff = 0;

        for (int j = 0; j < (int)topEdge.size(); j++) {
            medianLine.push_back(cv::Point(smoothTopEdge[j].x, (smoothTopEdge[j].y + smoothBottomEdge[j].y) / 2));
            //m_HandleImage.at<cv::Vec3b>(medianLine[j].y, medianLine[j].x) = cv::Vec3b(0,0,0);
        }

        for (int j = 1; j < (int)topEdge.size(); j++) {
            diff += (double)abs(medianLine[j].y - medianLine[j - 1].y);
        }

        m_HandleFeatures[i].m_SegMedianScore = diff / (double)topEdge.size();
    }



    return true;
}

void CrateHandle::eraseHandles(const std::vector<int>& keepList, const std::vector<int>& eraseList)
{
    std::vector<cv::Vec3b> newHandleColors;
    std::vector<CrateHandleFeatures> newHandleFeatures;

    for (int i = 0; i < (int)eraseList.size(); i++) {
        ImageSegment seg = m_SegmentsMap.at(m_HandleColors[eraseList[i]]);

        for (int i = seg.m_Left; i <= seg.m_Right; i++)
            for (int j = seg.m_Top; j <= seg.m_Bottom; j++) {
                if (m_HandleImage.at<cv::Vec3b>(j, i) == seg.m_ColorSegImage || m_HandleImage.at<cv::Vec3b>(j, i) == cv::Vec3b(0, 0, 0))
                    m_HandleImage.at<cv::Vec3b>(j, i) = cv::Vec3b(220, 220, 220);
            }
    }


    for (int i = 0; i < (int)keepList.size(); i++) {
        newHandleColors.push_back(m_HandleColors[keepList[i]]);
        newHandleFeatures.push_back(m_HandleFeatures[keepList[i]]);
    }

    m_HandleColors = newHandleColors;
    m_HandleFeatures = newHandleFeatures;
}

bool CrateHandle::computeDBEntry(CLogoHandleData& lhd) {
    if (m_HandleFeatures.size()!= 1)
        return false;
    
    CLogoHandleFeatures lhf;
    
    lhf.m_ImageHeight = m_HandleFeatures[0].m_ImageHeight;
    lhf.m_ImageWidth = m_HandleFeatures[0].m_ImageWidth;
    lhf.m_Width = m_HandleFeatures[0].m_Width;
    lhf.m_Height = m_HandleFeatures[0].m_Height;
    lhf.m_WidthHeight = m_HandleFeatures[0].m_WidthHeight;
    lhf.m_WidthRatio = m_HandleFeatures[0].m_WidthRatio;
    lhf.m_HeightRatio = m_HandleFeatures[0].m_HeightRatio;
    lhf.m_FillFactor = m_HandleFeatures[0].m_FillFactor;
    lhf.m_CenterMassHalfWidth = m_HandleFeatures[0].m_CenterMassHalfWidth;
    lhf.m_CenterMassSectionHeight = m_HandleFeatures[0].m_CenterMassSectionHeight;
    lhf.m_CenterMassBottomSection = m_HandleFeatures[0].m_CenterMassBottomSection;
    lhf.m_CenterMassLatSectionHeight = m_HandleFeatures[0].m_CenterMassLatSectionHeight;
    lhf.m_CenterMassLatBottomSection = m_HandleFeatures[0].m_CenterMassLatBottomSection;
    lhf.m_BottomHeight = m_HandleFeatures[0].m_BottomHeight;
    lhf.m_Contour = m_HandleFeatures[0].m_Contour;
    lhf.m_CenterMass = QPoint(m_HandleFeatures[0].m_CenterMass.x, m_HandleFeatures[0].m_CenterMass.y);
    lhf.m_CenterMassLeft = QPoint(m_HandleFeatures[0].m_CenterMassLeft.x, m_HandleFeatures[0].m_CenterMassLeft.y);
    lhf.m_CenterMassRight = QPoint(m_HandleFeatures[0].m_CenterMassRight.x, m_HandleFeatures[0].m_CenterMassRight.y);
    lhf.m_BottomCenterLeft = m_HandleFeatures[0].m_BottomCenterLeft;
    lhf.m_BottomCenterRight = m_HandleFeatures[0].m_BottomCenterRight;
    lhf.m_BottomCenter = m_HandleFeatures[0].m_BottomCenter;
    lhf.m_BottomCenterHalfLeft = m_HandleFeatures[0].m_BottomCenterHalfLeft;
    lhf.m_BottomCenterHalfRight = m_HandleFeatures[0].m_BottomCenterHalfRight;
    lhf.m_TopCornerYHeight = m_HandleFeatures[0].m_TopCornerYHeight;
    
    lhd.m_Features.push_back(lhf);

    return true;
}


bool CrateHandle::extractFeatures(CLogoHandleData& lhd) {
    QSize size = QSize(lhd.m_SegmentedImageWidth, lhd.m_SegmentedImageHeight);
    cv::Mat handleImage(size.height(), size.width(), CV_8UC3, cv::Scalar(255, 255, 255));
    std::vector<QPoint> editedContour = lhd.m_EditedContour;
    
    std::vector<int> poly;
    for (int i = 0; i < (int)editedContour.size(); i++) {
        poly.push_back(editedContour[i].x());
        poly.push_back(editedContour[i].y());
    }
    std::vector<std::pair<int, int>> contour = interpolateHandleContour(poly);
    cv::Point pointsCV[1][contour.size()];

    for (int i = 0; i < (int)contour.size(); i++)
        pointsCV[0][i] = cv::Point(contour[i].first, contour[i].second);

    int* listSize = new int[1];
    listSize[0] = (int)contour.size();
    const cv::Point* listPoly = { pointsCV[0] };
    cv::fillPoly(handleImage, &listPoly, listSize, 1, cv::Scalar(127, 127, 127));
    delete[] listSize;
        
    cv::Mat scaleImage;
    cv::Mat supHalfMat(handleImage, cv::Range(0, handleImage.rows / 2), cv::Range(0, handleImage.cols));
    cv::resize(supHalfMat, scaleImage, cv::Size(), 0.5, 0.5, CV_INTER_AREA);

    //resize destroys the colours in the image and we must restore them.   
    ///@todo: maybe we can do in another way here
    restoreBinaryColours(supHalfMat);

    QString imageName = "/home/cucu/temp/images/";
    int number = rand() % 100;
    imageName += QString::number(number) + QString(".png");
    std::string imageNameStd(qPrintable(imageName));
    cv::imwrite(imageNameStd, supHalfMat);
    
    lhd.m_Features.clear();
    
    CrateHandle handle(supHalfMat);
    if (!handle.computeDBEntry(lhd))
        return false;

    std::vector<QPoint> contour1 = lhd.m_Features[0].m_Contour;
    const int noPoints = 20;
    const int distPixels = 10;
    
    if (!contour1.size())
        return false;
    
    int step = (int)contour1.size()/noPoints;
    
    std::vector<QPoint> keyPoints;
    for (int i = 0; i < noPoints; i++) {
        keyPoints.push_back(contour1[i*step]);
    }
    
    for (int i = 0; i < (int)keyPoints.size(); i++) {
        QPoint prevPoint = keyPoints[(i-1+(int)keyPoints.size())%(int)keyPoints.size()];
        QPoint nextPoint = keyPoints[(i+1)%(int)keyPoints.size()];
        QPoint point = keyPoints[i];
        
        bool perp = false;
        double slope = 0.0;
            
        if (nextPoint.y() == prevPoint.y()) {
            perp = true;
        } else {
            slope = -(double)(nextPoint.x() - prevPoint.x())/(double)(nextPoint.y() - prevPoint.y());
        }
        
        QPoint point1, point2;
        
        if (perp) {
            point1 = QPoint(point.x(), point.y() - distPixels);
            point2 = QPoint(point.x(), point.y() + distPixels);
        } else {
            point1 = QPoint(point.x() + (int)((double)distPixels/sqrt(slope*slope+1)), point.y() + (int)(slope*(double)distPixels/sqrt(slope*slope+1)));
            point2  = QPoint(point.x() - (int)((double)distPixels/sqrt(slope*slope+1)), point.y() - (int)(slope*(double)distPixels/sqrt(slope*slope+1)));            
        }
        
        for (int k = 0; k < 2; k++) {
            cv::Point pointsCV1[1][noPoints];

            for (int j = 0; j < noPoints; j++) {
                if (j == i) {
                    if (!k)
                        pointsCV1[0][j] = cv::Point(point1.x(), point1.y());
                    else 
                        pointsCV1[0][j] = cv::Point(point2.x(), point2.y());
                } else {
                    pointsCV1[0][j] = cv::Point(keyPoints[j].x(), keyPoints[j].y());
                }
            }
            
            const cv::Point* listPoly1 = { pointsCV1[0] };
            int* listSize1 = new int[1];
            listSize1[0] = (int)noPoints;
            cv::Mat handleImage1(supHalfMat.rows, supHalfMat.cols, CV_8UC3, cv::Scalar(255, 255, 255));
            cv::fillPoly(handleImage1, &listPoly1, listSize1, 1, cv::Scalar(127, 127, 127));
            delete[] listSize1;
            QString fileName("/home/cucu/temp/Contour");
            fileName = fileName + QString::number(i*2+k) + QString(".png");
            cv::imwrite(qPrintable(fileName), handleImage1);
            
            CrateHandle handle1(handleImage1);
            if (!handle1.computeDBEntry(lhd))
                return false;
        }
    }
    return true;
}

std::vector< std::pair< int, int > > CrateHandle::interpolateHandleContour(const std::vector< int >& poly)
{
    std::vector<std::pair<int, int> > contourPoints;
    int len = (int)poly.size()/2;
    
    for (int i = 0; i < len / 3; i++) {
        std::vector<int> x(3, 0.0);
        std::vector<int> y(3, 0.0);
        
        for (int j = 0; j < 3; j++) {
            x[j] = poly[2 * (3 * i + j)];
            y[j] = poly[2 * (3 * i + j) + 1];
        }
        std::vector<std::pair<int, int>> newPoints = findInterpolating(x, y);
        for (int j = 0; j < (int)newPoints.size(); j++) {
            contourPoints.push_back(newPoints[j]);
        }
    }

    int i1 = len / 3;
    int j1 = len % 3 + 1;

    std::vector<int> x(j1, 0.0);
    std::vector<int> y(j1, 0.0);

    for (int j = 0; j < j1; j++) {
        x[j] = poly[2 * ((3 * i1 + j) % len)];
        y[j] = poly[2 * ((3 * i1 + j) % len) + 1];
    }

    std::vector<std::pair<int, int>> newPoints = findInterpolating(x, y);

    for (int j = 0; j < (int)newPoints.size(); j++)
        contourPoints.push_back(newPoints[j]);

    return contourPoints;
}


std::vector<std::pair<int, int>> CrateHandle::findInterpolating(const std::vector<int>& x, const std::vector<int>& y)
{
    std::vector<std::pair<int, int>> res;

    if (x.size() != y.size()) {
        printf("Blabla\n");
        return res;
    }

    for (int i = 0; i < (int)x.size(); i++) {
        res.push_back(std::pair<int, int>(x[i], y[i]));
    }

    if (x.size() != 3) {
        return res;
    }

    int maxX = 0;
    int maxY = 0;

    for (int i = 0; i < 3; i++)
        for (int j = i + 1; j < 3; j++) {
            if (std::abs(x[i] - x[j]) > maxX)
                maxX = abs(x[i] - x[j]);

            if (std::abs(y[i] - y[j]) > maxY)
                maxY = abs(y[i] - y[j]);
        }


    if (maxY > maxX / 3 && maxX > maxY / 3) {
        return res;
    }

    res.clear();
    res = interpolateSecondOrder(x, y, maxX > maxY);

    return res;
}


std::vector<std::pair<int, int> > CrateHandle::interpolateSecondOrder(const std::vector< int >& x, const std::vector< int >& y, bool onX)
{
    std::vector<std::pair<int, int>> res;

    if (x.size() != y.size()) {
        return res;
    }

    for (int i = 0; i < (int)x.size(); i++) {
        res.push_back(std::pair<int, int>(x[i], y[i]));
    }

    if (x.size() != 3)
        return res;

    if (!onX) {
        std::vector<std::pair<int, int>> res1;
        res1 = interpolateSecondOrder(y, x, true);
        res.clear();

        for (int i = 0; i < (int)res1.size(); i++)
            res.push_back(std::pair<int, int>(res1[i].second, res1[i].first));

        return res;
    }

    if (!((x[0] > x[1]) && (x[1] > x[2])) && !((x[0] < x[1]) && (x[1] < x[2])))
        return res;

    double factor = fabs(double(x[0] - x[1])) / fabs(double(x[1] - x[2]));

    if (factor < 0.2 || factor > 5.0)
        return res;

    cv::Mat A(3, 3, CV_64FC1);

    for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) {
            double sum = 0;

            for (int l = 0; l < 3; l++)
                sum += pow(x[l], j + k);

            A.at<double>(j, k) = sum;
        }

    cv::Mat b(3, 1, CV_64FC1);

    for (int j = 0; j < 3; j++) {
        double sum = 0;

        for (int l = 0; l < 3; l++)
            sum += (double)y[l] * pow(x[l], j);

        b.at<double>(j, 0) = sum;
    }

    cv::Mat a(3, 1, CV_64FC1);

    if (!cv::solve(A, b, a))
        return res;

    int step = 0;

    if (x[0] > x[1])
        step = -1;
    else
        step = +1;

    res.clear();

    for (int i = x[0]; i != x[2]; i += step) {
        double temp = a.at<double>(0, 0) + a.at<double>(1, 0) * (double)i + a.at<double>(2, 0) * (double)i * (double)i;
        res.push_back(std::pair<int, int>(i, (int)lrint(temp)));
    }
    
    return res;
}

void CrateHandle::restoreBinaryColours(cv::Mat& mat) {   
    for (int i = 0; i < mat.rows; i++) 
    for (int j = 0; j < mat.cols; j++) {
        cv::Vec3b color = mat.at<cv::Vec3b>(i, j);
        int avgColor = (color[0] + color[1] + color[2])/3;
        if (avgColor < 255)
            mat.at<cv::Vec3b>(i, j) = cv::Vec3b(127, 127, 127);
    }    
}
