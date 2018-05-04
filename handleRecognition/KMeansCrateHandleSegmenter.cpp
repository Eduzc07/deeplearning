#include "KMeansCrateHandleSegmenter.h"
#include <QDebug>
#include <QTime>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "OpenCVKMeansClusterer.h"
#include "RGBImageClusteringSegmenter.h"

///public
bool KMeansCrateHandleSegmenter::findHandles(std::vector< CrateHandle >& handles)
{
    bool foundHandle = false;
    cv::Mat segmentedImage;
    cv::Mat extraSegmentedImage;
    CrateHandle handle;
    CrateHandle handleWithoutPostProcessing;
    CrateHandle handleWithoutCompletion;
    CrateHandle handleCompletionWithoutPostProcessing;

    std::map<cv::Vec3b, ImageSegment, LessVec3b> segments;

    if (!segment(segmentedImage, segments)) {
        qDebug() << "Failure segmentation";
        return false;
    }

//     cv::namedWindow("Segmented image", CV_WINDOW_KEEPRATIO);
//     imshow("Segmented image", segmentedImage);

    foundHandle = handle.detectAndCreate(segmentedImage, segments);

    if (foundHandle) {
        qDebug() << "Detected with" << m_ClusterNo << "classes";
        handleWithoutPostProcessing = handle.clone();
        handleWithoutCompletion = handle.clone();
        optimizeEdges(handleWithoutCompletion, false);
        std::map<cv::Vec3b, ImageSegment, LessVec3b> extraSegments;
        ///optimization - apply KMeans only on the handle region

        if (m_CompleteKMeans) {
            setClusterNo(clusterNo() + 1);
            segment(extraSegmentedImage, extraSegments);
            setClusterNo(clusterNo() - 1);
            mergeLabels(handle, segments, extraSegmentedImage, extraSegments);
//             cv::namedWindow("Extra Segmented image", CV_WINDOW_KEEPRATIO);
//             imshow("Extra Segmented image", extraSegmentedImage);
            handleCompletionWithoutPostProcessing = handle.clone();
            ///@todo: here must correct
            optimizeEdges(handle, false);
        }
    }

    handles.clear();

    if (foundHandle) {
//         cv::namedWindow("Segmented image KMeans", CV_WINDOW_KEEPRATIO);
//         imshow("Segmented image KMeans", segmentedImage);
        if (m_CompleteKMeans) {
//             cv::namedWindow("Handle detection completed KMeans", CV_WINDOW_KEEPRATIO);
//             imshow("Handle detection completed KMeans", handle.m_HandleImage);
            handles.push_back(handle);
//             cv::namedWindow("Completion without postprocessing KMeans", CV_WINDOW_KEEPRATIO);
//             imshow("Completion without postprocessing KMeans", handleCompletionWithoutPostProcessing.m_HandleImage);
            handles.push_back(handleCompletionWithoutPostProcessing);
        }

//         cv::namedWindow("Handle detection KMeans", CV_WINDOW_KEEPRATIO);
//         imshow("Handle detection KMeans", handleWithoutPostProcessing.m_HandleImage);
        handles.push_back(handleWithoutPostProcessing);
//         cv::namedWindow("Without completion KMeans", CV_WINDOW_KEEPRATIO);
//         imshow("Without completion KMeans", handleWithoutCompletion.m_HandleImage);
        handles.push_back(handleWithoutCompletion);
    }


    return foundHandle;
}

///protected
bool KMeansCrateHandleSegmenter::segment(cv::Mat& output, std::map<cv::Vec3b, ImageSegment, LessVec3b>& segments)
{
    OpenCVKMeansClusterer omkc(m_InputImage, m_ClusterNo);
    RGBImageClusteringSegmenter rics(&omkc);
    if (!rics.execute(segments, output))
        return false;
    return true;
}


bool KMeansCrateHandleSegmenter::threshold(const cv::Mat& input, int thresh, const QRect& targetRect, cv::Mat& output, std::map< cv::Vec3b, ImageSegment, LessVec3b >& segments)
{
    cv::Mat threshImage(input.size(), input.type());

    for (int i = 0 ; i < input.rows; i++)
        for (int j = 0; j < input.cols; j++) {
            cv::Vec3b color = input.at<cv::Vec3b>(i, j);

            if ((color[0] + color[1] + color[2]) / 3 > thresh)
                threshImage.at<cv::Vec3b>(i, j) = cv::Vec3b(255, 255, 255);
            else
                threshImage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
        }

    ConnectedComponentLabelling ccl(input, threshImage);
    ccl.setBoundingRect(targetRect);
    ccl.execute(segments, output);


    return true;
}


bool KMeansCrateHandleSegmenter::checkLateralLabels(const CrateHandle& handle)
{
    if (handle.m_HandleColors.size() == 1) {
        cv::Mat extraSegmentedImage;
        std::map<cv::Vec3b, ImageSegment, LessVec3b> extraSegments;

        double avgColor = 0.0;
        int count = 0;
        ImageSegment seg = handle.m_SegmentsMap.at(handle.m_HandleColors.at(0));

        for (int i = seg.m_Left; i <= seg.m_Right; i++)
            for (int j = seg.m_Top; j <= seg.m_Bottom; j++) {
                cv::Vec3b color = handle.m_HandleImage.at<cv::Vec3b>(j, i);

                if (color == handle.m_HandleColors[0]) {
                    cv::Vec3b color1 = m_InputImageOriginal.at<cv::Vec3b>(j, i);
                    avgColor += (double)((color1[0] + color1[1] + color1[2]) / 3);
                    count++;
                }
            }

        avgColor /= (double)count;

        QRect targetRect(0, seg.m_Top,  m_InputImageOriginal.cols, seg.m_Bottom - seg.m_Top);
        threshold(m_InputImageOriginal, 5 * floor(avgColor), targetRect, extraSegmentedImage, extraSegments);

        std::map<cv::Vec3b, ImageSegment, LessVec3b>::iterator it = extraSegments.begin();
        QRect criticalRect1(0, seg.m_Top, seg.m_Left, seg.m_Bottom - seg.m_Top);
        QRect criticalRect2(seg.m_Right, seg.m_Top, m_InputImageOriginal.cols - seg.m_Right, seg.m_Bottom - seg.m_Top);
        int threshLabelNoPoints = 50;

        while (it != extraSegments.end()) {
            ImageSegment seg1 = it->second;

            if (seg1.m_ColorInitialImage == cv::Vec3b(255, 255, 255)) {
                if (seg1.m_NoPoints > threshLabelNoPoints && (seg1.intersects(criticalRect1) || seg1.intersects(criticalRect2))) {
                    qDebug() << "Dark crate with lateral labels!!";
                    return false;
                }
            }

            ++it;
        }

//         cv::namedWindow("Extra Segmented image dark", CV_WINDOW_KEEPRATIO);
//         imshow("Extra Segmented image dark", extraSegmentedImage);
    }

    return true;
}


void KMeansCrateHandleSegmenter::optimizeEdges(CrateHandle& handle, bool markInterior)
{
    QTime* time = new QTime();
    time->start();

    std::vector<cv::Vec3b> handleColors = handle.m_HandleColors;

//    qDebug() << "Number of colors " << handleColors.size();

    //mark the interior points of the handle area
    if (markInterior) {
        for (int i = 0; i < (int)handleColors.size(); i++) {
            markInteriorPoints(handle.m_HandleImage, handle.m_SegmentsMap[handleColors[i]]);
        }
    }

    //find the top edge of the handle and when it is a line fill the parts of the segment beneath that line
    for (int i = 0; i < (int)handleColors.size(); i++) {
        std::vector<cv::Point> topEdge;
        std::vector<cv::Point> bottomEdge;
//        qDebug() << "Optimizing top edge";
        optimizeEdge(handle.m_HandleImage, handle.m_SegmentsMap[handleColors[i]], true, topEdge);
//        qDebug() << "Optimizing bottom edge";
        optimizeEdge(handle.m_HandleImage, handle.m_SegmentsMap[handleColors[i]], false, bottomEdge);
    }

    int duration = time->elapsed();
    qDebug() << "KMeans Postprocessing" << duration;
}

void KMeansCrateHandleSegmenter::mergeLabels(CrateHandle& handle, std::map< cv::Vec3b, ImageSegment, LessVec3b >& segments, const cv::Mat& extraSegmentedImage, const std::map< cv::Vec3b, ImageSegment, LessVec3b >& extraSegments)
{
    std::vector<cv::Vec3b> handleColors = handle.m_HandleColors;

    ///join the handle with the neighboring small segments

    for (int l = 0; l < (int)handleColors.size(); l++) {
        std::map<cv::Vec3b, ImageSegment, LessVec3b>::const_iterator it = extraSegments.begin();

        for (; it != extraSegments.end(); ++it) {
            if (segments[handleColors[l]].intersects(it->second)) {
                ImageSegment seg = it->second;

                if (seg.m_NoPoints > m_ThreshMergeLabelsMaxAdditionSize)
                    continue;

//                 qDebug() << "Found intersection";
                int starti = seg.m_Top;
                int endi = seg.m_Bottom;
                int startj = seg.m_Left;
                int endj = seg.m_Right;

                for (int i = starti; i <= endi; i++)
                    for (int j = startj; j <= endj; j++) {
                        if (extraSegmentedImage.at<cv::Vec3b>(i, j) == seg.m_ColorSegImage
                                && i >= segments[handleColors[l]].m_Top && i <= segments[handleColors[l]].m_Bottom
                                && j >= segments[handleColors[l]].m_Left && j <= segments[handleColors[l]].m_Right) {
                            handle.m_HandleImage.at<cv::Vec3b>(i, j) = handleColors[l];
                        }
                    }
            }
        }
    }

    std::map<cv::Vec3b, ImageSegment, LessVec3b> segmentsTemp;
    cv::Mat segmentedTemp;
    ///apply connected component labeling und extract the bigest resulting segment
    ConnectedComponentLabelling ccl(handle.m_HandleImage, handle.m_HandleImage);
    ccl.execute(segmentsTemp, segmentedTemp);

    ///select the biggest segment of each color in the handle image
    std::vector<int> maxSegmentsSizes(handleColors.size(), 0);
    std::vector<cv::Vec3b> maxSegmentColor(handleColors.size(), cv::Vec3b(0, 0, 0));

    std::map<cv::Vec3b, ImageSegment, LessVec3b>::const_iterator it = segmentsTemp.begin();

    for (; it != segmentsTemp.end(); ++it) {
        ImageSegment seg = it->second;

        for (int i = 0; i < (int)handleColors.size(); i++) {
            if (seg.m_ColorInitialImage == handleColors[i]) {
                if (seg.m_NoPoints > maxSegmentsSizes[i]) {
                    maxSegmentsSizes[i] = seg.m_NoPoints;
                    maxSegmentColor[i] = it->first;
                }
            }
        }
    }

    ///marking of interior segments of a segment
    for (int i = 0; i < (int)maxSegmentColor.size(); i++) {
        if (maxSegmentsSizes[i] == 0)
            qDebug() << "Error segmentation of optimized images";

        ImageSegment seg;

        if (segmentsTemp.find(maxSegmentColor[i]) != segmentsTemp.end())
            seg = segmentsTemp[maxSegmentColor[i]];
        else
            qDebug() << "Segment fail " << maxSegmentColor[i][0] << maxSegmentColor[i][1] << maxSegmentColor[i][2];

        std::map<cv::Vec3b, ImageSegment, LessVec3b>::const_iterator it = segmentsTemp.begin();

        while (it != segmentsTemp.end()) {
            if (seg.contains(segmentedTemp, it->second)) {
                it->second.markWithNewColor(segmentedTemp, seg.m_ColorSegImage);
                it =  segmentsTemp.erase(it);
                continue;
            }

            ++it;
        }
    }

    handle.build(segmentedTemp, segmentsTemp, maxSegmentColor);
//     cv::namedWindow("Completed", CV_WINDOW_KEEPRATIO);
//     imshow("Completed", segmentedTemp);
}

///private

bool KMeansCrateHandleSegmenter::optimizeEdge(cv::Mat& handleImage, const HandleRecognitionSegment& seg, bool top, std::vector<cv::Point>& edgePoints)
{
    seg.generateEdge(handleImage, true, top, edgePoints);
//    qDebug() << "Total edge points" << edgePoints.size();
    double vx = 0.0, vy = 0.0, x0 = 0.0, y0 = 0.0, variance = 1000.0;
//     qDebug() << "edge points " << edgePoints.size() << countPoints;

    bool isCurve = isEdgeCurve(edgePoints);

    if (isCurve) {
//         qDebug() << "Interpolate curve";
        interpolateEdgeCurve(handleImage, seg, top, edgePoints);
    }

    bool isLine = isEdgeLineHough(edgePoints, top, vx, vy, x0, y0, variance);

    if (isLine) {
        qDebug() << "Interpolate line" << (top ? "top" : "bottom");
        interpolateEdgeLine(handleImage, seg, edgePoints, top, vx, vy, x0, y0, variance);
    }

    return true;
}

bool KMeansCrateHandleSegmenter::isEdgeLineRegression(const std::vector<cv::Point>& edgePoints, bool top, double& vx, double& vy, double& x0, double& y0, double& variance)
{
    //test first whether the top edge is a line
    bool isLine = true;
    double lineParams[4] = {0.0, 0.0, 0.0, 0.0};
    bool start = true;
    m_ThreshMinUnalteredTopEdgePoints = edgePoints.size() / 3;
    m_ThreshMinUnalteredBottomEdgePoints = edgePoints.size() / 2;
    int handleMinUnalteredEdgePoints = top ? m_ThreshMinUnalteredTopEdgePoints : m_ThreshMinUnalteredBottomEdgePoints;
    double handleMaxLineFittingEdgeVariance = top ? m_ThreshMaxLineFittingBottomEdgeVariance : m_ThreshMaxLineFittingBottomEdgeVariance;
    int count = 0;

    std::vector<cv::Point> localEdgePoints = edgePoints;

    while ((variance > handleMaxLineFittingEdgeVariance) &&
            ((int)localEdgePoints.size() > handleMinUnalteredEdgePoints) && (count < 30)) {
        //given a fitted line eliminates all the points in the top edge that are below that line
        if (!start) {
            vx = lineParams[0];
            vy = lineParams[1];
            x0 = lineParams[2];
            y0 = lineParams[3];

            std::vector<cv::Point> topEdgePointsTmp;

            for (int i = 0; i < (int)localEdgePoints.size(); i++) {
                double y1 = vy / vx * (localEdgePoints[i].x - x0) + y0;

                if (top) {
                    if (localEdgePoints[i].y - y1 <= 0)
                        topEdgePointsTmp.push_back(localEdgePoints[i]);
                } else {
                    if (localEdgePoints[i].y - y1 >= 0)
                        topEdgePointsTmp.push_back(localEdgePoints[i]);
                }
            }

            localEdgePoints = topEdgePointsTmp;
        }

        //least squares linear regression for 2D

        try {
            cv::Vec4f cvParams(lineParams[0], lineParams[1], lineParams[2], lineParams[3]);
            cv::fitLine(localEdgePoints, cvParams, CV_DIST_L2, 0, 0.01, 0.01);
        } catch(cv::Exception& e) {
            return false;
        }


        count++;

        vx = lineParams[0];
        vy = lineParams[1];
        x0 = lineParams[2];
        y0 = lineParams[3];

//        qDebug() << "Line fitted through" << localEdgePoints.size() << " points. Slope: " << vy/vx << " Point on line " <<  x0 << y0;

        if (std::abs(vx) < 0.001)
            return false;

        //find the variance of the line fitting
        double sumVar = 0.0;

        for (int i = 0; i < (int)localEdgePoints.size(); i++) {
            double y1 = vy / vx * (localEdgePoints[i].x - x0) + y0;
            sumVar += std::abs(localEdgePoints[i].y - y1);
        }

        variance = sumVar / localEdgePoints.size();

//        qDebug() << "Line fitting variance " << variance;
        start = false;
    }

    if (variance > handleMaxLineFittingEdgeVariance || (int)localEdgePoints.size() < handleMinUnalteredEdgePoints)
        isLine = false;

    return isLine;
}

bool KMeansCrateHandleSegmenter::isEdgeLineHough(const std::vector< cv::Point >& edgePoints, bool top, double& vx, double& vy, double& x0, double& y0, double& variance)
{
    if (!edgePoints.size())
        return false;

    cv::Mat contourImage = cv::Mat::zeros(m_InputImage.size(), CV_8UC1);

    for (int j = 0; j < (int)edgePoints.size(); j++)
        contourImage.at<uchar>(edgePoints[j].y, edgePoints[j].x) = 255;

    m_ThreshMinUnalteredTopEdgePoints = edgePoints.size() / 3;
    m_ThreshMinUnalteredBottomEdgePoints = edgePoints.size() / 2;
    int handleMinUnalteredEdgePoints = top ? m_ThreshMinUnalteredTopEdgePoints : m_ThreshMinUnalteredBottomEdgePoints;
//    qDebug() << "Thresh edge line " << handleMinUnalteredEdgePoints;

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(contourImage, lines, 1, CV_PI / 180, handleMinUnalteredEdgePoints, 10, 20);

    double maxLength = 0.0;
    int maxIndex = -1;

    for (size_t j = 0; j < lines.size(); j++) {
        cv::Point pt1(lines[j][0], lines[j][1]);
        cv::Point pt2(lines[j][2], lines[j][3]);

        double distX = (pt1.x - pt2.x);
        double distY = (pt1.y - pt2.y);
        double dist = sqrt(distX * distX + distY * distY);

        if (dist > maxLength) {
            maxLength = dist;
            maxIndex = j;
        }
    }

    if (maxLength == 0.0)
        return false;

    variance = 0.5;
    x0 = lines[maxIndex][0];
    y0 = lines[maxIndex][1];
    double x1 = lines[maxIndex][2];
    double y1 = lines[maxIndex][3];
    vx = x1 - x0;
    vy = y1 - y0;

    return true;
}

///@todo: to implement when a good idea comes
bool KMeansCrateHandleSegmenter::isEdgeCurve(const std::vector< cv::Point >& edgePoints)
{
    return true;
}


void KMeansCrateHandleSegmenter::interpolateEdgeLine(cv::Mat& handleImage, const ImageSegment& seg, std::vector< cv::Point >& edgePoints, bool top, double vx, double vy, double x0, double y0, double variance)
{
    if (fabs(vx) < 0.01)
        return;

    double slope = vy / vx;

    //fill the segment below the found line
    //and update the top edge line
    int i, j;
    bool startFillLeft = false;
    bool startFillRight = false;

    for (i = 0, j = edgePoints.size() - 1 ; i <= j; i++, j--) {
        double yleft = slope * (edgePoints[i].x - x0) + y0;
        double yright = slope * (edgePoints[j].x - x0) + y0;

        if (!startFillLeft && std::abs(yleft - edgePoints[i].y) < variance) {
            startFillLeft = true;
        }

        if (!startFillRight && std::abs(yright - edgePoints[j].y) < variance) {
            startFillRight = true;
        }

        if (startFillLeft) {
            int startk = 0, endk = 0;

            if (top) {
                startk = yleft;
                endk = edgePoints[i].y;
            } else {
                startk = edgePoints[i].y;
                endk = yleft;
            }

            for (int k = startk; k <= endk ; k++)
                handleImage.at<cv::Vec3b>(k, edgePoints[i].x) = seg.m_ColorSegImage;

            edgePoints[i] = cv::Point(edgePoints[i].x, yleft);
        }

        if (startFillRight) {
            int startk = 0, endk = 0;

            if (top) {
                startk = yright;
                endk = edgePoints[j].y;
            } else {
                startk = edgePoints[j].y;
                endk = yright;
            }

            for (int k = startk; k <= endk; k++)
                handleImage.at<cv::Vec3b>(k, edgePoints[j].x) = seg.m_ColorSegImage;

            edgePoints[j] = cv::Point(edgePoints[j].x, yright);
        }
    }
}

///@todo: to change this method to use approxPolyDp
void KMeansCrateHandleSegmenter::interpolateTopEdgeCurve(cv::Mat& handleImage, const ImageSegment& seg, const std::vector< cv::Point >& edgePoints, int imin)
{
    int angleSteps = 90 / m_ThreshCurveTrackingStepAngle;
    int delta = 0;

    for (int i = imin; i >= m_ThreshCurveTrackingDelta; i--) {
        delta =  - edgePoints[i].y + edgePoints[i - m_ThreshCurveTrackingDelta].y;

        if (delta >= m_ThreshCurveTracking) {
//             qDebug() << "Found discontinuity" << i;
            ///found discontinuity in curve
            for (int j = 0; j < angleSteps + 1; j++) {
                ///for every angle alpha trace the line passing through edgePoints[i] and that has the angle alpha
                ///and intersect it with the edge curve
                double alpha = double(j * m_ThreshCurveTrackingStepAngle) * M_PI / 180.0;
                int kIntersect = -1;

//                 qDebug() << "Alpha " << alpha;
                ///look for the intersection with the top edge
                for (int k = i - 1; k >= 0; k--) {
                    double y = (double)(i - k) * tan(alpha) + edgePoints[i].y;

                    if (y >= edgePoints[k].y) {
                        ///found intersection
                        kIntersect = k;
                        break;
                    }
                }

                ///interpolate the top edge
                if (kIntersect >= 0) {
//                     qDebug() << "Interpolate " << i << kIntersect;
                    for (int k = kIntersect + 1; k <= i; k++) {
                        double y = (double)(i - k) * tan(alpha) + edgePoints[i].y;

                        for (int l = y; l < edgePoints[k].y; l++)
                            handleImage.at<cv::Vec3b>(l, edgePoints[k].x) = seg.m_ColorSegImage;
                    }

                    i = kIntersect;
                    break;
                }
            }
        }
    }

    for (int i = imin; i < (int)edgePoints.size() - m_ThreshCurveTrackingDelta; i++) {
        delta = edgePoints[i + m_ThreshCurveTrackingDelta].y - edgePoints[i].y;

        if (delta >= m_ThreshCurveTracking) {
//             qDebug() << "Found discontinuity" << i;
            ///found discontinuity in curve
            for (int j = 0; j < angleSteps + 1; j++) {
                ///for every angle alpha trace the line passing through edgePoints[i] and that has the angle alpha
                ///and intersect it with the edge curve
                double alpha = double(j * m_ThreshCurveTrackingStepAngle) * M_PI / 180.0;
                int kIntersect = -1;

//                 qDebug() << "Alpha " << alpha;
                ///look for the intersection with the top edge
                for (int k = i + 1; k < (int)edgePoints.size(); k++) {
                    double y = (double)(k - i) * tan(alpha) + edgePoints[i].y;

                    if (y >= edgePoints[k].y) {
                        ///found intersection
                        kIntersect = k;
                        break;
                    }
                }

                ///interpolate the top edge
                if (kIntersect >= 0) {
//                     qDebug() << "Interpolate " << i << kIntersect;
                    for (int k = i; k <= kIntersect - 1; k++) {
                        double y = (double)(k - i) * tan(alpha) + edgePoints[i].y;

                        for (int l = y; l < edgePoints[k].y; l++)
                            handleImage.at<cv::Vec3b>(l, edgePoints[k].x) = seg.m_ColorSegImage;
                    }

                    i = kIntersect;
                    break;
                }
            }
        }
    }
}

///for the top edge assumes that it is a convex curve
void KMeansCrateHandleSegmenter::interpolateEdgeCurve(cv::Mat& handleImage, const ImageSegment& seg, bool top, std::vector< cv::Point >& edgePoints)
{
    ///for the moment only for top edges
    if (!top || !edgePoints.size()) {
//         qDebug() << "Interpolation of curves works only for top edge";
        return;
    }

    int imin = -1;
    int valmin = seg.m_Bottom;
//     int imax = -1;
//     int valmax = seg.m_Top;

    int edgeSize = edgePoints.size();

    for (int i = 0; i < edgeSize; i++) {
//         qDebug() << edgePoints[i].x() << edgePoints[i].y();

//         if (edgePoints[i].y > valmax) {
//             imax = i;
//             valmax = edgePoints[i].y;
//         }
        if (edgePoints[i].y < valmin) {
            imin = i;
            valmin = edgePoints[i].y;
        }
    }

    if (imin < edgeSize / 3 || imin > edgeSize * 2 / 3) {
//         qDebug() << "The maximum of the top edge closer to the sides of the griff" << imin << edgeSize;
        ///this is not really required
        return;
    }
    interpolateTopEdgeCurve(handleImage, seg, edgePoints, imin);
}

void KMeansCrateHandleSegmenter::markInteriorPoints(cv::Mat& handleImage, const HandleRecognitionSegment& seg)
{
    std::map< int, std::vector<std::pair<int, int> > >  verticalHolesMap;
    std::map< int, std::vector<std::pair<int, int> > >  horizontalHolesMap;

    findHolePositions(handleImage, seg, false, verticalHolesMap);
    findHolePositions(handleImage, seg, true, horizontalHolesMap);

    std::map< int, std::vector<std::pair<int, int> > >::iterator itVertical = verticalHolesMap.begin();

    for (; itVertical != verticalHolesMap.end(); ++itVertical) {
        int i = itVertical->first;

        for (int j = 0; j < (int)itVertical->second.size(); j++)
            for (int k = itVertical->second[j].first; k <= itVertical->second[j].second; k++)  {
                std::map< int, std::vector<std::pair<int, int> > >::iterator itHorizontal = horizontalHolesMap.find(k);

                if (itHorizontal == horizontalHolesMap.end())
                    continue;

                for (int l = 0; l < (int)itHorizontal->second.size(); l++) {
                    if (i >= itHorizontal->second[l].first && i <= itHorizontal->second[l].second) {
                        handleImage.at<cv::Vec3b>(k, i) = seg.m_ColorSegImage; //interior point found mark
                        break;
                    } // if i
                } // for l
            } // for k
    } // for itVertical
}

void KMeansCrateHandleSegmenter::findHolePositions(const cv::Mat& handleImage, const ImageSegment& seg, bool horizontal, std::map<int, std::vector<std::pair<int, int>>>& holesMap)
{
    int i_start, i_stop, j_start, j_stop;

    //initialization vertical horizontal
    if (!horizontal) {
        i_start = seg.m_Left;
        i_stop = seg.m_Right;
        j_start = seg.m_Top;
        j_stop = seg.m_Bottom;
    } else {
        j_start = seg.m_Left;
        j_stop = seg.m_Right;
        i_start = seg.m_Top;
        i_stop = seg.m_Bottom;
    }

    //functor objects used to test whether a point belongs to the segment or not
    ColorCompareVertical verticalHoleFinder;
    ColorCompareHorizontal horizontalHoleFinder;

    //for every line (horizontal/vertical)
    for (int i = i_start; i <= i_stop; i++) {
        //create the list of line segments
        std::vector<std::pair<int, int> > holes;

        //initialization
        bool foundHole = false;
        bool foundSegment = false;
        int firstPointHole, lastPointHole;
        int j = 0;

        //for every point on the chosen line
        for (j = j_start; j <= j_stop; j++) {
            //test first if the point belongs to segment
            bool isAHole;

            if (!horizontal)
                isAHole = verticalHoleFinder(handleImage, i, j, seg.m_ColorSegImage);
            else
                isAHole = horizontalHoleFinder(handleImage, i, j, seg.m_ColorSegImage);

            //when a first point of a hole is found save the first point of the segment
            if (isAHole) {
                if (foundSegment && !foundHole) {
                    firstPointHole = j;
                    foundHole = true;
                }

                //when the last point not belonging to the segment was found save it
            } else {
                if (foundHole && foundSegment) {
                    lastPointHole = j - 1;
                    holes.push_back(std::pair<int, int>(firstPointHole, lastPointHole));
                    foundHole = false;
                }

                foundSegment = true;
            }
        }

        //save the line only when points not belonging to the segment were found on it
        if (holes.size())
            holesMap[i] = holes;
    }
}

void KMeansCrateHandleSegmenter::printHolePositions(std::map< int, std::vector<std::pair<int, int> > >& holesMap)
{
    std::map< int, std::vector<std::pair<int, int> > >::iterator it = holesMap.begin();

    for (; it != holesMap.end(); ++it) {
        printf("Line %d: ", it->first);

        for (int i = 0; i < (int)it->second.size(); i++)
            printf("%d - %d ", it->second[i].first, it->second[i].second);

        printf("\n");
    }
}


