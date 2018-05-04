#include "FloodFillCrateHandleSegmenter.h"

#include <QTime>
#include <QDebug>

#include "FloodFillCrateHandleSegmenter.h"
#include "ConnectedComponentLabelling.h"
#include <opencv2/imgproc/imgproc.hpp>
///public members

bool FloodFillCrateHandleSegmenter::findHandles(std::vector< CrateHandle >& handles)
{
    QTime* t1 = new QTime();
    t1->start();

    cv::Mat mask;

    if (!removeBlackAndWhiteBackground(mask, true)) {
        return false;
    }

    cv::Mat blackAndWhiteMask(mask.rows, mask.cols, mask.type());
    blackAndWhiteMask = cv::Scalar::all(255) - mask;
    cv::Scalar diffLow(20, 20, 20);

    if (!removeCrate(mask, diffLow))
        return false;

    m_SegmentationOK = true;
    int maxi = mask.rows;
    int maxj = mask.cols;

    for (int i = 0; i < maxi; i++)
        for (int j = 0; j < maxj; j++)
            if (mask.at<uchar>(i, j) == 0 || blackAndWhiteMask.at<uchar>(i, j) == 0)
                mask.at<uchar>(i, j) = 0;

    std::map<cv::Vec3b, ImageSegment, LessVec3b> segments;
    cv::Mat output;
    cv::Mat mask_3u(mask.size(), CV_8UC3);
    int from_to[] = { 0, 0, 0, 1, 0, 2 };
    cv::mixChannels(&mask, 1, &mask_3u, 1, from_to, 3);
    CrateHandle handle;

    ConnectedComponentLabelling ccl(m_InputImage, mask_3u);
    ccl.setComputeAvgColor(true);
    ccl.execute(segments, output);
    removeOutlierPixels(mask, segments, output);

    bool found = false;

    if (!handle.detectAndCreate(output, segments)) {
        cv::Scalar diffHigh(30, 30, 30);
        removeCrate(mask, diffHigh);
        cv::mixChannels(&mask, 1, &mask_3u, 1, from_to, 3);

        segments.clear();
        ConnectedComponentLabelling ccl(m_InputImage, mask_3u);
        ccl.setComputeAvgColor(true);
        ccl.execute(segments, output);
        removeOutlierPixels(mask, segments, output);

        if (handle.detectAndCreate(output, segments))
            found = true;
    } else {
        found = true;
    }

    if (found) {
//         if (!handle.erode())
//             return false;
        int duration = t1->elapsed();
        qDebug() << "Handle detection flood fill " << duration << "ms";
//         cv::namedWindow("Flood Fill Handle", CV_WINDOW_KEEPRATIO);
//         cv::imshow("Flood Fill Handle", handle.m_HandleImage);
        handles.push_back(handle);
        return true;

    } else {
        qDebug() << "Handle not found in image";
        int duration = t1->elapsed();
        qDebug() << "Handle detection flood fill " << duration << "ms";
    }

    return false;
}

///protected

bool FloodFillCrateHandleSegmenter::removeBlackAndWhiteBackground(cv::Mat& mask, bool halfImage)
{
    QTime* t1 = new QTime();
    t1->start();

    if (!removeWhiteBackground(mask, halfImage))
        return false;

    bool ret = false;

    if (!removeBlackBackgroundAndSegment(mask, true, halfImage))
        ret = removeBlackBackgroundAndSegment(mask, false, halfImage);
    else
        ret = true;

    int duration = t1->elapsed();
    qDebug() << "Black and white background removal" << duration << "ms";

//     cv::namedWindow("Black and white removal mask 3", CV_WINDOW_KEEPRATIO);
//     imshow("Black and white removal mask 3", mask);

//    qDebug() << "ret" << ret;
    return ret;
}

bool FloodFillCrateHandleSegmenter::removeWhiteBackground(cv::Mat& mask, bool halfImage)
{
    QTime* t1 = new QTime();
    t1->start();

    if (!preprocessingWhiteBackgroundRemoval())
        return false;

    mask = cv::Mat::zeros(m_InputImage.rows + 2, m_InputImage.cols + 2, CV_8UC1);

    std::list<cv::Point> borderPoints;
    prepareBorderPointsWhiteRemoval(borderPoints, halfImage);

    PointToFlood* pointToFlood = new WhitePointToFlood();
    const cv::Scalar diff(20, 20, 20);

    if (!floodWithBorderPoints(mask, borderPoints, pointToFlood, diff))
        return false;

    delete pointToFlood;

    return true;
}

bool FloodFillCrateHandleSegmenter::removeBlackBackground(cv::Mat& mask, bool bigThresh, bool halfImage)
{
    QTime* t1 = new QTime();
    t1->start();


    std::list<cv::Point> borderPoints;
    prepareBorderPointsBlackRemoval(borderPoints, mask, 15, halfImage);

    const cv::Scalar lowDiff = cv::Scalar(10, 10, 10);
    const cv::Scalar bigDiff = cv::Scalar(20, 20, 20);

    PointToFlood* pointToFlood = new BlackPointToFlood();

    if (bigThresh) {
        if (!floodWithBorderPoints(mask, borderPoints, pointToFlood, bigDiff))
            return false;

    } else {
        if (!floodWithBorderPoints(mask, borderPoints, pointToFlood, lowDiff))
            return false;
    }

    delete pointToFlood;

    int duration = t1->elapsed();
    qDebug() << "Black background extraction " << duration << " ms";

    return true;
}


bool FloodFillCrateHandleSegmenter::removeBlackBackgroundAndSegment(cv::Mat& mask, bool thresHigh, bool halfImage)
{
    if (thresHigh)
        removeBlackBackground(mask, true, halfImage);
    else
        removeBlackBackground(mask, false, halfImage);

//     imshow("Black and white removal mask 1", mask);
    cv::Mat mask_3u(mask.size(), CV_8UC3);
    int from_to[] = { 0, 0, 0, 1, 0, 2 };
    cv::mixChannels(&mask, 1, &mask_3u, 1, from_to, 3);
    cv::Mat segmentedMask;
    std::map<cv::Vec3b, ImageSegment, LessVec3b> segments;
    ConnectedComponentLabelling ccl(m_InputImage, mask_3u);
    ccl.execute(segments, segmentedMask);

    cv::Vec3b colorCrateSegment(0, 0, 0);
    bool foundCrate = false;

    std::map<cv::Vec3b, ImageSegment, LessVec3b>::const_iterator it = segments.begin();

    for (; it != segments.end(); ++it) {
        if (it->second.m_ColorInitialImage != cv::Vec3b(0, 0, 0))
            continue;

        if (it->second.m_NoPoints >=  m_InputImage.rows * m_InputImage.cols / 2) {
            qDebug() << "Flood fill: Found segment";
            colorCrateSegment = it->first;
            foundCrate = true;
        }
    }

    if (!foundCrate) {
        qDebug() << "Crate area has not been correctly segmented";
        return false;
    }

    int maxi = mask.rows;
    int maxj = mask.cols;

    m_Left = maxj;
    m_Top = maxi;
    m_Bottom = m_Right = 0;
    
    for (int i = 0; i < maxi; i++)
        for (int j = 0; j < maxj; j++) {
            if (segmentedMask.at<cv::Vec3b>(i, j) != colorCrateSegment) {
                mask.at<uchar>(i, j) = 255;
            } else {
                if (i < m_Top)
                    m_Top = i;
                if (i > m_Bottom)
                    m_Bottom = i;
                if (j < m_Left)
                    m_Left = j;
                if (j > m_Right)
                    m_Right = j;
            }
        }

//     cv::namedWindow("Black and white removal mask 2", CV_WINDOW_KEEPRATIO);
//     imshow("Black and white removal mask 2", segmentedMask);


    return true;
}

bool FloodFillCrateHandleSegmenter::removeCrate(cv::Mat& mask, const cv::Scalar& thresh)
{
    QTime* t1 = new QTime();
    t1->start();

    std::list<cv::Point> borderPoints;
    prepareBorderPointsCrateRemoval(borderPoints, mask, 25, 5);

    PointToFlood* pointToFlood = new PointToFlood();

    if (!floodWithBorderPoints(mask, borderPoints, pointToFlood, thresh))
        return false;

    delete pointToFlood;

    return true;
}


bool FloodFillCrateHandleSegmenter::removeOutlierPixels(const cv::Mat& mask, std::map< cv::Vec3b, ImageSegment, LessVec3b >& segments, cv::Mat& segmentedImage)
{
    std::map<cv::Vec3b, ImageSegment, LessVec3b> tempSegments;
    std::map<cv::Vec3b, int, LessVec3b> colorSegMap;
    std::map<cv::Vec3b, bool, LessVec3b> computedSegMap;

//     cv::namedWindow("Black and White Mask", CV_WINDOW_KEEPRATIO);
//     imshow("Black and White Mask", segmentedImage);

    int count = 0;
    int maxPointCrateSegment = 0;
    ImageSegment crateSegment;
    ///filter the segments
    std::map<cv::Vec3b, ImageSegment, LessVec3b>::iterator it = segments.begin();

    for (; it != segments.end(); ++it)
    for (; it != segments.end(); ++it) {
        ImageSegment seg = it->second;

        if (seg.m_Left == 0 || seg.m_Right == mask.cols || seg.m_Top == 0)
            continue;

        if (seg.m_NoPoints < 500)
            continue;

        if (seg.m_ColorInitialImage == cv::Vec3b(0, 0, 0)) {
            tempSegments[seg.m_ColorSegImage] = seg;
            colorSegMap[seg.m_ColorSegImage] = count;
            computedSegMap[seg.m_ColorSegImage] = false;
            count++;

        } else {
            if (seg.m_NoPoints > maxPointCrateSegment) {
                crateSegment = seg;
                maxPointCrateSegment = seg.m_NoPoints;
                qDebug() << "Flood fill: Found crate segment";
            }
        }
    }

    if (!maxPointCrateSegment)
        return false;



    segments = tempSegments;

    cv::Vec3f avgCrateSegmentColor = crateSegment.m_AvgColor;
    double avgCrateSegmentColor0 = avgCrateSegmentColor[0] / crateSegment.m_NoPoints * 255.0;
    double avgCrateSegmentColor1 = avgCrateSegmentColor[1] / crateSegment.m_NoPoints * 255.0;
    double avgCrateSegmentColor2 = avgCrateSegmentColor[2] / crateSegment.m_NoPoints * 255.0;
    avgCrateSegmentColor = cv::Vec3f(avgCrateSegmentColor0, avgCrateSegmentColor1, avgCrateSegmentColor2);

    int maxi = mask.rows;
    int maxj = mask.cols;

    for (int i = 0; i < maxi; i++)
        for (int j = 0; j < maxj; j++) {
            if (mask.at<uchar>(i, j) != 0)
                continue;

            cv::Vec3b pixelSegmentedColor = segmentedImage.at<cv::Vec3b>(i, j);
            std::map<cv::Vec3b, int, LessVec3b>::iterator it = colorSegMap.find(pixelSegmentedColor);

            if (it == colorSegMap.end())
                continue;

            ImageSegment seg = segments[it->second];
            cv::Vec3b pixelColor = m_InputImage.at<cv::Vec3b>(i, j);
            cv::Vec3f avgSegmentColor = seg.m_AvgColor;

            if (!computedSegMap[pixelSegmentedColor]) {
//                 qDebug() << avgSegmentColor[0] << avgSegmentColor[1] << avgSegmentColor[2];
                double temp0 = avgSegmentColor[0] / (double)seg.m_NoPoints * 255.0;
                double temp1 = avgSegmentColor[1] / (double)seg.m_NoPoints * 255.0;
                double temp2 = avgSegmentColor[2] / (double)seg.m_NoPoints * 255.0;
                avgSegmentColor = cv::Vec3f(temp0, temp1, temp2);
//                 qDebug() << avgSegmentColor[0] << avgSegmentColor[1] << avgSegmentColor[2];
                segments[it->second].m_AvgColor = avgSegmentColor;
                computedSegMap[pixelSegmentedColor] = true;
//                 qDebug() << "Avg color calculated for " << segments[it->second].m_ColorSegImage[0] << segments[it->second].m_ColorSegImage[1] << segments[it->second].m_ColorSegImage[2];
//                 qDebug() << "No points " << seg.m_NoPoints;
//                 qDebug() << temp0 << temp1 << temp2;
            }

            cv::Vec3f floatPixelColor = static_cast<cv::Vec3f>(pixelColor);

            if (dist(floatPixelColor, avgCrateSegmentColor) < 0.3 * dist(floatPixelColor, avgSegmentColor)) {
//                 qDebug() << "Pixel color" << floatPixelColor[0] << floatPixelColor[1] << floatPixelColor[2];
//                 qDebug() << "Segment color" << avgSegmentColor[0] << avgSegmentColor[1] << avgSegmentColor[2];
//                 qDebug() << "Crate segment color" << avgCrateSegmentColor[0] << avgCrateSegmentColor[1] << avgCrateSegmentColor[2];
                segmentedImage.at<cv::Vec3b>(i, j) = crateSegment.m_ColorSegImage;
            }
        }

    return true;
}


///private


bool FloodFillCrateHandleSegmenter::preprocessingWhiteBackgroundRemoval()
{
    try {
        int maxBorderThickness = 10;
        std::vector<cv::Point> topWhiteBorder;

        for (int i = 1; i < m_InputImage.cols - 1; i++) {
            for (int j = 0; j < maxBorderThickness; j++) {
                cv::Vec3b pixelColor = m_InputImage.at<cv::Vec3b>(j, i);

                if (pixelColor[0] < 150 && pixelColor[1] < 150 && pixelColor[2] < 150) {
//                     m_InputImage.at<cv::Vec3b>(j,i) = cv::Vec3b(125,125,125);
                    topWhiteBorder.push_back(cv::Point(i, j));
                    break;
                }
            }
        }

        if (topWhiteBorder.size() < 20)
            return false;

        cv::Vec4f lineParams;
        double vx, vy, x0, y0;
        cv::fitLine(topWhiteBorder, lineParams, CV_DIST_L2, 0, 0.01, 0.01);
        vx = lineParams[0];
        vy = lineParams[1];
        x0 = lineParams[2];
        y0 = lineParams[3];

        if (std::abs(vx) < 0.001)
            return false;

        //qDebug() << "Line fitted through" << topWhiteBorder.size() << "points. Slope:" << vy/vx <<  "Point on line "  << x0 << y0;

        //find the variance of the line fitting
        double sumVar = 0.0;

        for (int i = 0; i < (int)topWhiteBorder.size(); i++) {
            double y1 = vy / vx * (topWhiteBorder[i].x - x0) + y0;
            sumVar += std::abs(topWhiteBorder[i].y - y1);
        }

//         double variance = sumVar/(double)topWhiteBorder.size();

        //qDebug() << "Line fitting variance " << variance;

        for (int i = 0; i < m_InputImage.cols; i++) {
            double y = vy / vx * ((double)i - x0) + y0;
            int j = y;
            m_InputImage.at<cv::Vec3b>(j + 2, i) = cv::Vec3b(0, 0, 0);
        }

        return true;
    } catch(cv::Exception& e) {
        return false;
    }
}


void FloodFillCrateHandleSegmenter::prepareBorderPointsWhiteRemoval(std::list<cv::Point>& borderPoints, bool halfImage)
{
    for (int i = 0; i < m_InputImage.rows; i++) {
        borderPoints.push_back(cv::Point(i, m_InputImage.cols - 1));
//         input.at<Vec3b>(i,input.cols-1) = (0,0,0);
        borderPoints.push_back(cv::Point(i, 0));
//         input.at<Vec3b>(i,0) = (0,0,0);
    }

    for (int i = 1; i < m_InputImage.cols - 1; i++) {
        borderPoints.push_back(cv::Point(0, i));

//         input.at<Vec3b>(0,i) = (0,0,0);
        if (!halfImage)
            borderPoints.push_back(cv::Point(m_InputImage.rows - 1, i));

//         input.at<Vec3b>(input.rows-1,i) = (0,0,0);
    }
}

void FloodFillCrateHandleSegmenter::prepareBorderPointsBlackRemoval(std::list< cv::Point >& borderPoints, cv::Mat& whiteRemovedMask, int maxDistThreshold, bool halfImage)
{
    prepareBorderPointsCrateRemoval(borderPoints, whiteRemovedMask, maxDistThreshold, 0, halfImage);
}
void FloodFillCrateHandleSegmenter::prepareBorderPointsCrateRemoval(std::list< cv::Point >& borderPoints, cv::Mat& blackAndWhiteRemovedMask, int maxDistThreshold, int topSpace, bool halfImage)
{
//     bool firstLeftFound = false;
    cv::Point firstLeft(0, 0);

    for (int i = 0; i < m_InputImage.rows; i++) {
        int y = 0;
        bool loop = true;
        bool found = false;

        while (loop) {
            int color = blackAndWhiteRemovedMask.at<uchar>(i + 1, y + 1);

            if (!color) {
                loop = false;
                found = true;
                break;
            }

            if (y >= maxDistThreshold)
                loop = false;

            y++;
        }

        if (found) {
            borderPoints.push_back(cv::Point(i, y + topSpace));
//             input.at<Vec3b>(i,y) = Vec3b(220,50,50);
//             if (!firstLeftFound) {
//                 firstLeft = Point(i,y);
//                 firstLeftFound = true;
//             }
        }
    }

//     bool firstRightFound = false;
//     cv::Point firstRight(0,0);

    for (int i = 0; i < m_InputImage.rows; i++) {
        int y = m_InputImage.cols - 1;
        bool loop = true;
        bool found = false;

        while (loop) {
            int color = blackAndWhiteRemovedMask.at<uchar>(i + 1, y + 1);

            if (!color) {
                loop = false;
                found = true;
                break;
            }

            if (y <= m_InputImage.cols - maxDistThreshold)
                loop = false;

            y--;
        }

        if (found) {
            borderPoints.push_back(cv::Point(i, y - topSpace));
//             input.at<Vec3b>(i,y) = Vec3b(220,50,50);
//             if (!firstRightFound) {
//                 firstRight = Point(i,y);
//                 firstRightFound = true;
//             }
        }
    }

    for (int i = 1; i < m_InputImage.cols - 1; i++) {
        int x = 0;
        bool loop = true;
        bool found = false;

        while (loop) {
            int color = blackAndWhiteRemovedMask.at<uchar>(x + 1, i + 1);

            if (!color) {
                loop = false;
                found = true;
                break;
            }

            if (x >= maxDistThreshold)
                loop = false;

            x++;
        }

        if (found) {
            borderPoints.push_back(cv::Point(x + topSpace, i));
//             input.at<Vec3b>(x,i) = Vec3b(50,50,220);
        }
    }

    if (!halfImage) {
        for (int i = 1; i < m_InputImage.cols - 1; i++) {
            int x = m_InputImage.rows - 1;
            bool loop = true;
            bool found = false;

            while (loop) {
                int color = blackAndWhiteRemovedMask.at<uchar>(x + 1, i + 1);

                if (!color) {
                    loop = false;
                    found = true;
                    break;
                }

                if (x <= m_InputImage.rows - maxDistThreshold)
                    loop = false;

                x--;
            }

            if (found) {
                borderPoints.push_back(cv::Point(x - topSpace, i));
                //             input.at<Vec3b>(x,i) = Vec3b(50, 220 ,50);
            }
        }
    }
}


bool FloodFillCrateHandleSegmenter::floodWithBorderPoints(cv::Mat& mask, std::list< cv::Point >& borderPoints, FloodFillCrateHandleSegmenter::PointToFlood* pointToFlood, const cv::Scalar& thresh)
{
    try {
        cv::Scalar newScalar = cv::Scalar(100, 100, 100);

        std::list<cv::Point>::const_iterator it = borderPoints.begin();

        for (; it != borderPoints.end(); ++it) {
//             cv::Point curPoint = *it;
            cv::Vec3b inputPixelColor = m_InputImage.at<cv::Vec3b>(it->x, it->y);
            int maskPixel = mask.at<uchar>(it->x + 1, it->y + 1);
            //         qDebug() << curPoint.x << "," << curPoint.y << " " << inputPixel[0] << "," << inputPixel[1] << "," << inputPixel[2] << "-" << maskPixel;

            if ((*pointToFlood)(inputPixelColor)  && !maskPixel)  {
                cv::floodFill(m_InputImage, mask, cv::Point(it->y, it->x), newScalar, 0, thresh, thresh, 4 | cv::FLOODFILL_FIXED_RANGE | cv::FLOODFILL_MASK_ONLY | 255 << 8);
                //             qDebug() << "flood";
            }
        }

        return true;
    } catch(cv::Exception& e) {
        return false;
    }
}


double FloodFillCrateHandleSegmenter::dist(const cv::Vec3f& f1, const cv::Vec3f& f2)
{
    double term0 = (f1[0] - f2[0]) * (f1[0] - f2[0]);
    double term1 = (f1[1] - f2[1]) * (f1[1] - f2[1]);
    double term2 = (f1[2] - f2[2]) * (f1[2] - f2[2]);

    return sqrt(term0 + term1 + term2);
}
