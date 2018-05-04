#include "CrateHandleExtractor.h"

#include <QTime>
#include <QDebug>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

bool CrateHandleExtractor::execute()
{
    QTime* totalTime = new QTime();
    totalTime->start();

    QTime* time = new QTime();
    time->start();

    qDebug() << "Step 1: Handle segmentation";

    m_InputImageCopy = m_InputImage.clone();
    int brightness = imageBrightness();
    CrateHandle handleFloodFill;
    std::vector<CrateHandle> handlesKMeans;


    bool foundFloodFill = false;
    ///detects the handle with the flood fill method
    bool brightImage = handleDetectionFloodFill(handleFloodFill, foundFloodFill);

    ///Depending on the brightness of the image uses different parameter and preprocessing for KMeans based segmentation
    if (brightImage) {
        switch (brightness) {
        case CRATE_BRIGHTNESS::DARK:
            equalizeColor();
            handleDetectionKMeans(handlesKMeans, 2, 5, false);
            break;

        case CRATE_BRIGHTNESS::BRIGHT:
            handleDetectionKMeans(handlesKMeans, 2, 5);
            break;

        case CRATE_BRIGHTNESS::NORMAL:
            handleDetectionKMeans(handlesKMeans, 2, 7);
            break;

        default:
            qDebug() << "unknown image type";
        }
    } else {
        equalizeColor();
        handleDetectionKMeans(handlesKMeans, 2, 5, false);
    }


    ///builds a list of all found handles
    std::vector<CrateHandle> handles = handlesKMeans;

    if (foundFloodFill) {
        handles.push_back(handleFloodFill);
        qDebug() << "Adding flood fill handle";
    }

    for (int i = 0; i < (int)handles.size(); i++) {
        m_Handles.push_back(handles[i]);
    }


    int duration = time->elapsed();
    qDebug() << "Total handle segmentation" << duration;

    qDebug() << handles.size() << "Handles found";

    qDebug() << "Step 2: Eliminating false segmentations";
    time->start();

    std::vector<CrateHandle> symmetricHandles;
    std::vector<int> keepList;

    ///check the top and bottom edges of the handles

    m_Handles = handles;

    ///test with the rotation angles of the handles are consistent and eliminates the inconsistent handles
    for (int i = 0; i < (int)handles.size(); i++) {
        handles[i].score(i);

        if (!handles[i].badSegmentation())
            keepList.push_back(i);
    }


    if (keepList.size() && keepList[keepList.size() - 1] != (int)handles.size() - 1) {
        handles.pop_back();
        keepList.pop_back();
        foundFloodFill = false;
    }


    std::vector<CrateHandle> tempHandles;

    for (int i = 0; i < (int)keepList.size(); i++)
        tempHandles.push_back(handles[keepList[i]]);

    handles = tempHandles;

    ///checks if the flood fill handle matches with the kmeans handles
    bool distinctHandles = false;

    if (foundFloodFill) {
        if (differentHandles(handles)) {
            distinctHandles = true;
            qDebug() << "distinctHandles";
        }

        if (!distinctHandles && removeFloodFillHandle(handles, handles[handles.size() - 1])) {
            handles.pop_back();
        }
    }

    if (distinctHandles) {
        qDebug() << "Distinct handles found ";
        return false;
    }

    ///filters the handles based on their segmentation score and on how many handle candidates are left in the CrateHandle
    for (int i = 0; i < (int)handles.size(); i++) {
        if (!handles[i].badScore() && !handles[i].tooManyHandles())
            symmetricHandles.push_back(handles[i]);
        else
            continue;
    }

    ///choses the optimal CrateHandle
    if (symmetricHandles.size()) {
        //m_Handles = symmetricHandles;
        CrateHandle handle = findOptimalHandle(symmetricHandles);
        handle.keepUniqueHandle(0);
        m_Handle = handle.clone();
    } else {
        qDebug() << "Symmetric horizontal handle not found";
        return false;
    }


    duration = time->elapsed();
    qDebug() << "Total segmentation analysis" << duration;

    qDebug() << "Step 3: Calculating handle characteristics";
    time->start();


    duration = time->elapsed();
    qDebug() << "Total time handle calculation" << duration;

    int totalDuration = totalTime->elapsed();
    qDebug() << "Total time handle extraction" << totalDuration;


    return true;
}

int CrateHandleExtractor::imageBrightness()
{
    int maxi = m_InputImage.rows;
    int maxj = m_InputImage.cols;

    double sum0 = 0, sum1 = 0, sum2 = 0;

    for (int i = 0; i < maxi; i++)
        for (int j = 0; j < maxj; j++) {
            cv::Vec3b pixelColor = m_InputImage.at<cv::Vec3b>(i, j);
            sum0 += pixelColor[0];
            sum1 += pixelColor[1];
            sum2 += pixelColor[2];
        }

    double mean0 = sum0 / maxi / maxj;
    double mean1 = sum1 / maxi / maxj;
    double mean2 = sum2 / maxi / maxj;

    qDebug() << "Average pixel color" << mean0 << mean1 << mean2;


    if (mean0 < m_ThresMaxDark && mean1 < m_ThresMaxDark && mean2 < m_ThresMaxDark) {
        qDebug() << "Dark crate ";
        return CRATE_BRIGHTNESS::DARK;
    } else if (mean0 > m_ThresMinLight && mean1 > m_ThresMinLight && mean2 > m_ThresMinLight) {
        qDebug() << "Bright crate";
        return CRATE_BRIGHTNESS::BRIGHT;
    }

    qDebug() << "Normal crate";
    return CRATE_BRIGHTNESS::NORMAL;
}


bool CrateHandleExtractor::handleDetectionFloodFill(CrateHandle& handle, bool& found)
{
    std::vector<CrateHandle> handles;

    FloodFillCrateHandleSegmenter ffchs(m_InputImage, m_InputImageCopy, true);

    if (ffchs.findHandles(handles)) {
        handle = handles[0];
        found = true;
        return true;
    }

    found = false;
    return ffchs.segmentationOK();
}


void CrateHandleExtractor::equalizeColor()
{
    std::vector<cv::Mat> channels;

    cv::cvtColor(m_InputImage, m_InputImage, CV_BGR2YCrCb);
    cv::split(m_InputImage, channels);
    cv::equalizeHist(channels[0], channels[0]);
    cv::merge(channels, m_InputImage);
    cv::cvtColor(m_InputImage, m_InputImage, CV_YCrCb2BGR);
}


void CrateHandleExtractor::handleDetectionKMeans(std::vector<CrateHandle>& handles, int minCluster, int maxCluster, bool completeKMeans)
{
    QTime* t1 = new QTime();
    t1->start();

    int count = minCluster;
    bool foundHandle = false;

    while (count < maxCluster && !foundHandle) {
        KMeansCrateHandleSegmenter kchs(m_InputImage, m_InputImageCopy, count, true);
        kchs.setCompleteKMeans(completeKMeans);

        if (kchs.findHandles(handles))
            break;

        count++;
    }

    int duration = t1->elapsed();
    qDebug() << "Total time handle detection kmeans " << duration << "ms";
}

bool CrateHandleExtractor::removeFloodFillHandle(const std::vector<CrateHandle>& handlesKMeans, const CrateHandle& handleFloodFill)
{
    bool removeFloodFillHandle = true;

    if (handlesKMeans.size() == 0)
        return false;

    if (handlesKMeans.size() <= 1)
        removeFloodFillHandle = false;

    for (int i = 0; i < (int)handlesKMeans.size(); i++) {
        if (handlesKMeans[i].getFeatures().size() != 1)
            removeFloodFillHandle = false;
    }

    if (handleFloodFill.getFeatures().size() != 1)
        removeFloodFillHandle = false;

    if (removeFloodFillHandle) {
        double meanTopDistance = 0.0;
        double meanBottomDistance = 0.0;
        double meanLeftDistance = 0.0;
        double meanRightDistance = 0.0;

        for (int i = 0; i < (int)handlesKMeans.size(); i++) {
            meanTopDistance += handlesKMeans[i].getFeatures()[0].m_TopDistance;
            meanBottomDistance += handlesKMeans[i].getFeatures()[0].m_BottomDistance;
            meanLeftDistance += handlesKMeans[i].getFeatures()[0].m_LeftDistance;
            meanRightDistance += handlesKMeans[i].getFeatures()[0].m_RightDistance;
        }

        meanTopDistance /= (double)handlesKMeans.size();
        meanBottomDistance /= (double)handlesKMeans.size();
        meanLeftDistance /= (double)handlesKMeans.size();
        meanRightDistance /= (double)handlesKMeans.size();

//         qDebug() << "Distances KMeans " << meanTopDistance << meanBottomDistance << meanLeftDistance << meanRightDistance;
//         qDebug() << "Distances Floodfill " << handleFloodFill.getFeatures()[0].m_TopDistance << handleFloodFill.getFeatures()[0].m_BottomDistance
//                         << handleFloodFill.getFeatures()[0].m_LeftDistance << handleFloodFill.getFeatures()[0].m_RightDistance;


        bool ok = false;
        int count = 0;

        if (!ok && (fabs(handleFloodFill.getFeatures()[0].m_TopDistance - meanTopDistance) < m_ThresFloodDimTol)) {
            count++;
        } else {
            if (!ok) {
                printf("Remove flood fill handle top %f %f %d \n", handleFloodFill.getFeatures()[0].m_TopDistance, meanTopDistance, m_ThresFloodDimTol);
                qDebug() << "Remove flood fill handle top" << handleFloodFill.getFeatures()[0].m_TopDistance << meanTopDistance << m_ThresFloodDimTol;
            }

            ok = true;
        }

        if (!ok && (fabs(handleFloodFill.getFeatures()[0].m_BottomDistance - meanBottomDistance) < m_ThresFloodDimTol)) {
            count++;
        } else {
            if (!ok) {
                printf("Remove flood fill handle bottom %f %f %d \n", handleFloodFill.getFeatures()[0].m_BottomDistance, meanBottomDistance, m_ThresFloodDimTol);
                qDebug() << "Remove flood fill handle bottom" << handleFloodFill.getFeatures()[0].m_BottomDistance << meanBottomDistance << m_ThresFloodDimTol;
            }

            ok = true;
        }

        if (!ok && (fabs(handleFloodFill.getFeatures()[0].m_LeftDistance - meanLeftDistance) < m_ThresFloodDimTol)) {
            count++;
        } else {
            if (!ok) {
                printf("Remove flood fill handle right %f %f %d \n", handleFloodFill.getFeatures()[0].m_LeftDistance, meanLeftDistance, m_ThresFloodDimTol);
                qDebug() << "Remove flood fill handle right" << handleFloodFill.getFeatures()[0].m_LeftDistance << meanLeftDistance << m_ThresFloodDimTol;
            }

            ok = true;
        }

        if (!ok && (fabs(handleFloodFill.getFeatures()[0].m_RightDistance - meanRightDistance) < m_ThresFloodDimTol)) {
            count++;
        } else {
            if (!ok) {
                printf("Remove flood fill handle top %f %f %d \n", handleFloodFill.getFeatures()[0].m_RightDistance, meanRightDistance, m_ThresFloodDimTol);
                qDebug() << "Remove flood fill handle top" << handleFloodFill.getFeatures()[0].m_RightDistance << meanRightDistance << m_ThresFloodDimTol;
            }
        }

        if (count == 4)
            removeFloodFillHandle = false;
    }

    return removeFloodFillHandle;
}


bool CrateHandleExtractor::differentHandles(const std::vector< CrateHandle >& handles) const
{
    for (int i = 0; i < (int)handles.size(); i++) {
        if (handles[i].getFeatures().size() != 1)
            return false;
    }

    if (handles.size() <= 1)
        return false;

    cv::Point avgCenterMassKMeans(0, 0);

    for (int i = 0; i < (int)handles.size() - 1; i++) {
        avgCenterMassKMeans += handles[i].getFeatures()[0].m_CenterMass;
    }

    avgCenterMassKMeans = cv::Point(avgCenterMassKMeans.x / (handles.size() - 1), avgCenterMassKMeans.y / (handles.size() - 1));

    cv::Point centerMassFloodFill = handles[handles.size() - 1].getFeatures()[0].m_CenterMass;

    double diffX = avgCenterMassKMeans.x - centerMassFloodFill.x;
    double diffY = avgCenterMassKMeans.y - centerMassFloodFill.y;

    if (sqrt(diffX * diffX + diffY * diffY) > 5)
        return true;
    else
        return false;
}

CrateHandle CrateHandleExtractor::findOptimalHandle(const std::vector< CrateHandle >& handles)
{
    std::vector<int> indices(handles.size());

    for (int i = 0; i < (int)indices.size(); i++)
        indices[i] = i;

    std::vector<int> contourIndices = indices;
    std::vector<int> medianIndices = indices;
    std::vector<double> scoreIndices(indices.size(), 0.0);

    ///sort indices by the contour score
    for (int i = 0; i < (int)indices.size(); i++) {
        for (int j = i + 1; j < (int)indices.size(); j++) {
            if (handles[contourIndices[i]].m_HandleFeatures[0].m_SegContourScore
                    >  handles[contourIndices[j]].m_HandleFeatures[0].m_SegContourScore) {
                int temp = contourIndices[i];
                contourIndices[i] = contourIndices[j];
                contourIndices[j] = temp;
            }
        }
    }

    ///sort indices by the median score
    for (int i = 0; i < (int)indices.size(); i++) {
        for (int j = i + 1; j < (int)indices.size(); j++) {
            if (handles[medianIndices[i]].m_HandleFeatures[0].m_SegMedianScore
                    >  handles[medianIndices[j]].m_HandleFeatures[0].m_SegMedianScore) {
                int temp = medianIndices[i];
                medianIndices[i] = medianIndices[j];
                medianIndices[j] = temp;
            }
        }
    }

    ///calculate a total score based on the contour and median scores
    for (int i = 0; i < (int)indices.size(); i++) {
        scoreIndices[contourIndices[i]] += (double)i / 2.0 / (double)indices.size();
        scoreIndices[medianIndices[i]] += (double)i / 2.0 / (double)indices.size();
    }

    double maxScore = 0.0;
    int maxIndex = -1;

    for (int i = 0; i < (int)indices.size(); i++) {
        double score = handles[i].m_HandleFeatures[0].m_SegFormScore * (1.0 - scoreIndices[i]);

//             qDebug() << i << symmetricHandles[i].m_HandleFeatures[0].m_SegFormScore << 1.0 - scoreIndices[i] << "Total score" << score;
        if (score > maxScore) {
            maxScore = score;
            maxIndex = i;
        }
    }

    return handles[maxIndex];
}

CrateHandle CrateHandleExtractor::getFoundHandle()
{
    printf("Get found handle\n");
    cv::imwrite("/home/cucu/temp/foundHandle.png", m_Handle.m_HandleImage);
    return m_Handle;   
}


