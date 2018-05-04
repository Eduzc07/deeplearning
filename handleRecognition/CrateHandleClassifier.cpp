#include "CrateHandleClassifier.h"
#include <vector>
#include <utility>
#include <algorithm>
#include <QTime>

#include <cmath>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QDebug>

CrateHandleClassifier::CrateHandleClassifier(): m_Debug(false)
{
}

CrateHandleClassifier::~CrateHandleClassifier()
{
}

void CrateHandleClassifier::setReferenceData(int crateID, int sampleNo, const CLogoHandleData& lhd)
{
    m_DataMap[std::pair<int, int>(crateID, sampleNo)] = lhd;
}

std::vector<std::pair<int, int>> CrateHandleClassifier::classify(const CLogoHandleData& lhd, const std::vector<std::pair<int, int>>& candidates)
{
    std::vector<std::pair<int, int>> targets;
    initTargets(candidates, targets);
    std::vector<std::pair<int, int>> results, results1;

    results1 = classify1(lhd, targets);

    if (!results1.size())
        return results1;

    results = classify4(lhd, results1);

    if (!results.size())
        return results1;

    results1 = classify6(lhd, results);

    if (!results1.size()) {
        return results;
    }

    return results1;
}

void CrateHandleClassifier::initTargets(const std::vector<std::pair<int, int>>& candidates, std::vector<std::pair<int, int>>& targets)
{
    if (candidates.size()) {
        targets = candidates;
    } else {
        std::map<std::pair<int, int>, CLogoHandleData>::iterator it = m_DataMap.begin();

        while (it != m_DataMap.end()) {
            targets.push_back(it->first);
            ++it;
        }
    }
}

std::vector<std::pair<int, int>> CrateHandleClassifier::classify1(const CLogoHandleData& lhd, const std::vector<std::pair<int, int>>& candidates)
{
    std::vector<std::pair<int, int>> targets;
    initTargets(candidates, targets);
    std::vector<std::pair<int, int>> results;

    for (int i = 0; i < (int)targets.size(); i++) {
        if (m_DataMap.find(targets[i]) != m_DataMap.end()) {
            if (classifyBasis1(lhd, m_DataMap[targets[i]]))
                results.push_back(targets[i]);
        }
    }

    return results;
}

bool CrateHandleClassifier::classifyBasis1(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2)
{
    double prob = 0.0;

    if (!lhd1.m_Features.size() || !lhd2.m_Features.size())
        return false;
    
    for (int i = 0; i < (int)lhd1.m_Features.size(); i++)
        for (int j = 0; j < (int)lhd2.m_Features.size(); j++) {
            CLogoHandleFeatures lhf1 = lhd1.m_Features[i];
            CLogoHandleFeatures lhf2 = lhd2.m_Features[j];
            double prob1 = 1.0;
            prob1 *= (1 - fabs(lhf1.m_WidthRatio - lhf2.m_WidthRatio));
            prob1 *= (1 - fabs(lhf1.m_HeightRatio - lhf2.m_HeightRatio));
            prob1 *= (1 - fabs(lhf1.m_TopCornerYHeight - lhf2.m_TopCornerYHeight));
            if (prob1 > prob)
                prob = prob1;
        }
    return  prob > m_ThresClassif1;
}

double CrateHandleClassifier::scorePositions(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2)
{
    double prob = 0.0;

    if (!lhd1.m_Features.size() || !lhd2.m_Features.size())
        return false;
    
    for (int i = 0; i < (int)lhd1.m_Features.size(); i++)
        for (int j = 0; j < (int)lhd2.m_Features.size(); j++) {
            CLogoHandleFeatures lhf1 = lhd1.m_Features[i];
            CLogoHandleFeatures lhf2 = lhd2.m_Features[j];
            double prob1 = 1.0;
            prob1 *= (1 - fabs(lhf1.m_WidthRatio - lhf2.m_WidthRatio));
            prob1 *= (1 - fabs(lhf1.m_HeightRatio - lhf2.m_HeightRatio));
            prob1 *= (1 - fabs(lhf1.m_TopCornerYHeight - lhf2.m_TopCornerYHeight));
            if (prob1 > prob)
                prob = prob1;
        }

    return prob;
}

std::vector<std::pair<int, int>> CrateHandleClassifier::classify4(const CLogoHandleData& lhd, const std::vector<std::pair<int, int>>& candidates)
{
    std::vector<std::pair<int, int>> targets;
    initTargets(candidates, targets);
    std::vector<std::pair<int, int>> results;

    for (int i = 0; i < (int)targets.size(); i++) {
        if (m_DataMap.find(targets[i]) != m_DataMap.end()) {
            if (classifyBasis4(lhd, m_DataMap[targets[i]]))
                results.push_back(targets[i]);
        }
    }

    return results;
}

double CrateHandleClassifier::scoreRelativeDimensions(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2)
{
    double prob = 0.0;

    if (!lhd1.m_Features.size() || !lhd2.m_Features.size())
        return false;
    
    for (int i = 0; i < (int)lhd1.m_Features.size(); i++)
        for (int j = 0; j < (int)lhd2.m_Features.size(); j++) {
            CLogoHandleFeatures lhf1 = lhd1.m_Features[i];
            CLogoHandleFeatures lhf2 = lhd2.m_Features[j];
            std::vector<double> tests;
            
            double prob1 = fabs(lhf1.m_WidthHeight - lhf2.m_WidthHeight);
            tests.push_back(prob);
            prob = fabs(lhf1.m_FillFactor - lhf2.m_FillFactor);
            tests.push_back(prob);
            prob = fabs(lhf1.m_CenterMassHalfWidth - lhf2.m_CenterMassHalfWidth);
            tests.push_back(prob);
            prob = fabs(lhf1.m_CenterMassSectionHeight - lhf2.m_CenterMassSectionHeight);
            tests.push_back(prob);
            prob = fabs(lhf1.m_CenterMassBottomSection - lhf2.m_CenterMassBottomSection);
            tests.push_back(prob);
            prob = fabs(lhf1.m_BottomHeight - lhf2.m_BottomHeight);
            tests.push_back(prob);
            prob = fabs(lhf1.m_CenterMassLatSectionHeight - lhf2.m_CenterMassLatSectionHeight);
            tests.push_back(prob);
            prob = fabs(lhf1.m_CenterMassLatBottomSection - lhf2.m_CenterMassLatBottomSection);

            std::sort(tests.begin(), tests.end());

//            prob1 = 1.0;

            for (int i = 0; i < 5; i++)
                prob1 *= (1 - tests[i]);
            
            if (prob1 > prob)
                prob = prob1;
        }

    return prob;
}

double CrateHandleClassifier::scoreRelativeDimensions1(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2, bool allFeatures)
{
    double prob = 0.0;

    if (!lhd1.m_HasHandle || !lhd2.m_HasHandle) {
        printf("Crate has no handle\n");
        return 0.0;
    }
    
    if (!lhd1.m_Features.size() || !lhd2.m_Features.size()) {
        printf("Handle has no features\n");
        return 0.0;
    }
    
    int stopi = 1;
    int stopj = 1;
    
    if (allFeatures) {
        stopi = (int)lhd1.m_Features.size();
        stopj = (int)lhd2.m_Features.size();
    }

     for (int i = 0; i < stopi; i++)
         for (int j = 0; j < stopj; j++) { 
            CLogoHandleFeatures lhf1 = lhd1.m_Features[i];
            CLogoHandleFeatures lhf2 = lhd2.m_Features[j];
                
            double prob1 = 1.0;

            prob1 *= (1.0 - fabs(lhf1.m_WidthHeight - lhf2.m_WidthHeight));
            prob1 *= (1.0 - fabs(lhf1.m_FillFactor - lhf2.m_FillFactor));
            prob1 *= (1.0 - fabs(lhf1.m_CenterMassHalfWidth - lhf2.m_CenterMassHalfWidth));
            prob1 *= (1.0 - fabs(lhf1.m_CenterMassSectionHeight - lhf2.m_CenterMassSectionHeight));
            prob1 *= (1.0 - fabs(lhf1.m_CenterMassBottomSection - lhf2.m_CenterMassBottomSection));
            prob1 *= (1.0 - fabs(lhf1.m_BottomHeight - lhf2.m_BottomHeight));
            prob1 *= (1.0 - fabs(lhf1.m_CenterMassLatSectionHeight - lhf2.m_CenterMassLatSectionHeight));
            prob1 *= (1.0 - fabs(lhf1.m_CenterMassLatBottomSection - lhf2.m_CenterMassLatBottomSection));
            
            if (prob1 > prob)
                prob = prob1;
        }
            
    return prob;
}

bool CrateHandleClassifier::classifyBasis4(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2)
{
    return (scoreRelativeDimensions(lhd1, lhd2) > m_ThresClassif4);
}

std::vector<std::pair<int, int>> CrateHandleClassifier::classify6(const CLogoHandleData& lhd, const std::vector<std::pair<int, int>>& candidates)
{
    std::vector<std::pair<int, int>> targets;
    initTargets(candidates, targets);
    std::vector<std::pair<int, int>> results;

    for (int i = 0; i < (int)targets.size(); i++) {
        if (m_DataMap.find(targets[i]) != m_DataMap.end()) {
            if (classifyBasis6(lhd, m_DataMap[targets[i]]))
                results.push_back(targets[i]);
        }
    }

    return results;
}

bool CrateHandleClassifier::classifyBasis6(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2)
{
    return (scoreIntersect(lhd1, lhd2, 0) > m_ThresClassif62);
}


void CrateHandleClassifier::initTargetsTrain(const std::vector<std::pair<int, int>>& candidates, std::vector<std::pair<int, int>>& targets)
{
    if (candidates.size()) {
        targets = candidates;
    } else {
        std::multimap<std::pair<int, int>, CLogoHandleData>::iterator it = m_TrainData.begin();

        while (it != m_TrainData.end()) {
            if (std::find(targets.begin(), targets.end(), it->first) == targets.end())
                targets.push_back(it->first);

            ++it;
        }
    }
}

std::vector<std::pair<int, int>> CrateHandleClassifier::classifyTrainedExistSimilar(const CLogoHandleData& lhd, const std::vector<std::pair<int, int>>& candidates)
{
    std::vector<std::pair<int, int>> targets;
    initTargetsTrain(candidates, targets);
    std::vector<std::pair<int, int>> results;

    m_ThresClassif1 += 0.1;
    m_ThresClassif4 += 0.1;
    m_ThresClassif61 += 0.1;
    m_ThresClassif62 += 0.1;

    for (int i = 0; i < (int)targets.size(); i++) {
        std::pair<std::multimap<std::pair<int, int>, CLogoHandleData>::const_iterator, std::multimap<std::pair<int, int>, CLogoHandleData>::const_iterator> itpair1 = m_TrainData.equal_range(targets[i]);
        std::multimap<std::pair<int, int>, CLogoHandleData>::const_iterator it1 = itpair1.first;
        bool found = false;

        while (it1 != itpair1.second) {
            if (/*classifyBasis6(lhd, it1->second,1) || */(classifyBasis1(lhd, it1->second) && classifyBasis4(lhd, it1->second) && classifyBasis6(lhd, it1->second))) {
                found = true;
                break;
            }

            ++it1;
        }

        if (found)
            results.push_back(targets[i]);
    }

    m_ThresClassif1 -= 0.1;
    m_ThresClassif4 -= 0.1;
    m_ThresClassif61 -= 0.1;
    m_ThresClassif62 -= 0.1;

    return results;
}

std::vector<std::pair<int, int>> CrateHandleClassifier::classifyTrainedStatistical(const CLogoHandleData& lhd, const std::vector<std::pair<int, int>>& candidates)
{
    std::vector<std::pair<int, int>> targets;
    initTargetsTrain(candidates, targets);
    std::vector<std::pair<int, int>> results;

    for (int i = 0; i < (int)targets.size(); i++) {
        std::pair<std::multimap<std::pair<int, int>, CLogoHandleData>::const_iterator, std::multimap<std::pair<int, int>, CLogoHandleData>::const_iterator> itpair1 = m_TrainData.equal_range(targets[i]);
        std::multimap<std::pair<int, int>, CLogoHandleData>::const_iterator it1 = itpair1.first;
        int countFound = 0;
        int countTotal = 0;

        while (it1 != itpair1.second) {
            if (classifyBasis1(lhd, it1->second) && classifyBasis4(lhd, it1->second) && classifyBasis6(lhd, it1->second)) {
                countFound++;
                break;
            }

            countTotal++;
            ++it1;
        }

        if ((double)countFound / (double)countTotal > 0.85)
            results.push_back(targets[i]);
    }

    return results;
}

std::vector<std::pair<int, int>> CrateHandleClassifier::classifyTrainedNearestNeighbourStatistical(const CLogoHandleData& lhd, const std::vector< std::pair< int, int > >& candidates)
{
    std::vector<std::pair<int, int>> targets;
    initTargetsTrain(candidates, targets);
    std::vector<std::pair<int, int>> results;
    std::vector<std::pair<std::pair<int, int>, double>> partialResults;

    for (int i = 0; i < (int)targets.size(); i++) {
        std::pair<std::multimap<std::pair<int, int>, CLogoHandleData>::const_iterator, std::multimap<std::pair<int, int>, CLogoHandleData>::const_iterator> itpair1 = m_TrainData.equal_range(targets[i]);
        std::multimap<std::pair<int, int>, CLogoHandleData>::const_iterator it1 = itpair1.first;
        int countFound = 0;
        int countTotal = 0;

        while (it1 != itpair1.second) {
            if (classifyBasis1(lhd, it1->second) && classifyBasis4(lhd, it1->second) /*&& classifyBasis6(lhd, it1->second)*/) {
                countFound++;
                break;
            }

            countTotal++;
            ++it1;
        }

        partialResults.push_back(std::pair<std::pair<int, int>, double>(targets[i], (double)countFound / (double)countTotal));
    }

    std::sort(partialResults.begin(), partialResults.end(), [](std::pair<std::pair<int, int>, double> a, std::pair<std::pair<int, int>, double> b) { return (a.second > b.second); });

    int index = 0;

    while (true) {
        if (partialResults[index].second > 0.7 || index < 5)
            results.push_back(partialResults[index].first);
        else
            break;

        index++;
    }

    return results;
}

std::vector<std::pair<double, double>> CrateHandleClassifier::calculateContour(const std::vector<QPoint>& contour, const QPoint& centerMass, const QPoint& centerMassLeft, const QPoint& centerMassRight, double& dist)
{
    std::vector<std::pair<double, double>> result;

    for (int i = 0; i < (int)contour.size(); i++)
        result.push_back(std::pair<double, double>(contour[i].x() - centerMass.x(), contour[i].y() - centerMass.y()));

    std::pair<double, double> centerMassLeft1 = std::pair<double, double>(centerMassLeft.x() - centerMass.x(), centerMassLeft.y() - centerMass.y());
    std::pair<double, double> centerMassRight1 = std::pair<double, double>(centerMassRight.x() - centerMass.x(), centerMassRight.y() - centerMass.y());
    double distLeft = sqrt(centerMassLeft1.first * centerMassLeft1.first + centerMassLeft1.second * centerMassLeft1.second);
    double distRight = sqrt(centerMassRight1.first * centerMassRight1.first + centerMassRight1.second * centerMassRight1.second);
    dist = (distLeft + distRight) / 2.0;
    return result;
}

void CrateHandleClassifier::scaleContour(std::vector<std::pair<double, double>>& contour, double factorX, double factorY)
{
    for (int i = 0; i < (int)contour.size(); i++) {
        contour[i] = std::pair<double, double>((double)contour[i].first * factorX, (double)contour[i].second * factorY);
    }
}

void CrateHandleClassifier::calculateSections(const std::vector<std::pair<double, double>>& contour1, int& left, int& right, std::map<int, double>& sectionsMin, std::map<int, double>& sectionsMax)
{
    double prev_x = 0.0, prev_y = 0.0;

    for (int i = 0; i < (int)contour1.size(); i++) {
        if (floor(contour1[i].first) < (double)left)
            left = (int)contour1[i].first;

        if (floor(contour1[i].first) > (double)right)
            right = (int)contour1[i].first;

        if (i && fabs(floor(contour1[i].first) - floor(prev_x)) > 1) {
            double min = floor(contour1[i].first);
            double max = floor(prev_x);
            double start_y = contour1[i].second;
            double stop_y = prev_y;

            if (floor(prev_x) < min) {
                min = floor(prev_x);
                max = floor(contour1[i].first);
                start_y = prev_y;
                stop_y = contour1[i].second;
            }

            ///interpolate missing points
            for (int j = min; j < max; j++) {
                addSectionElement(j, start_y + (stop_y - start_y) / (fabs(floor(contour1[i].first) - prev_x)) * ((double)j - min), sectionsMin, sectionsMax);
            }
        } else {
            addSectionElement(contour1[i].first, contour1[i].second, sectionsMin, sectionsMax);
        }

        prev_x = contour1[i].first;
        prev_y = contour1[i].second;
    }
}

void CrateHandleClassifier::addSectionElement(double x, double y, std::map< int, double >& sectionsMin, std::map< int, double >& sectionsMax)
{
    std::map<int, double>::iterator it1 = sectionsMin.find((int)x);

    if ((it1 == sectionsMin.end()) || (it1 != sectionsMin.end() && y < it1->second)) {
        sectionsMin[(int)x] = y;
    }

    std::map<int, double>::iterator it2 = sectionsMax.find((int)x);

    if ((it2 == sectionsMax.end()) || (it2 != sectionsMax.end() && y > it2->second)) {
        sectionsMax[(int)x] = y;
    }
}

double CrateHandleClassifier::calculateIntersectArea(const std::vector< std::pair< double, double > >& contour1, const std::vector< std::pair< double, double > >& contour2, double& area1, double& area2)
{
    int left = 1000;
    int right = -1000;
    std::map<int, double> sections1Min;
    std::map<int, double> sections1Max;
    calculateSections(contour1, left, right, sections1Min, sections1Max);
    std::map<int, double> sections2Min;
    std::map<int, double> sections2Max;
    calculateSections(contour2, left, right, sections2Min, sections2Max);

    double intersectArea = 0;
    std::vector<double> diffs1, diffs2;

    for (int i = left; i <= right; i++) {
        double min1 = 0.0, max1 = 0.0, min2 = 0.0, max2 = 0.0;

        if (sections1Min.find(i) != sections1Min.end() && sections1Max.find(i) != sections1Max.end()) {
            min1 = sections1Min.find(i)->second;
            max1 = sections1Max.find(i)->second;
            area1 += (max1 - min1);

            if (max1 < min1)  {
                printf("Error %f %f\n", min1, max1);
                exit(1);
            }
        }

        if (sections2Min.find(i) != sections2Min.end() && sections2Max.find(i) != sections2Max.end()) {
            min2 = sections2Min.find(i)->second;
            max2 = sections2Max.find(i)->second;
            area2 += (max2 - min2);

            if (max2 < min2)  {
                printf("Error %f %f\n", min2, max2);
                exit(1);
            }
        }

        if (sections1Min.find(i) == sections1Min.end() || sections2Min.find(i) == sections2Min.end()
                || sections1Max.find(i) == sections1Max.end() || sections2Max.find(i) == sections2Max.end())
            continue;

//          printf("%d %f %f %f %f \n", i, min1, max1, min2, max2);

        if (max1 < min2 || max2 < min1)
            continue;

        double min = std::max(min1, min2);
        double max = std::min(max1, max2);

        intersectArea += max - min;
    }

//     printf("--------------------------");

    return intersectArea;
}

double CrateHandleClassifier::scoreIntersect(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2, int count)
{
    if (!lhd1.m_Features.size() || !lhd2.m_Features.size())
        return 0.0;
    
    double dist1 = 0.0;
    std::vector< std::pair< double, double>> contour1 = calculateContour(lhd1.m_Features[0].m_Contour, lhd1.m_Features[0].m_CenterMass, lhd1.m_Features[0].m_CenterMassLeft, lhd1.m_Features[0].m_CenterMassRight, dist1);

    double dist2 = 0.0;
    std::vector<std::pair<double, double>> contour2 = calculateContour(lhd2.m_Features[0].m_Contour, lhd2.m_Features[0].m_CenterMass, lhd2.m_Features[0].m_CenterMassLeft, lhd2.m_Features[0].m_CenterMassRight, dist2);

    std::vector<std::pair<double, double >> contour21 = contour2;

    scaleContour(contour2, dist1 / dist2, dist1 / dist2);

    for (int i = 0; i < (int)contour1.size(); i++)
        contour1[i] = std::pair<double, double>(contour1[i].first + (double)lhd1.m_Features[0].m_CenterMass.x(), contour1[i].second + (double)lhd1.m_Features[0].m_CenterMass.y());


    for (int i = 0; i < (int)contour2.size(); i++)
        contour2[i] = std::pair<double, double>(contour2[i].first + (double)lhd1.m_Features[0].m_CenterMass.x(), contour2[i].second + (double)lhd1.m_Features[0].m_CenterMass.y());


    double area1 = 0.0, area2 = 0.0;
    double intersectArea = calculateIntersectArea(contour1, contour2, area1, area2);

    double res2 = intersectArea / area2;
    double res1 = intersectArea / area1;
    double res = 0.0;

    if (res1 < res2)
        res = res1;
    else
        res = res2;

    if (m_Debug) {
        std::vector<cv::Point> cvContour1;

        for (int i = 0; i < (int)contour1.size(); i++)
            cvContour1.push_back(cv::Point(contour1[i].first, contour1[i].second));

        std::vector<cv::Point> cvContour2;

        for (int i = 0; i < (int)contour2.size(); i++)
            cvContour2.push_back(cv::Point(contour2[i].first, contour2[i].second));

        cv::Mat img1 = cv::Mat::zeros(100, 200, CV_8UC1);
        std::vector<std::vector<cv::Point>> contours1;
        contours1.push_back(cvContour1);
        cv::drawContours(img1, contours1, -1, cv::Scalar(255, 255, 255));
        std::vector<std::vector<cv::Point>> contours2;
        contours2.push_back(cvContour2);
        cv::drawContours(img1, contours2, -1, cv::Scalar(128, 128, 128));
        cv::imwrite(qPrintable(QString("/home/cucu/TesteHandleRecognition/Contours/") + QString::number(count) + QString("-") + QString::number(res)
                               + QString("-") + QString::number(intersectArea) + QString("-")
                               + QString::number(area1) + QString("-") + QString::number(area2) + QString(".png")), img1);

        //printf(" %d %f %f %f %f \n", count, res, intersectArea, area1, area2);
    }

    return res;
}

double CrateHandleClassifier::scoreShapeCV(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2)
{
    if (!lhd1.m_Features.size() || !lhd2.m_Features.size())
        return 0.0;
    
    std::vector<cv::Point> contour1;

    for (int i = 0; i < (int)lhd1.m_Features[0].m_Contour.size(); i++) {
        contour1.push_back(cv::Point(lhd1.m_Features[0].m_Contour[i].x(), lhd1.m_Features[0].m_Contour[i].y()));
    }

    std::vector<cv::Point> contour2;

    for (int i = 0; i < (int)lhd2.m_Features[0].m_Contour.size(); i++) {
        contour2.push_back(cv::Point(lhd2.m_Features[0].m_Contour[i].x(), lhd2.m_Features[0].m_Contour[i].y()));
    }

    double res = cv::matchShapes(contour1, contour2, CV_CONTOURS_MATCH_I3, 0);

    return (1.0 - res);
}

std::map<int, double> CrateHandleClassifier::classifRelativeDims(const CLogoHandleData& lhd)
{
    std::map<int, double> res;

/*    CLogoHash* logoHash = m_LogoDB->getLogos();
    CLogoHash::iterator it = logoHash->begin();
    CLogoCrateData* crate = nullptr;
    
    printf("Classify relative dims \n");

    while (it != logoHash->end()) {
        crate = it->second;
        std::vector<CLogoSampleData*> samples = crate->m_Samples;

        for (int i = 0; i < (int)samples.size(); i++) {
            CLogoSampleData* sample = samples[i];
            ClcLogoSampleData* lcSample = dynamic_cast<ClcLogoSampleData*>(sample);

            if (!lcSample) {
                printf("Error 2 when openinng logo database\n");
                exit(1);
            }

            CLogoHandleData lhd1 = lcSample->getLogoHandleData();

            if (lhd1.m_HasHandle) {
                double prob = scoreRelativeDimensions1(lhd, lhd1);
                int handleType = static_cast<int>(lhd1.m_HandleType);

                if (res.find(handleType) == res.end()) {
                    res[handleType] = prob;
                } else {
                    if (prob > res[handleType])
                        res[handleType] = prob;
                }
            }
        }

        ++it;
    }
*/
    printf("Handle compare scores: %f %f %f\n", res[0], res[1], res[2]);
    return res;
}

std::map<int, double> CrateHandleClassifier::classifForm(const CLogoHandleData& lhd)
{
    std::map<int, double> res;

/*    CLogoHash* logoHash = m_LogoDB->getLogos();
    CLogoHash::iterator it = logoHash->begin();
    CLogoCrateData* crate = nullptr;

    while (it != logoHash->end()) {
        crate = it->second;
        std::vector<CLogoSampleData*> samples = crate->m_Samples;

        for (int i = 0; i < (int)samples.size(); i++) {
            CLogoSampleData* sample = samples[i];
            ClcLogoSampleData* lcSample = dynamic_cast<ClcLogoSampleData*>(sample);

            if (!lcSample) {
                printf("Error 2 when openinng logo database\n");
                exit(1);
            }

            CLogoHandleData lhd1 = lcSample->getLogoHandleData();

            if (lhd1.m_HasHandle) {
                double prob = scoreIntersect(lhd1, lhd, 0);
                int handleType = static_cast<int>(lhd1.m_HandleType);

                if (res.find(handleType) == res.end()) {
                    res[handleType] = prob;
                } else {
                    if (prob > res[handleType])
                        res[handleType] = prob;
                }
            }
        }

        ++it;
    }
*/
    return res;
}

std::map<int, double> CrateHandleClassifier::classifMoments(const CLogoHandleData& lhd)
{
    std::map<int, double> res;

/*    CLogoHash* logoHash = m_LogoDB->getLogos();
    CLogoHash::iterator it = logoHash->begin();
    CLogoCrateData* crate = nullptr;

    while (it != logoHash->end()) {
        crate = it->second;
        std::vector<CLogoSampleData*> samples = crate->m_Samples;

        for (int i = 0; i < (int)samples.size(); i++) {
            CLogoSampleData* sample = samples[i];
            ClcLogoSampleData* lcSample = dynamic_cast<ClcLogoSampleData*>(sample);

            if (!lcSample) {
                printf("Error 2 when openinng logo database\n");
                exit(1);
            }

            CLogoHandleData lhd1 = lcSample->getLogoHandleData();

            if (lhd1.m_HasHandle) {
                double prob = scoreShapeCV(lhd1, lhd);
                int handleType = static_cast<int>(lhd1.m_HandleType);

                if (res.find(handleType) == res.end()) {
                    res[handleType] = prob;
                } else {
                    if (prob > res[handleType])
                        res[handleType] = prob;
                }
            }
        }

        ++it;
    }
*/
    return res;
}

