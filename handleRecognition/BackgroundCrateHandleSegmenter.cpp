#include "BackgroundCrateHandleSegmenter.h"
#include "OpenCVKMeansClusterer.h"
#include "OpenCVEMClusterer.h"
#include "GenericKMeansImageClusterer.h"
#include "MostRepresentedColors.h"
#include "RGBImageClusteringSegmenter.h"

BackgroundCrateHandleSegmenter::BackgroundCrateHandleSegmenter(const cv::Mat& inputImage, const cv::Mat& originalImage)  : CrateHandleSegmenter(inputImage, originalImage, false, false) 
{
    m_BackgroundComposition = cv::Mat::zeros(m_InputImage.size(), CV_8UC3);
}

bool BackgroundCrateHandleSegmenter::findHandles(std::vector<CrateHandle>& handles)
{

    return true;
}

void BackgroundCrateHandleSegmenter::createClusterers()
{
    for (int i = m_MinClusterNo; i <= m_MaxClusterNo; i++) {
//         m_Clusterers.push_back(new OpenCVKMeansClusterer(m_InputImage, i));
//         m_Clusterers.push_back(new OpenCVEMClusterer(m_InputImage, i));
        ///background segmentation works optimally with the clusterers given below
        m_Clusterers.push_back(new GenericKMeansImageClusterer<VectorDouble>(m_InputImage, i, 1.0));
        m_Clusterers.push_back(new MostRepresentedColors(m_InputImage, 30, 5.0 * (i - 1)));
    }
}

int BackgroundCrateHandleSegmenter::estimateBackgroundImage()
{
    createClusterers();
    int countGreyShades = 0;    
    
    for (auto clusterer : m_Clusterers) {
        RGBImageClusteringSegmenter rics(clusterer);
        printf("Clusterer %p\n", clusterer);
        cv::Mat segmentedImage;
        std::map<cv::Vec3b, ImageSegment, LessVec3b> segments;
        ///@todo: should test if the segmentation was successfull
        rics.execute(segments, segmentedImage);
        cv::Vec3b color;

        if (findBackgroundSegment(segments, color)) {
            m_FoundBackgrounds.push_back(std::make_pair(color, segmentedImage.clone())); ///@todo: is a clone necessary??
            countGreyShades++;
        }
    }
    printf("Count grey shades %d\n", countGreyShades);
    buildCompositionImage(countGreyShades);
    m_FoundBackgrounds.clear();
    for (unsigned int i = 0; i < m_Clusterers.size(); i++)
        delete m_Clusterers[i];
    return countGreyShades;
}

bool BackgroundCrateHandleSegmenter::findBackgroundSegment(const std::map<cv::Vec3b, ImageSegment, LessVec3b>& segments, cv::Vec3b& backId)
{
    if (segments.empty())
        return false;
    printf("Find background segments\n");
    auto biggestSegmentIds = segments.begin();
    for (auto iter = segments.begin(); iter != segments.end(); iter++) {
        if (iter->second.m_NoPoints > biggestSegmentIds->second.m_NoPoints) {
            biggestSegmentIds = iter;
        }
    }
    int imageSize = m_InputImage.rows * m_InputImage.cols;
    printf("Biggest segment has %d . Image size %d \n", biggestSegmentIds->second.m_NoPoints, imageSize);
    if (biggestSegmentIds->second.m_NoPoints > imageSize / 2) {
        backId = biggestSegmentIds->first;
        return true;
    } else {
        return false;
    }    
}

void BackgroundCrateHandleSegmenter::buildCompositionImage(int greyGrades)
{
    if (greyGrades <= 0)
        return;
//     int greyStep = 255 / greyGrades;
    
    for (int i = 0; i < m_BackgroundComposition.rows; i++)
        for (int j = 0; j < m_BackgroundComposition.cols; j++) {
            int count = 0;
            for (auto p : m_FoundBackgrounds) {
                if (p.second.at<cv::Vec3b>(i, j) == p.first)
                    count++;
            }
//             int col = count * greyStep;
//             m_BackgroundComposition.at<cv::Vec3b>(i,j) = cv::Vec3b(col, col, col);
            if (count > int(m_FoundBackgrounds.size() / 2))
                m_BackgroundComposition.at<cv::Vec3b>(i,j) = cv::Vec3b(255, 255, 255);
            else
                m_BackgroundComposition.at<cv::Vec3b>(i,j) = cv::Vec3b(0, 0, 0);            
        }
}
