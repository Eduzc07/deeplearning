#include "TestWithTensorFlowModel.h"

#include <exception>
#include <iostream>
#include <QFile>
#include <QDir>
#include <QTextStream>
#include <QFileInfo>
#include <QDebug>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

bool TestWithTensorFlowModel::readTrainedModel()
{
    try {
        m_Net  = cv::dnn::readNetFromTensorflow(m_TrainedModelPath);
        std::vector<cv::String> layerNames = m_Net.getLayerNames();
        
        for (cv::String layer : layerNames)
            printf("%s\n", layer.c_str());
    } catch(std::exception e) {
        printf("Exception when loading network from tensorflow: %s\n", e.what());
    }
    return true;
}

bool TestWithTensorFlowModel::readTestData()
{
    QDir rootDir(m_TestDataPath);
    QStringList classDirs = rootDir.entryList(QDir::Dirs | QDir::NoDot | QDir::NoDotDot);
    
    for (QString classDirName : classDirs) {
        bool okConv = false;
        int classID =  classDirName.toInt(&okConv);
        if (!okConv)
            continue;
        m_Labels.insert(classID);

        //create the folder in the mirror folder
        QString classDirPath = m_TestDataPath + QDir::separator() + classDirName;
        QDir classDir(classDirPath);
        QStringList imageList = classDir.entryList(QDir::Files);        
        
        for (QString image : imageList) {
            //printf("Class %s : %s \n", qPrintable(classDirName), qPrintable(image));
            m_TestData[classDirPath + QDir::separator() + image] = classID;
        }
    } 
    
    return true;
}

void TestWithTensorFlowModel::testAllTrainingData()
{
    int countTestedImages = 0;
    int countCorrRecog = 0;
    
    std::vector<cv::Mat> imagesBatch;
    std::vector<int> labelsBatch;
    std::vector<QString> pathsBatch;
    int batchCounter = 0;
    
    for (auto testItem : m_TestData) {        
        if (batchCounter > 0 && batchCounter % m_BatchSize == 0) {
            //recognize and update counters
            cv::Mat netInput = cv::dnn::blobFromImages(imagesBatch);                  
            //printf("Input Batch: size %d %d %d %d\n", netInput.size[0], netInput.size[1], netInput.size[2], netInput.size[3]);
            m_Net.setInput(netInput, "input/Identity");        //set the network input
            cv::Mat result = m_Net.forward("output/Softmax");     //compute output
            //printf("Batch: Result size %d %d \n", result.size[0], result.size[1]);
            
            for (int i = 0; i < m_BatchSize; ++i) {
                int maxIdx = 0;
                double maxVal = result.at<float>(i, 0);
                for (int j = 1; j < result.size[1]; ++j) {
                    double val = result.at<float>(i, j);
                    if (val > maxVal) {
                        maxVal = val;
                        maxIdx = j;
                    }
                }
                
                if (labelsBatch[i] == maxIdx) {
                   countCorrRecog++; 
                } else {
                    printf("Not rec: %s\n", qPrintable(pathsBatch[i]));
                    printf("Groundtruth: %d, Recognized %d\n", labelsBatch[i], maxIdx);
                    printf("Softmax probabilities: ");
                    std::vector<std::pair<int, double>> results;
                    for (int j = 0; j < result.size[1]; ++j) {
                        if (result.at<float>(i, j) > 0.00001) {
                            //printf("%d - %f ", j, result.at<float>(i, j));
                            results.push_back(std::make_pair(j, result.at<float>(i, j)));
                        }
                    }
                    
                    std::sort(results.begin(), results.end(), [] (std::pair<int, double> a, std::pair<int, double> b) {
                        return a.second > b.second;
                    });
                    
                    for (auto s : results) {
                        printf("%d - %f ", s.first, s.second);
                    }
                    
                    printf("\n");
                }
            }
            
            batchCounter = 0;
            imagesBatch.clear();
            labelsBatch.clear();
            pathsBatch.clear();
            countTestedImages += m_BatchSize;
        }
        
        cv::Mat img = cv::imread(testItem.first.toUtf8().constData());
        if (img.empty()) {
            std::cerr << "Can't read image from the file: " << testItem.first.toUtf8().constData() << std::endl;
            exit(-1);
        }
        
//         cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        if (m_InputImageSize != img.size())
            cv::resize(img, img, m_InputImageSize); //Resize image to input size       
        imagesBatch.push_back(img);
        labelsBatch.push_back(testItem.second);
        pathsBatch.push_back(testItem.first);
        batchCounter++;
    }
    
    if (countTestedImages)
        printf("%d images read. %d images recognized. %f recognition rate\n", countTestedImages, countCorrRecog, double(countCorrRecog)/double(countTestedImages));
    else
        printf("No image was tested\n");
}

void TestWithTensorFlowModel::testSingleImage()
{
    cv::dnn::Net network;
    
    try {
        network  = cv::dnn::readNetFromTensorflow(m_SingleImageModelPath);
        std::vector<cv::String> layerNames = network.getLayerNames();
        
        for (cv::String layer : layerNames)
            printf("%s\n", layer.c_str());
    } catch(std::exception e) {
        printf("Exception when loading network from tensorflow: %s\n", e.what());
        return;
    }

    cv::Mat img = cv::imread(m_SingleImageTestPath);
    
 /*   cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );// Create a window for display.
    cv::imshow( "Display window", img );                   // Show our image inside it.

    cv::waitKey(0);  */       
    
//     cv::transpose(img, img);
 
    cv::Mat netInput = cv::dnn::blobFromImage(img);
    network.setInput(netInput);
    cv::Mat output = network.forward("output/Softmax");
    
    cv::Mat displayMat  = output;
    
//     for (int i = 0; i < displayMat.size[0]; ++i)
//         for (int j = 0; j < displayMat.size[1]; ++j) {
//             printf("Single channel image\n");
//             for (int k = 0; k < displayMat.size[2]; ++k) {
//                 for (int l = 0; l < displayMat.size[3]; ++l) {
//                     cv::Vec4i idx(i, j, k, l);
//                     printf("%f ", displayMat.at<float>(idx));
//                 }
//                 printf("\n");
//             }
//         }
//      printf("%d %d %d %d\n", displayMat.size[0], displayMat.size[1], displayMat.size[2], displayMat.size[3]);


    for (int i = 0; i < displayMat.size[1]; ++i) {
        printf("%d - %f ", i, displayMat.at<float>(0, i));
    }
    printf("\n");
}
