#include "CreateBackgroundSubtractionTest.h"

#include <QFileInfo>
#include <QDir>
#include <QDebug>
#include <QDateTime>
#include <QImage>
#include <QRegExp>
#include <algorithm>
#include <thread>

#include "CVMatQImageConversion.h"
#include "BackgroundCrateHandleSegmenter.h"
#include "ConnectedComponentLabelling.h"

CreateBackgroundSubtractionTest::CreateBackgroundSubtractionTest()
{    
//     m_DataFolders.push_back("herrieden-rechts-copy-more20/");
//     m_DataFolders.push_back("Training-apolda-links/");
//     m_DataFolders.push_back("Training-apolda-rechts/");
//     m_DataFolders.push_back("weilheim-links-copy/");
//     m_DataFolders.push_back("test191/");
    m_DataFolders.push_back("Freilassing1/");
//     m_DataFolders.push_back("Freilassing2/");
//     m_DataFolders.push_back("Freilassing3/");
//     m_DataFolders.push_back("Freilassing4/");
}

void CreateBackgroundSubtractionTest::processTestFolder(const QString& folder)
{
    QString dirPath = m_RootPath + folder;
    QFileInfo qfi(dirPath);
    if (!qfi.exists())
        return;
    if (!qfi.isDir())
        return;
    
    QString transformedBaseDirPath = m_RootPath + m_TestFolder;
    QDir dirBase(transformedBaseDirPath);
    dirBase.mkdir(folder);
    
    QDir dir(dirPath);
    QStringList imageDirs = dir.entryList(QDir::Dirs | QDir::NoDot | QDir::NoDotDot);
    
    for (QString imageDirName : imageDirs) {
        //create the folder in the mirror folder
        QString classDirPath = dirPath + imageDirName;
        if (m_ClassFilter.indexIn(imageDirName) != -1)
            processImagesFolder(folder, classDirPath);
    }
}

/**
 * @param[in] - testFolder e.g. Freilassing1 - market
 * @param[in] - folder e.g. 9 - class ID
 */

void CreateBackgroundSubtractionTest::processImagesFolder(const QString& testFolder, const QString& folder)
{   
    QDir imageDir(folder);
    QDir marketDir(testFolder);
    QString momentsFileName = m_RootPath + "/" + m_MomentsFileName + "_" + marketDir.dirName() + "_" + imageDir.dirName() + "_" + QDateTime::currentDateTime().toString("yyyyMMdd_hh:mm:ss") + ".txt";
    qDebug() << "moments file " << momentsFileName;
    m_MomentsFile = new QFile(momentsFileName);
    if (m_MomentsFile->open(QIODevice::WriteOnly)) {
        m_MomentsStream = new QTextStream(m_MomentsFile);
    } else {
        qDebug() << "cannot open moments file " << m_MomentsFileName;
        exit(1);
    }
    qDebug() << "Process Market:" <<  testFolder << " Class: " << folder << " Moments file: " << momentsFileName;
    
    ///creates the images 
    QDir dir(folder);
    if (!dir.exists())                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        return;
    QString dirName = dir.dirName();
    QString transformedBaseDirPath = m_RootPath + m_TestFolder + testFolder + "/";                                                                                                          
    QDir baseDir(transformedBaseDirPath);
    if (!baseDir.exists())
        return;
    baseDir.mkdir(dirName);

    qDebug() << "Created " << dirName << " in " << transformedBaseDirPath;
    
    QStringList imageList = dir.entryList(QDir::Files);
    
//     int num_threads;
//     std::thread t[num_threads];
//  
//     //Launch a group of threads
//     for (int i = 0; i < num_threads; ++i) {
//         t[i] = std::thread(processImage, m_MomentsStream, i, num_threads);
//     }
//  
// 
//     //Join the threads with the main thread
//     for (int i = 0; i < num_threads; ++i) {
//             t[i].join();
//     }    
    
    
    for (QString imagePath : imageList) {
        //create the folder in the mirror folder
        qDebug() << "Processing " << folder + "/" + imagePath;
        if (!imagePath.endsWith(".ppm"))
            continue;
        if (imagePath.contains("ignore"))
            continue;
        QImage img(folder + "/" + imagePath);

        QString transformedFilePath = transformedBaseDirPath + dirName + "/" + QFileInfo(imagePath).fileName();
        if (QFileInfo(transformedFilePath).exists()) {
            qDebug() << "Path already exists: " << transformedFilePath << "Doing nothing .. ";
            continue;
        }

        img = img.scaled(QSize(200, 200));
        cv::Mat cvImg = CVMatQImageConversion::QImage2Mat(img);
        BackgroundCrateHandleSegmenter bchs(cvImg, cvImg);
        int count = bchs.estimateBackgroundImage();

        QImage convImg;

        if (count < 1) {
            convImg = CVMatQImageConversion::Mat2QImage(bchs.getBackgroundCompositionImage());            
            convImg.save(transformedFilePath);
            continue;
        }
        
        bool calcContours = false; 
        bool calcSegments = true;
        if (calcContours) {
            cv::Mat bcompImg = bchs.getBackgroundCompositionImage();
            cv::Mat bcompImg_gray;
            cvtColor(bcompImg, bcompImg_gray, CV_BGR2GRAY);
            cv::Mat canny;
            cv::Canny(bcompImg_gray, canny, 50, 100);
            std::vector<std::vector<cv::Point>> contours;
            std::vector<cv::Vec4i> hierarchy;       
            cv::findContours(canny, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
            cv::Mat drawing = cv::Mat::zeros(canny.size(), CV_8UC3);
            for (unsigned int i = 0; i < contours.size(); i++) {
                if (contours[i].size() > 100) 
                    cv::drawContours(drawing, contours, i, cv::Scalar(255,255,255,255));
            }
            convImg = CVMatQImageConversion::Mat2QImage(drawing);
        } else if (calcSegments) {
            cv::Mat bcompImg = bchs.getBackgroundCompositionImage();
            ConnectedComponentLabelling ccl(bcompImg, bcompImg);
            ///complex data structure where the information about each segment after segmentation are saved.
            std::map<cv::Vec3b, ImageSegment, LessVec3b> segments;
            cv::Mat segmentedImage;
            ccl.execute(segments, segmentedImage);
            std::vector<std::vector<double>> segFeatures;
            for (auto s: segments) {
//                 printf("Points: %d \n", s.second.m_NoPoints);
                if (s.second.m_NoPoints < 200)
                    continue;
                cv::Mat segImg = cv::Mat::zeros(bcompImg.size(), CV_8UC1);
                for (int i = s.second.m_Left; i <= s.second.m_Right; i++)
                    for (int j = s.second.m_Top; j <= s.second.m_Bottom; j++) {
                        if (segmentedImage.at<cv::Vec3b>(j, i) == s.first)
                            segImg.at<uchar>(j, i) = 1;
                    }
                cv::Moments m = cv::moments(segImg);
                std::vector<double> features(10, 0.0);
                features[0] = m.m00/40000.0;
                features[1] = m.m01/m.m00/200.0;
                features[2] = m.m10/m.m00/200.0;
                features[3] = m.nu20;
                features[4] = m.nu11;
                features[5] = m.nu02;
                features[6] = m.nu30;
                features[7] = m.nu21;
                features[8] = m.nu12;
                features[9] = m.nu03;
                segFeatures.push_back(features);
            }
            
            if (!segFeatures.empty()) {
                std::sort(segFeatures.begin(), segFeatures.end(), [](const std::vector<double>& a, const std::vector<double>&  b) -> bool { return a[0] > b[0]; });
                printf("BlaBla %s\n", qPrintable(transformedFilePath));
                *m_MomentsStream << transformedFilePath.replace("//", "/") << endl;
            } else {
                *m_MomentsStream << transformedFilePath.replace("//", "/") << " Nothing" << endl;
            }
            
            for (auto f: segFeatures) {
                printf("BlaBla %f %f %f %f %f %f %f %f %f %f\n", f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8], f[9]);
                *m_MomentsStream << f[0] << " " << f[1] << " " << f[2] << " " << f[3] << " " << f[4] << " ";
                *m_MomentsStream << f[5] << " " << f[6] << " " << f[7] << " " << f[8] << " " << f[9] << endl;
            }
            
            convImg = CVMatQImageConversion::Mat2QImage(segmentedImage);            
        } else {
            convImg = CVMatQImageConversion::Mat2QImage(bchs.getBackgroundCompositionImage());            
        }        
        qDebug() << "Saving " << transformedFilePath;
        convImg.save(transformedFilePath);        
    }

    m_MomentsStream->flush();
    delete m_MomentsStream;
    m_MomentsStream = nullptr;
    delete m_MomentsFile;
    m_MomentsFile = nullptr;
}


void CreateBackgroundSubtractionTest::execute(const QString& classFilter)
{
    m_ClassFilter = QRegExp(classFilter, Qt::CaseSensitive, QRegExp::RegExp);
    for (QString folderName : m_DataFolders) {
        processTestFolder(folderName);
    }
}
