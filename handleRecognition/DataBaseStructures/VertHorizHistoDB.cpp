#include "VertHorizHistoDB.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <QDir>
#include <QStringList>
#include <numeric>

VertHorizHistograms::VertHorizHistograms(const cv::Mat& image)  {
    m_MaxValX = image.cols;
    m_MaxValY = image.rows;
    
    m_BackgroundBright = std::vector<bool>(m_Channels, false);
    m_ForegroundPointsCount = std::vector<int>(m_Channels, 0);
    
    for (int i = 0; i < m_Channels; ++i) {
        m_VertHistogram.push_back(std::vector<int>(m_MaxValY + 1, 0));  ///should be m_MaxVal - m_MinVal + 1
        m_HorizHistogram.push_back(std::vector<int>(m_MaxValX + 1, 0));
    }

    ///@todo: do we really need this initialization?
    std::vector<cv::Mat> tmpMatVect;
    cv::Mat tmpMat;

    cv::cvtColor(image, tmpMat, CV_BGR2Lab);
    cv::split(tmpMat, tmpMatVect);
    m_SingleChannelImages.push_back(tmpMatVect[0]);

    std::vector<cv::Mat> tmpMatVect1;
    cv::Mat tmpMat1;
    
    cv::cvtColor(image, tmpMat1, CV_BGR2HSV);
    cv::split(tmpMat1, tmpMatVect1);
    m_SingleChannelImages.push_back(tmpMatVect1[2]);
    
//     cv::split(image, m_SingleChannelImages); 
    cv::Mat img = cv::Mat::zeros(image.size(), CV_8UC1);
    cv::cvtColor(image, img, CV_BGR2GRAY);
    m_SingleChannelImages.push_back(img);
    
    generateOtsuImages();
    generateHistograms();
    computeHistoPointCount(m_Channels - 1);
}

void VertHorizHistograms::generateOtsuImages()
{
    for (int i = 0; i < m_Channels; ++i) {
        cv::Mat binImg = cv::Mat::zeros(m_SingleChannelImages[i].size(), CV_8UC1);
        cv::threshold(m_SingleChannelImages[i], binImg, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
        m_BinarisedImages.push_back(binImg);
        cv::Scalar meanVal = cv::mean(m_BinarisedImages[i]);
        if (meanVal[0] > 128)
            m_BackgroundBright[i] = true;
        else 
            m_BackgroundBright[i] = false;
    }
}

void VertHorizHistograms::generateHistograms()
{
    for (int i = 0; i < m_Channels; ++i) {
        generateVertHistograms(m_BinarisedImages[i], m_VertHistogram[i], m_BackgroundBright[i]);
        cv::Mat transp;
        cv::transpose(m_BinarisedImages[i], transp);
        generateVertHistograms(transp, m_HorizHistogram[i], m_BackgroundBright[i]);
    }
}

void VertHorizHistograms::generateVertHistograms(const cv::Mat& binImg, std::vector<int>& histo, bool backgroundBright)
{
    unsigned char background = backgroundBright ? 255 : 0;
    for (int i = 0; i < binImg.cols; ++i) {
        int counter = 0;
        bool started = false;
        for (int j = 0; j < binImg.rows; ++j) {
            if (binImg.at<unsigned char>(j, i) != background) {
                counter++;
            } else {
                if (started) {
                    if (counter > 0)
                        histo[counter]++;
                }
                counter = 0;
                started = true;
            }
        }
    }
}

cv::Mat VertHorizHistograms::getVertHistAsImage(int index) {
    cv::Mat retMat = cv::Mat(200, m_MaxValY + 1, CV_8UC3, cv::Scalar(255, 255, 255));
    for (unsigned int i = 0; i < m_VertHistogram[index].size(); ++i) {
        cv::line(retMat, cv::Point(i, 200), cv::Point(i, 200 - m_VertHistogram[index][i]), cv::Scalar(0, 0, 0));
    }
    return retMat;
}

cv::Mat VertHorizHistograms::getHorizHistAsImage(int index) {
    cv::Mat retMat = cv::Mat(200, m_MaxValX + 1, CV_8UC3, cv::Scalar(255, 255, 255));
    for (unsigned int i = 0; i < m_HorizHistogram[index].size(); ++i) {
        cv::line(retMat, cv::Point(i, 200), cv::Point(i, 200 - m_HorizHistogram[index][i]), cv::Scalar(0, 0, 0));
    }
    return retMat;
}

void VertHorizHistograms::computeHistoPointCount(int index)
{
    m_ForegroundPointsCount[index] = 0;
    for (unsigned int i = 0; i < m_VertHistogram[index].size(); ++i)
        m_ForegroundPointsCount[index] += m_VertHistogram[index][i];
}

void VertHorizHistoDataBase::train(const QString& path, int maxClassId)
{
    m_DB.clear();
    QDir rootDir(path);
    QStringList imageDirs = rootDir.entryList(QDir::Dirs | QDir::NoDot | QDir::NoDotDot);
    
    for (QString imageDirName : imageDirs) {
        //create the folder in the mirror folder
        bool okConvert = false;
        int crateID = imageDirName.toInt(&okConvert);
        if (!okConvert) 
            continue;
        if (crateID > maxClassId) {
            printf("Skipping %d\n", crateID);
            continue;
        }
        printf("Training %d\n", crateID);
        QString classDirPath = path + "/" + imageDirName;
        QDir classDir(classDirPath);
        QStringList images = classDir.entryList(QDir::Files);
        for (QString imageName : images)  {
            QString imagePath = classDirPath + "/" + imageName;
            if (imageName.contains("_ignore_"))
                continue;
            printf("Train %s\n", qPrintable(imagePath));
            std::string cvImagePath = imagePath.toUtf8().constData();
            cv::Mat cvImg = cv::imread(cvImagePath);
            cv::Mat cvImg1;
            cv::resize(cvImg, cvImg1, cv::Size(200,200));
            VertHorizHistograms vhh(cvImg1);
            VertHorizHistoDBEntry vhh_entry;
            vhh_entry.m_ID = crateID;
            vhh_entry.m_VertHist = vhh.getVertHistograms();
            vhh_entry.m_HorizHist = vhh.getHorizHistograms();
            m_DB.insert(std::pair<int, VertHorizHistoDBEntry>(vhh.getForegroundPointCount(m_Channels - 1), vhh_entry));
        }
    }
}

std::vector<std::pair<int, double>> VertHorizHistoDataBase::predictClass(const std::vector<std::vector<int>>& vertHist, const std::vector<std::vector<int>>& horizHist) {
    if (m_DB.empty())
        return std::vector<std::pair<int, double>>();
    
    std::vector<std::pair<int, double>> results;
    
///calculate foreground point count 
    int sum = std::accumulate(vertHist[m_Channels - 1].begin(), vertHist[m_Channels - 1].end(), 0);
    int tol = sum / 3;
    
    for (int count = sum - tol; count <= sum + tol; ++count) {
        auto range = m_DB.equal_range(count);
        for (auto it = range.first; it != range.second; ++it) {
            VertHorizHistoDBEntry vhh_entry = it->second;
            double scoreVert = 0.0;
            double scoreHoriz = 0.0;
            for (int i = 0; i < m_Channels; ++i) {
                scoreVert += compareHist(vertHist[i], vhh_entry.m_VertHist[i]);
                scoreHoriz += compareHist(horizHist[i], vhh_entry.m_HorizHist[i]);
            }
            std::pair<int, double> score = std::make_pair(vhh_entry.m_ID, scoreVert + scoreHoriz);
            results.push_back(score);
        }
    }
    
    std::sort(results.begin(), results.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) -> bool { return a.second > b.second; });
    return results;
}

double VertHorizHistoDataBase::compareHist(const std::vector<int>& hist1, const std::vector<int>& hist2) {
    if(hist1.size() != hist2.size())
        return 0.0;

    double sum = 0;    
    for (unsigned int i = 0; i < hist1.size(); ++i) {
        if (hist1[i] == 0 && hist2[i] == 0)
            continue;
        sum += double(hist1[i] - hist2[i]) * double(hist1[i] - hist2[i]) / double(hist1[i] + hist2[i]);
    }
    
    return 1.0/(1.0 + sum);
}
