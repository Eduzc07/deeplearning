#include "TestWithCaffeModel.h"

#include <QFile>
#include <QTextStream>
#include <QFileInfo>
#include <QDebug>
#include <string>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

void getMaxClass(cv::dnn::Blob &probBlob, int *classId, double *classProb)
{
    cv::Mat probMat = probBlob.matRefConst().reshape(1, 1); //reshape the blob to 1x1000 matrix
    cv::Point classNumber;
    cv::minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
    *classId = classNumber.x;
}
// std::vector<cv::String> readClassNames(const char *filename = "./caffe_model/mnist/label.txt")
// {
//     std::vector<cv::String> classNames;
//     std::ifstream fp(filename);
//     if (!fp.is_open())
//     {
//         std::cerr << "File with classes labels not found: " << filename << std::endl;
//         exit(-1);
//     }
//     std::string name;
//     while (!fp.eof())
//     {
//         std::getline(fp, name);
//         if (name.length())
//             classNames.push_back( name.substr(name.find(' ')+1) );
//     }
//     fp.close();
//     return classNames;
// }

bool TestWithCaffeModel::readTrainedModel()
{
//     m_Net = cv::dnn::readNetFromCaffe(m_SolverTxtPath, m_TrainedModelPath);
 
    return true;
}

bool TestWithCaffeModel::readTestData()
{
    QFile inputFile(m_TestDataPath);
    int count = 0;
    if (inputFile.open(QIODevice::ReadOnly))
    {      
        QTextStream in(&inputFile);
        while (!in.atEnd())
        {
            QString line = in.readLine();
            QStringList elems = line.split(" ", QString::SkipEmptyParts);
            if (elems.size() < 2)
                continue;
            if (QFileInfo(elems[0]).exists()) {
                m_TestData[elems[0]] = elems[1].toInt();
                qDebug() << elems[0] << " " << elems[1].toInt();
            }
        }
        inputFile.close();
    }
    qDebug() << count << " test data were read ";
    return true;
}

void TestWithCaffeModel::testData()
{
    for (auto testItem : m_TestData) {
        cv::Mat img = cv::imread(testItem.first.toUtf8().constData());
        if (img.empty())
        {
            std::cerr << "Can't read image from the file: " << testItem.first.toUtf8().constData() << std::endl;
            exit(-1);
        }
        cv::dnn::Blob inputBlob = cv::dnn::Blob(img);                  
        m_Net.setBlob("mnist", inputBlob);        //set the network input
        m_Net.forward();                          //compute output
        cv::dnn::Blob loss = m_Net.getBlob("loss");

        int classId;
        double classProb;
        getMaxClass(loss, &classId, &classProb);//find the best class
        //std::vector<cv::String> classNames = readClassNames();
        std::cout << testItem.first.toUtf8().constData() << "  " << testItem.second;
        std::cout << "Best class: #" << classId << " '" << /*classNames.at(classId) <<*/ "'" << std::endl;
        std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
    }
}
