#ifndef _TESTWITHCAFFEMODEL_
#define _TESTWITHCAFFEMODEL_

#include <QString>
#include <map>
#include <opencv2/dnn.hpp>

class TestWithCaffeModel {

public:
    TestWithCaffeModel()
    {
        readTrainedModel();
        readTestData();
    }
    void testData();
    
private:
    bool readTrainedModel();
    bool readTestData();
    
private:
    std::string m_SolverTxtPath = "/home/cucu/caffe/cucuLogoRec/cucu_train.prototxt";
    std::string m_TrainedModelPath = "/home/cucu/caffe/cucuLogoRec/Models/_9_BinaryWith473AndFalseSegmentation/lenet_iter_10000.caffemodel";
    QString m_TestDataPath = "/home/cucu/caffe/cucuLogoRec/Models/_9_BinaryWith473AndFalseSegmentation/testset.txt";
    std::vector<int> m_Labels;
    std::map<QString, int> m_TestData;
    
    cv::dnn::Net m_Net;
};


#endif
