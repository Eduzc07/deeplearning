#ifndef _TESTWITHTENSORFLOWMODEL_
#define _TESTWITHTENSORFLOWMODEL_

#include <QString>
#include <map>
#include <set>
#include <opencv2/dnn.hpp>

class TestWithTensorFlowModel {

public:
    TestWithTensorFlowModel() {
        readTrainedModel();
        readTestData();
    }
    
    /**************************
     * Runs the forward inference on the m_Net with the
     * images from the m_TestDataPath
     * ************************/
    void testAllTrainingData();
    
    /**************************
     * Tests a single image with the network with mini-batch size 1
     * ************************/
    void testSingleImage();    
    
private:
    /***********************************
     * Read the tensorflow model (.pb file) 
     * and initialize m_Net
     * *********************************/
    bool readTrainedModel();
    /***********************************
     * Assigns to each test image(QString) the label to be recognized(int)
     * Reads the labels in a std::set
     ***********************************/
    bool readTestData();
    
private:
    std::string m_TrainedModelPath = "/home/cucu/sources/rvm_griff/_PlayGround/TensorFlow/infer_graph_Model.ckpt.ciresan.64.pb.pb";
    std::string m_SingleImageModelPath = "/home/cucu/sources/rvm_griff/_PlayGround/TensorFlow/infer_graph_Model.ckpt.ciresan.1.pb.pb"; 
    QString m_TestDataPath = "/storage/cucu/MachineLearning/Freilassing1_All_48X48_png/";
    std::string m_SingleImageTestPath = "/storage/cucu/MachineLearning/Freilassing1_All_48X48_png/77/140820_075127_Segment_1.png";
    std::set<int> m_Labels;
    std::map<QString, int> m_TestData;
    
    cv::dnn::Net m_Net;
    cv::Size m_InputImageSize = cv::Size(48, 48);
    int m_BatchSize = 64;
};


#endif


