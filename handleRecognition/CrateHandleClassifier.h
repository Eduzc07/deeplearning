#ifndef CRATEHANDLECLASSIFIER_H
#define CRATEHANDLECLASSIFIER_H

#include <map>
#include <utility>
#include "CrateHandle.h"

class CrateHandleClassifier
{
    ///used when training the classifier only with one object
    std::map<std::pair<int, int>, CLogoHandleData> m_DataMap;

    ///debug flag
    bool m_Debug;

    ///real training data
    std::multimap<std::pair<int, int>, CLogoHandleData> m_TrainData;
    std::vector<std::pair<int, int>> m_TrainedClasses;
    std::vector<std::pair<int, int>> m_TrainedRectangleClasses;
    
//     ClcLogoData* m_LogoDB   = nullptr;

    ///Threshold optimized through tests
    double m_ThresClassif1  = 0.8;
    double m_ThresClassif4  = 0.7;
    double m_ThresClassif61 = 0.7;
    double m_ThresClassif62 = 0.7;

public:
    CrateHandleClassifier();
    ~CrateHandleClassifier();
    void setDebug(bool debug) { m_Debug = debug; }
    void setReferenceData(int crateID, int sampleNo, const CLogoHandleData& lhd);
//     void setLogoDB(ClcLogoData* logoDB) { m_LogoDB = logoDB; }
    ///save direct the handles
    void train(const std::multimap<std::pair<int, int>, CLogoHandleData>& trainData) { m_TrainData = trainData; }
    
    ///CLASSIFICATION METHODS

    ///Various classification functions for the case when the training is made with a single sample per class
    std::vector< std::pair< int, int > > classify(const CLogoHandleData& lhd, const std::vector< std::pair< int, int > >& candidates = std::vector< std::pair< int, int >> ());
    ///classification using the relative dimensions of handle and image as well as the location of the griff in the logo image
    std::vector< std::pair< int, int > > classify1(const CLogoHandleData& lhd, const std::vector< std::pair< int, int > >& candidates = std::vector< std::pair< int, int >> ());
    ///classification using dimensions of the handle
    std::vector< std::pair< int, int > > classify4(const CLogoHandleData& lhd, const std::vector< std::pair< int, int > >& candidates = std::vector< std::pair< int, int >> ());
    ///classification using the form of the handle
    std::vector< std::pair< int, int > > classify6(const CLogoHandleData& lhd, const std::vector< std::pair< int, int > >& candidates = std::vector< std::pair< int, int >> ());

    ///classify with a train database
    ///main method used for classification, after preliminary tests
    std::vector< std::pair< int, int > > classifyTrainedExistSimilar(const CLogoHandleData& lhd, const std::vector< std::pair< int, int > >& candidates = std::vector< std::pair< int, int >> ());
    std::vector< std::pair< int, int > > classifyTrainedStatistical(const CLogoHandleData& lhd, const std::vector< std::pair< int, int > >& candidates = std::vector< std::pair< int, int >> ());
    std::vector< std::pair< int, int > > classifyTrainedNearestNeighbourStatistical(const CLogoHandleData& lhd, const std::vector< std::pair< int, int > >& candidates = std::vector< std::pair< int, int >> ());

    
    ///classification with training data from the logo database
    ///classification using dimensions of the handle
    std::map< int, double > classifRelativeDims(const CLogoHandleData& lhd);
    std::map< int, double > classifForm(const CLogoHandleData& lhd);
    std::map< int, double > classifMoments(const CLogoHandleData& lhd);
    
    double scoreRelativeDimensions1(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2, bool allFeatures = false);
    double scorePositions(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2);
    

private:
    ///classification
    bool classifyBasis1(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2);
    bool classifyBasis4(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2);
    bool classifyBasis6(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2);

    ///form recognition
    std::vector<std::pair<double, double>> calculateContour(const std::vector<QPoint>& contour, const QPoint& centerMass, const QPoint& centerMassLeft, const QPoint& centerMassRight, double& dist);
    void scaleContour(std::vector<std::pair<double, double>>& contour, double factorX, double factorY);
    double calculateIntersectArea(const std::vector< std::pair< double, double > >& contour1, const std::vector< std::pair< double, double > >& contour2, double& area1, double& area2);
    void calculateSections(const std::vector< std::pair< double, double>>& contour1, int& left, int& right, std::map<int, double>& sectionsMin, std::map<int, double>& sectionsMax);
    double scoreIntersect(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2, int count /*for debugging purposes*/);
    void addSectionElement(double x, double y, std::map< int, double >& sectionsMin, std::map<int, double >& sectionsMax);

    ///form recognition with opencv
    double scoreShapeCV(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2);
    
    
    ///training
    void getTrainedClasses(const std::multimap<std::pair<int, int>, CLogoHandleData>& trainData);

    ///utility functions
    void initTargets(const std::vector< std::pair< int, int > >& candidates, std::vector< std::pair< int, int > >& targets);
    void initTargetsTrain(const std::vector< std::pair< int, int > >& candidates, std::vector< std::pair< int, int > >& targets);

    ///score relative dimensions
    
    double scoreRelativeDimensions(const CLogoHandleData& lhd1, const CLogoHandleData& lhd2);
};



#endif // CRATEHANDLECLASSIFIER_H
