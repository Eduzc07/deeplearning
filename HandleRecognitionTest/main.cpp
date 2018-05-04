#include <QApplication>
#include "MainWidget.h"
#include "CreateBackgroundSubtractionTest.h"
#include "ArgumentList.h"
#include "TestWithTensorFlowModel.h"
#include "ClusterMoments.h"

int main(int argc, char* argv[])
{
    QApplication fileViewer(argc, argv);
    
    ///read the argument list
    ArgumentList al(argc, argv);
    if (al.getSwitch("-createTest")) {
        CreateBackgroundSubtractionTest cbst;
        if (al.size() > 1)
            cbst.execute(al[1].trimmed());
        else
            cbst.execute("*");
        return 0;
    }
    
    if (al.getSwitch("-analyzeMoments")) {
        if (al.size() <= 1) {
            printf("No moments file specified\n");
            exit(1);
        }
        ClusterMoments cm(al[1].trimmed());
        cm.execute();
        return 0;
    }
    
    if (al.getSwitch("-testdnn")) {
        TestWithTensorFlowModel t;
        //t.testAllTrainingData();
        t.testSingleImage();
        return 0;
    }
    
    MainWidget w(nullptr);
    w.show();

    fileViewer.exec();
    return 0;
}
