#include "MainWidget.h"

#include <QHBoxLayout>
#include <QDebug>

///Window with two sections: left a tree view 
///right: icons of images and their transformations
///additionaly there is an options page

MainWidget::MainWidget(QWidget* parent): QWidget(parent)
{
    ///Tree view with folder structure
    m_DirTree = new QTreeView();
    m_DirModel = new QFileSystemModel();
//     m_DirModel->setRootPath("/storage/cucu/HandleRecognition/SegmenteForHandleRecognition");
    m_DirModel->setRootPath("/storage/cucu/MachineLearning");
    m_DirTree->setModel(m_DirModel);
//     m_DirTree->setRootIndex(m_DirModel->index("/storage/cucu/HandleRecognition/SegmenteForHandleRecognition"));
    m_DirTree->setRootIndex(m_DirModel->index("/storage/cucu/MachineLearning"));
    m_DirTree->hideColumn(1);
    m_DirTree->hideColumn(2);
    m_DirTree->hideColumn(3);
    m_DirTree->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Expanding);

    connect(m_DirTree, SIGNAL(clicked(const QModelIndex&)),
            this, SLOT(updateRightTabWidget(const QModelIndex&)));

    //List of histograms page
    m_TabWidget = new QTabWidget();
    m_ImagesCanvas = new ImagesCanvas();
    m_TabWidget->addTab(m_ImagesCanvas->getView(), "Images");    
    
    m_TestImageWidget = new QWidget();
    QVBoxLayout* tvLayout = new QVBoxLayout();
    QHBoxLayout* thLayout = new QHBoxLayout();
    m_TestedImageLabel = new QLabel();
    m_RecognitionResult = new QTextEdit();
    thLayout->addWidget(m_TestedImageLabel);
    thLayout->addWidget(m_RecognitionResult);
    tvLayout->insertLayout(1, thLayout);
    tvLayout->addStretch(4);
    m_TestImageWidget->setLayout(tvLayout);
    m_TabWidget->addTab(m_TestImageWidget, "Recognition");
    
    //Options page
    m_OptionsWidget = new QWidget();
    QVBoxLayout* ovLayout = new QVBoxLayout();

    ///What happens when clicking on a file in the tree view
    QHBoxLayout* ohLayout = new QHBoxLayout();
    QLabel* fileTransfClickText = new QLabel("When clicking on a file");
    m_FileTransfClickCombo = new QComboBox();
    m_FileTransfClickCombo->addItem("KMeans Transformations");
    m_FileTransfClickCombo->addItem("Background Subtraction");
    m_FileTransfClickCombo->addItem("Otsu based Histograms");
    m_FileTransfClickCombo->setCurrentIndex(0);
    connect(m_FileTransfClickCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(setAlgorithmOnFile(int)));
    ohLayout->addWidget(fileTransfClickText);
    ohLayout->addWidget(m_FileTransfClickCombo);
    ohLayout->addStretch(4);
    
    ///What happens when clicking on a folder in the tree view
    QHBoxLayout* ohLayout1 = new QHBoxLayout();
    QLabel* folderTransfClickText = new QLabel("When clicking on a folder");
    m_FolderTransfClickCombo = new QComboBox();
    m_FolderTransfClickCombo->addItem("Background Subtraction");
    m_FolderTransfClickCombo->addItem("Otsu based Histograms");
    m_FolderTransfClickCombo->setCurrentIndex(0);
    connect(m_FolderTransfClickCombo, SIGNAL(currentIndexChanged(int)), this, SLOT(setAlgorithmOnFolder(int)));
    ohLayout1->addWidget(folderTransfClickText);
    ohLayout1->addWidget(m_FolderTransfClickCombo);
    ohLayout1->addStretch(4);
    
    ///DataBase path + calculate database 
    QHBoxLayout* dhLayout1 = new QHBoxLayout();
    QHBoxLayout* dhLayout2 = new QHBoxLayout();
    m_CalculateDB = new QPushButton("Calculate DB");
    m_LeDBMaxId = new QLineEdit();
    m_LeDBPath = new QLineEdit();
    m_DataBasePath = QString("/storage/cucu/MachineLearning/Freilassing1/");
    m_LeDBPath->setText(m_DataBasePath);
    m_LeDBPath->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    m_LeDBPath->setReadOnly(true);
    QLabel* dbMaxIdLabel = new QLabel("Maximum Id in database");
    QLabel* dbPathLabel = new QLabel("DataBase Path");
    connect(m_CalculateDB, SIGNAL(clicked(bool)), this, SLOT(calculateDB()));
    dhLayout1->addWidget(dbPathLabel);
    dhLayout1->addWidget(m_LeDBPath);
    dhLayout2->addWidget(dbMaxIdLabel);
    dhLayout2->addWidget(m_LeDBMaxId);
    dhLayout2->addWidget(m_CalculateDB);
    dhLayout2->addStretch(4);    

    ovLayout->insertLayout(1, ohLayout);
    ovLayout->insertLayout(2, ohLayout1);
    ovLayout->insertLayout(3, dhLayout1);
    ovLayout->insertLayout(4, dhLayout2);
    ovLayout->addStretch(4);
    m_OptionsWidget->setLayout(ovLayout);
    m_TabWidget->addTab(m_OptionsWidget, "Options");        
    
    QHBoxLayout* hLayout = new QHBoxLayout();
    hLayout->addWidget(m_DirTree);
    hLayout->addWidget(m_TabWidget);
    setLayout(hLayout);
}

void MainWidget::updateRightTabWidget(const QModelIndex& current)
{
    switch(m_TabWidget->currentIndex()) {
        case 0:
            updateImagesCanvas(current);
            break;
        case 1:
            updateRecognitionTab(current);
            break; 
    }
}

void MainWidget::updateImagesCanvas(const QModelIndex& current) {
    m_ImagesCanvas->clearImagePaths();
    m_ImagesCanvas->clearPictograms();

    bool isDir = false;
    bool isFile = false;
    
    QString currentPath = m_DirModel->filePath(current);
    qDebug() << "Calculating for " << currentPath;
//     return;
    if (QFileInfo(currentPath).isDir()) {
        isDir = true;
        for (int i = 0; i < m_DirModel->rowCount(current); i++) {
            QModelIndex child = current.child(i, 0);
            QString childPath = m_DirModel->filePath(child);

            if (QFileInfo(childPath).isFile() && childPath.endsWith("ppm"))
                m_ImagesCanvas->pushImagePath(childPath);
        }
    } else if (QFileInfo(currentPath).isFile()) {
        if (currentPath.endsWith("ppm")) {
            isFile = true;
            m_ImagesCanvas->pushImagePath(currentPath);
        }
    }

    if (isFile || isDir)
        m_ImagesCanvas->showPictograms(isDir);    
}

void MainWidget::updateRecognitionTab(const QModelIndex& current) {
    QString currentPath = m_DirModel->filePath(current);
    if (!QFileInfo(currentPath).isFile()) 
        return;
    if (!currentPath.endsWith(".ppm"))
        return;
    
    ///load the current image in the QLabel
    QPixmap pix(currentPath);
    m_TestedImageLabel->setPixmap(pix);
    
    std::string cvImagePath = currentPath.toUtf8().constData();
    cv::Mat cvImg = cv::imread(cvImagePath);
    cv::Mat cvImg1;
    cv::resize(cvImg, cvImg1, cv::Size(200,200));
    VertHorizHistograms vhh(cvImg1);
    std::vector<std::pair<int, double>> resList = m_DataBase.predictClass(vhh.getVertHistograms(), vhh.getHorizHistograms());
    
    QString resString;
    for (unsigned int i = 0; i < resList.size(); ++i) {
        resString += QString("Class ID: %1 Result: ").arg(resList[i].first);
        resString += QString::number(resList[i].second);
        resString += "\n";
    }
    m_RecognitionResult->setText(resString);
    ///perform the recognition with the current database and display the results in the QTextEdit
}

///reads all the images except for those marked with ignore, processes them and then saves the database
void MainWidget::calculateDB()
{
    m_DataBase.train(m_DataBasePath, m_DataBaseMaxId);
}
