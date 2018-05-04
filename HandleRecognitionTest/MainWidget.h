#ifndef MAINWIDGET
#define MAINWIDGET


#include <QWidget>
#include <QTreeView>
#include <QFileSystemModel>
#include <QTabWidget>
#include <QComboBox>
#include <QLineEdit>
#include <QPushButton>
#include <QTextEdit>
#include <QLabel>
#include <map>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "ImagesCanvas.h"
#include "VertHorizHistoDB.h"

/**********************************************
 * Window with two sections: left a tree view 
 * right: icons of images and their transformations
 * additionaly there is an options page
 **********************************************/

class MainWidget : public QWidget {
    Q_OBJECT

public:
    MainWidget(QWidget* parent);

    QSize sizeHint() const override {
        return QSize(1000, 500);
    }

private slots:
    /***********************************************
     * Updates the window on the right after a click in the tree view
     * ***************************************************/
    void updateRightTabWidget(const QModelIndex& current);
    /*********************************************************
     * Reacts to click on a node in the tree view
     * Sends list of files in the clicked folder or
     * file name (when a file is clicked) to the ImagesCanvas
     * and sends update command to the canvas
     * *******************************************************/
    void updateImagesCanvas(const QModelIndex& current);
    /*********************************************************
     * Reacts to click on a node in the tree view
     * Saves the files name in the MainWidget and updates the Recognition tab
     * *******************************************************/
    void updateRecognitionTab(const QModelIndex& current);

    /***********************************************
     * What to do when clicked on a file
     * ***************************************************/    
    inline void setAlgorithmOnFile(int idx) { m_ImagesCanvas->setAlgorithmOnFile(idx); }
    /***********************************************
     * What to do when clicked on a folder
     * ***************************************************/
    inline void setAlgorithmOnFolder(int idx) { m_ImagesCanvas->setAlgorithmOnFolder(idx); }
    /***********************************************
     * Calculates images recognition database from training path
     * ***************************************************/
    void calculateDB();

private:
    QTreeView* m_DirTree;
    QFileSystemModel* m_DirModel;
    
    ///images display window in the tab widget
    ImagesCanvas* m_ImagesCanvas;

    ///tab widget on the right side of the folder tree view
    QTabWidget* m_TabWidget;

    ///Options window in the tab widget
    QWidget* m_OptionsWidget;
    QComboBox* m_FileTransfClickCombo;
    QComboBox* m_FolderTransfClickCombo;
    QPushButton* m_CalculateDB;
    QLineEdit* m_LeDBPath;
    QLineEdit* m_LeDBMaxId;
    
    ///Test image with database
    QWidget* m_TestImageWidget;
    QLabel* m_TestedImageLabel;
    QTextEdit* m_RecognitionResult;
    
    QString m_DataBasePath;
    int m_DataBaseMaxId = 530;
    VertHorizHistoDataBase m_DataBase;
};

#endif
