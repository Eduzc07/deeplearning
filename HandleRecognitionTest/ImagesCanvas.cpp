#include "ImagesCanvas.h"
#include <QFileInfo>
#include <QProgressDialog>

#include "KMeansCrateHandleSegmenter.h"
#include "ImageIconGraphicsItem.h"
#include "KMeansIconGraphicsItem.h"
#include "BackgroundExtractionGraphicsItem.h"
#include "VertHorizHistoGraphicsItem.h"

ImagesCanvas::ImagesCanvas()
{
    m_Scene = new QGraphicsScene();
//     m_Scene->setSceneRect(QRectF(0, 0, 500, 500));
    m_View = new QGraphicsView(m_Scene);
}

void ImagesCanvas::showPictograms(bool isDir)
{
    if (isDir) {
        switch (m_AlgorithmOnFolder) {
            case 0: 
                showPictogramsFolderBackSubtr();
                break;
            case 1:
                showPictogramsFolderHisto();
                break;
        }
    } else {
        switch (m_AlgorithmOnFile) {
            case 0: 
                showPictogramsFileKMeans();
                break; 
            case 1:
                showPictogramsFileBackSubtr();
                break;
            case 2:
                showPictogramsFileHisto();
                break;
        }
    }
        
    m_View->viewport()->update();
}

void ImagesCanvas::showPictogramsFolderBackSubtr() 
{
    int count = 0;

    QProgressDialog prgDlg("Berechnen Bildtransformationen ..", "Abbrechnen", 0, m_ImagesPaths.size(), 0);
    prgDlg.setWindowModality(Qt::WindowModal);
    prgDlg.setMinimumDuration(200);

    for (auto path : m_ImagesPaths) {
        BackgroundExtractionGraphicsItem* pictoT = new BackgroundExtractionGraphicsItem(0, 200, path);
        pictoT->setImage();
        ImageIconGraphicsItem* pictoO = new ImageIconGraphicsItem(200, path);
        pictoO->setImage();
        m_Scene->addItem(pictoT);
        m_Scene->addItem(pictoO);
        pictoT->setPos((count % 2) * 2 * 250, count / 2 * 250);
        pictoO->setPos((2 * (count % 2) + 1) * 250, count / 2 * 250);
        prgDlg.setValue(count);
        if (prgDlg.wasCanceled())
            break;
        count++;
//         if (count > 0)
//             break;
    }
}

void ImagesCanvas::showPictogramsFolderHisto() 
{
    int count = 0;

    QProgressDialog prgDlg("Berechnen Bildtransformationen ..", "Abbrechnen", 0, m_ImagesPaths.size(), 0);
    prgDlg.setWindowModality(Qt::WindowModal);
    prgDlg.setMinimumDuration(200);

    for (auto path : m_ImagesPaths) {
        ImageIconGraphicsItem* pictoO = new ImageIconGraphicsItem(200, path);
        pictoO->setImage();
        m_Scene->addItem(pictoO);
        pictoO->setPos(0, count * 250);
        for (int i = 0; i < 3; i++) {
            VertHorizHistoGraphicsItem* pictoH = new VertHorizHistoGraphicsItem(i, 200, path);
            pictoH->setImage();
            m_Scene->addItem(pictoH);
            pictoH->setPos(250 * (i + 1), count * 250);
        }
        prgDlg.setValue(count);
        if (prgDlg.wasCanceled())
            break;
        count++;
//         if (count > 0)
//             break;
    }
}

void ImagesCanvas::showPictogramsFileKMeans()
{
    ImageIconGraphicsItem* pictoO = new ImageIconGraphicsItem(200, m_ImagesPaths[0]);
    pictoO->setImage();
    m_Scene->addItem(pictoO);
    pictoO->setPos(0, 0);
    for (int j = 0; j < 4; j++) 
        for (int i = 2; i < 6; i++) {
            KMeansIconGraphicsItem* pictoT = new KMeansIconGraphicsItem(j, i, 200, m_ImagesPaths[0]);
            pictoT->setImage();
            m_Scene->addItem(pictoT);
            pictoT->setPos((i - 2) * 250, (j + 1) * 250);
        }
}

void ImagesCanvas::showPictogramsFileBackSubtr()
{
    ImageIconGraphicsItem* pictoO = new ImageIconGraphicsItem(200, m_ImagesPaths[0]);
    pictoO->setImage();
    m_Scene->addItem(pictoO);
    pictoO->setPos(0, 0);
    
    BackgroundExtractionGraphicsItem* pictoT = new BackgroundExtractionGraphicsItem(0, 400, m_ImagesPaths[0]);
    pictoT->setImage();    
    m_Scene->addItem(pictoT);
    pictoT->setPos(250, 0);
}

void ImagesCanvas::showPictogramsFileHisto() {

    ImageIconGraphicsItem* pictoO = new ImageIconGraphicsItem(200, m_ImagesPaths[0]);
    pictoO->setImage();
    m_Scene->addItem(pictoO);
    pictoO->setPos(0, 0);
    
    for (int i = 0; i < 3; i++) {
        printf("VertHorizHistoGraphicsItem %d\n", i);
        VertHorizHistoGraphicsItem* pictoH = new VertHorizHistoGraphicsItem(i, 200, m_ImagesPaths[0]);
        pictoH->setImage();
        m_Scene->addItem(pictoH);
        pictoH->setPos(250 * (i + 1), 0);
    }
}
