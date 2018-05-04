#ifndef IMAGESCANVAS
#define IMAGESCANVAS

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>

class ImagesCanvas {
public:
    ImagesCanvas();

    inline QWidget* getView() { return m_View; }
    inline void clearImagePaths() { m_ImagesPaths.clear(); }
    inline void pushImagePath(const QString& path) { m_ImagesPaths.push_back(path); }
    inline void clearPictograms() {
        m_Scene->clear();
    }
    inline void setAlgorithmOnFile(int val) {
        m_AlgorithmOnFile = val;
    }

    inline void setAlgorithmOnFolder(int val) {
        m_AlgorithmOnFolder = val;
    }    
    
    virtual void showPictograms(bool isDir);

private:
    void showPictogramsFolderBackSubtr();     
    void showPictogramsFolderHisto();
    void showPictogramsFileKMeans();
    void showPictogramsFileBackSubtr();
    void showPictogramsFileHisto();    

private:
    QGraphicsScene* m_Scene;
    QGraphicsView* m_View;

    std::vector<QString> m_ImagesPaths;
    int m_AlgorithmOnFile = 0;
    int m_AlgorithmOnFolder = 0;
    
    bool m_ShowKMeans = true;
};

#endif







