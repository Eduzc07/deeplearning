#ifndef KMEANSICONGRAPHICSITEM
#define KMEANSICONGRAPHICSITEM

#include "ImageIconGraphicsItem.h"

class KMeansIconGraphicsItem : public ImageIconGraphicsItem {
public:
    KMeansIconGraphicsItem(int segMethod, int noClusters, int width, const QString& imgPath, QGraphicsItem* parent = 0) : 
        ImageIconGraphicsItem(width, imgPath, parent), m_SegMethod(segMethod), m_NoClusters(noClusters) {}
    QImage imageTransform(const QImage& img) override;
    
    QString getLabel() { return m_MethodNames[m_SegMethod] + QString::number(m_NoClusters);  }
    
private:
    int m_SegMethod = 0;
    int m_NoClusters = 0;
    
    const QStringList m_MethodNames = { "KMeansOpenCV", "EMOpenCV", "MostRepresent", "KMeans" };
};

#endif