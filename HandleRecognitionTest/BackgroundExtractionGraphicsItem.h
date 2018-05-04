#ifndef _BACKGROUNDEXTRACTION_GRAPHICSITEM_
#define _BACKGROUNDEXTRACTION_GRAPHICSITEM_

#include <QString>
#include <QGraphicsItem>

#include "ImageIconGraphicsItem.h"

class BackgroundExtractionGraphicsItem : public ImageIconGraphicsItem {
public:
    BackgroundExtractionGraphicsItem(int method, int width, const QString& imgPath, QGraphicsItem* parent = 0) : 
        ImageIconGraphicsItem(width, imgPath, parent), m_Method(method) {}
    QImage imageTransform(const QImage& img) override;    
    
private:
    int m_Method = 0;
};

#endif 
