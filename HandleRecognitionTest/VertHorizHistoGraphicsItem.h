#ifndef __VERTHORIZHISTO_
#define __VERTHORIZHISTO_

#include <QString>
#include <QGraphicsItem>

#include "ImageIconGraphicsItem.h"

class VertHorizHistoGraphicsItem : public ImageIconGraphicsItem {
public:
    VertHorizHistoGraphicsItem(int type, int width, const QString& imgPath, QGraphicsItem* parent = 0) : 
        ImageIconGraphicsItem(width, imgPath, parent), m_Type(type) {}
    QImage imageTransform(const QImage& img) override;    
    
private:
    int m_Type = 0;
};















#endif
