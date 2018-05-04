#ifndef IMAGEICONGRAPHICSITEM
#define IMAGEICONGRAPHICSITEM

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <QGraphicsItem>
#include <QFileInfo>
#include <QPainter>
#include <QStyleOptionGraphicsItem>
#include <QGraphicsSceneContextMenuEvent>
#include <QWidget>
#include <QMenu>
#include <QAction>


class ImageIconGraphicsItem : public QGraphicsItem {
public:
    ImageIconGraphicsItem(int width, const QString& imgPath, QGraphicsItem* parent = 0);

    inline QRectF boundingRect() const {
        return QRectF(0, 0, m_Width, m_Width);
    }

    void paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget = 0);
    virtual QImage imageTransform(const QImage& img) { return img; }
    void setImage();
    virtual QString getLabel() { return QFileInfo(m_ImgPath).baseName(); }

    void contextMenuEvent(QGraphicsSceneContextMenuEvent *event);
    
protected:
    QImage m_Image;
    int m_Width = 0;
    int m_ImageWidth = 0;
    QString m_ImgPath;
};

#endif
