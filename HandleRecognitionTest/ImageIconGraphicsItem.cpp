#include "ImageIconGraphicsItem.h"
#include <QFileInfo>
#include <QMenu>
#include <QAction>
#include <QDebug>
#include <QDir>



ImageIconGraphicsItem::ImageIconGraphicsItem(int width, const QString& imgPath, QGraphicsItem* parent) : QGraphicsItem(parent), m_Width(width), m_ImgPath(imgPath) 
{
}

void ImageIconGraphicsItem::setImage()
{
    m_ImageWidth = (3 * m_Width) / 4;
    m_Image = QImage(m_ImgPath);
    m_Image = imageTransform(m_Image);
    m_Image = m_Image.scaled(QSize(m_ImageWidth, m_ImageWidth));
}

void ImageIconGraphicsItem::paint(QPainter* painter, const QStyleOptionGraphicsItem* option, QWidget* widget)
{
    Q_UNUSED(option)
    Q_UNUSED(widget)

    painter->drawImage(QRectF((m_Width - m_ImageWidth)/2, 0, m_ImageWidth, m_ImageWidth), m_Image);
    QString iconText = getLabel();
    QFontMetrics fm = painter->fontMetrics();
    int pixelsWide = fm.width(iconText);
    int pixelsHigh = fm.height();
    painter->drawText(QPoint((m_Width - pixelsWide) / 2, m_ImageWidth + (m_Width - m_ImageWidth - pixelsHigh / 2) / 2), iconText);
//     painter->drawRect(QRectF(0, 0, m_Width, m_Width));
    ///@todo  deal with longer file names
}

void ImageIconGraphicsItem::contextMenuEvent(QGraphicsSceneContextMenuEvent* event)
{
    QMenu menu;
    menu.addAction("Send to database");
    QAction* selectedAction = menu.exec(event->screenPos());
    qDebug() << "User clicked " << selectedAction->text();
    qDebug() << "Image path " << m_ImgPath << " should be sent to database";
    QFileInfo qfi(m_ImgPath);
    if (!qfi.exists())
        return;
    QDir dir = qfi.dir();
    QString fileName = dir.dirName();
    QFile fileToCopy(m_ImgPath);
    QString newPath = "/home/cucu/HandleRecognition/SegmenteForHandleRecognition/DB/";
    QString newFileName = newPath + fileName + ".ppm";
    QFile file1(newFileName);
    if (file1.exists())
        file1.remove();
    fileToCopy.copy(newFileName);
    qDebug() << "Copied file to " << newFileName;
}
