#ifndef LOGOHANDLE_H
#define LOGOHANDLE_H

#include <QPoint>

#include "FileIO.h"

struct CLogoHandleFeatures {
    double m_ImageWidth;
    double m_ImageHeight;
    double m_Width;
    double m_Height;
    double m_WidthHeight;
    double m_WidthRatio;
    double m_HeightRatio;
    double m_FillFactor;
    double m_TopCornerYHeight;
    double m_CenterMassHalfWidth;
    double m_CenterMassSectionHeight;
    double m_CenterMassBottomSection;
    double m_CenterMassLatBottomSection;
    double m_CenterMassLatSectionHeight;
    double m_BottomHeight;
    std::vector<QPoint> m_Contour;
    QPoint m_CenterMass;
    QPoint m_CenterMassLeft;
    QPoint m_CenterMassRight;
    double m_BottomCenterLeft;
    double m_BottomCenterRight;
    double m_BottomCenter;
    QPoint m_BottomCenterHalfLeft;
    QPoint m_BottomCenterHalfRight;

public:
    CLogoHandleFeatures() {init();}
    ~CLogoHandleFeatures() {}
    void Write(CFileIO &fio);
    void Read(CFileIO &fio);

private:
    void init();
};

struct CLogoHandleData {
    enum HandleTypes { Rectangle = 0, Banana = 1, Lemon = 2 };
    std::vector<CLogoHandleFeatures> m_Features;

    QString m_CompleteLogoFile;
    QString m_SegmentedLogoFile;
    std::vector<QPoint> m_EditedContour;
    double m_SegmentedImageWidth;
    double m_SegmentedImageHeight;
    HandleTypes m_HandleType;
    bool m_HasHandle;

public:
    CLogoHandleData() {init();}
    ~CLogoHandleData() {}

private:
    void init();
};


#endif
