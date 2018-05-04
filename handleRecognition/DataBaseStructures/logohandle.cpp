#include "logohandle.h"

void CLogoHandleFeatures::init() {
    m_ImageWidth = 0.0;
    m_ImageHeight = 0.0;
    m_Width = 0.0;
    m_Height = 0.0;
    m_WidthHeight = 0.0;
    m_TopCornerYHeight = 0.0;
    m_WidthRatio = 0.0;
    m_HeightRatio = 0.0;
    m_FillFactor = 0.0;
    m_CenterMassHalfWidth = 0.0;
    m_CenterMassSectionHeight = 0.0;
    m_CenterMassBottomSection = 0.0;
    m_CenterMassLatSectionHeight = 0.0;
    m_CenterMassLatBottomSection = 0.0;
    m_BottomHeight = 0.0;
    m_CenterMass = QPoint(0, 0);
    m_CenterMassLeft = QPoint(0, 0);
    m_CenterMassRight = QPoint(0, 0);
    m_BottomCenterLeft = 0.0;
    m_BottomCenterRight = 0.0;
    m_BottomCenter = 0.0;
    m_BottomCenterHalfLeft = QPoint(0, 0);
    m_BottomCenterHalfRight = QPoint(0, 0);
}

void CLogoHandleFeatures::Write(CFileIO &fio)
{
    fio.Put("WidthHeight", m_WidthHeight);
    fio.Put("FillFactor", m_FillFactor);
    fio.Put("CenterMassHalfWidth", m_CenterMassHalfWidth);
    fio.Put("CenterMassSectionHeight", m_CenterMassSectionHeight);
    fio.Put("CenterMassBottomSection", m_CenterMassBottomSection);
    fio.Put("BottomHeight", m_BottomHeight);
    fio.Put("CenterMassLatSectionHeight", m_CenterMassLatSectionHeight);
    fio.Put("CenterMassLatBottomSection", m_CenterMassLatBottomSection);
    fio.Put("WidthRatio", m_WidthRatio);
    fio.Put("HeightRatio", m_HeightRatio);
    fio.Put("TopCornerYHeight", m_TopCornerYHeight);
}

void CLogoHandleFeatures::Read(CFileIO &fio)
{
    fio.Get("WidthHeight", m_WidthHeight);
    fio.Get("FillFactor", m_FillFactor);
    fio.Get("CenterMassHalfWidth", m_CenterMassHalfWidth);
    fio.Get("CenterMassSectionHeight", m_CenterMassSectionHeight);
    fio.Get("CenterMassBottomSection", m_CenterMassBottomSection);
    fio.Get("BottomHeight", m_BottomHeight);
    fio.Get("CenterMassLatSectionHeight", m_CenterMassLatSectionHeight);
    fio.Get("CenterMassLatBottomSection", m_CenterMassLatBottomSection);

    fio.Get("WidthRatio", m_WidthRatio);
    fio.Get("HeightRatio", m_HeightRatio);
    fio.Get("TopCornerYHeight", m_TopCornerYHeight);
}


void CLogoHandleData::init() {
    m_SegmentedImageWidth = 0.0;
    m_SegmentedImageHeight = 0.0;
    m_HandleType = HandleTypes::Rectangle;
    m_HasHandle = false;
}

