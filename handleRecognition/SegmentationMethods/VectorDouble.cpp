#include "VectorDouble.h"

#include <QString>
#include <cmath>

VectorDouble::VectorDouble(std::vector<double> val)
{
    m_Val = val;
    m_Dim = val.size();
}

VectorDouble::VectorDouble(double comp, unsigned int dim)
{
    m_Dim = dim;
    for (unsigned int i = 0; i < dim; i++)
        m_Val.push_back(comp);
}

VectorDouble::VectorDouble(const cv::Vec3b& rgbVal, std::pair<unsigned int, unsigned int> idx)
{
    m_Val = transformFromVec3bAndIndex(rgbVal, idx);
    m_Dim = 3;
}

///condition: the two vectors should have the same size
VectorDouble VectorDouble::operator+(const VectorDouble& other)
{
    if (m_Dim != other.m_Dim)
        return *this;
    for (unsigned int i = 0; i < m_Dim; i++)
        m_Val[i] += other.m_Val[i];
    return *this;
}

VectorDouble VectorDouble::operator/(double val)
{
    for (unsigned int i = 0; i < m_Dim; i++)
        m_Val[i] /= val;
    return *this;
}

VectorDouble VectorDouble::operator*(double val)
{
    for (unsigned int i = 0; i < m_Dim; i++)
        m_Val[i] *= val;        
    return *this;
}

bool VectorDouble::operator==(const VectorDouble& other)
{
    if (other.m_Dim != m_Dim)
        return false;
    
    for (unsigned int i = 0; i < m_Dim; i++) {
        if (int(m_Val[i]) != int(other.m_Val[i]))
            return false;
    }
    return true;
}


/**
 * Euclidian distance divided to the square root of the number of dimensions
 */

double VectorDouble::distTo(const VectorDouble& other)
{
    if (!m_Dim)
        return -1.0;

    if (m_Dim != other.m_Dim)
        return -1.0;

    double sum = 0.0;
    for (unsigned int i = 0; i < m_Dim; i++)
        sum += (m_Val[i] - other.m_Val[i]) * (m_Val[i] - other.m_Val[i]);

    sum = sqrt(sum) / sqrt(double(m_Dim));
    return sum;
}

std::vector<double> VectorDouble::transformFromVec3bAndIndex(const cv::Vec3b& rgbVal, std::pair< unsigned int, unsigned int > idx)
{
    std::vector<double> retVal;
    retVal.push_back(double(rgbVal[0]));
    retVal.push_back(double(rgbVal[1]));
    retVal.push_back(double(rgbVal[2]));
//     retVal.push_back(double(idx.first / 5));
//     retVal.push_back(double(idx.second / 5));
    return retVal;
}

QString VectorDouble::toString() 
{
    QString retVal = "(";
    for (auto val : m_Val)
        retVal += QString::number(val) + " ";
    retVal += ")";
    return retVal;
}

bool operator==(const VectorDouble& v1, const VectorDouble& v2) {
    if (v1.m_Dim != v2.m_Dim)
        return false;
    
    for (unsigned int i = 0; i < v1.m_Dim; i++) {
        if (int(v1.m_Val[i]) != int(v2.m_Val[i]))
            return false;
    }
    return true;    
}

/**
ProductVectorDouble::ProductVectorDouble(const cv::Vec3b& rgbVal, std::pair<unsigned int, unsigned int> idx)
{
    m_Val = transformFromVec3bAndIndex(rgbVal, idx);
    m_Dim = 5;
}

std::vector<double> ProductVectorDouble::transformFromVec3bAndIndex(const cv::Vec3b& rgbVal, std::pair< unsigned int, unsigned int > idx)
{
    std::vector<double> retVal = VectorDouble::transformFromVec3bAndIndex(rgbVal, idx);
    retVal.push_back(double(idx.first));
    retVal.push_back(double(idx.second));
    return retVal;
}

ProductVectorDouble& ProductVectorDouble::operator=(const ProductVectorDouble& other)
{
    m_Val = other.m_Val;
    m_Dim = other.m_Dim;
    
    return *this;
}
*/

    
