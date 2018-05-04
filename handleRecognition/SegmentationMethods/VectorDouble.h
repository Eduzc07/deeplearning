#ifndef _MULTIDIMDOUBLEVALUE_
#define _MULTIDIMDOUBLEVALUE_

#include <vector>
#include <opencv2/core/core.hpp>
#include <QString>


class VectorDouble {
protected:
    std::vector<double> m_Val;
    unsigned int m_Dim;
    
public:
    VectorDouble() {}
    /**
     * Initializing from a double vector
     */
    VectorDouble(std::vector<double> val);
    /**
     * Initialize with a value whose every component is comp
     */
    VectorDouble(double comp, unsigned int dim);
    /**
     * Initialize from rgb value and index
     */
    VectorDouble(const cv::Vec3b& rgbVal, std::pair<unsigned int, unsigned int> idx);
    /**
     * Distance to another multidimensional value
     */
    
    virtual double distTo(const VectorDouble& other);
    VectorDouble operator+(const VectorDouble& other);
    VectorDouble operator/(double val);
    VectorDouble operator*(double val);
    
    ///for hashing
    bool operator==(const VectorDouble& other);
    friend bool operator==(const VectorDouble& v1, const VectorDouble& v2);
    inline unsigned int getDim() { return m_Dim; }
    QString toString();
    
    std::vector<double> transformFromVec3bAndIndex(const cv::Vec3b& rgbVal, std::pair<unsigned int, unsigned int> idx);
    inline double getComp(unsigned int comp) const {  return m_Val[comp];  }
};

namespace std {

  template <>
  struct hash<VectorDouble>
  {
    std::size_t operator()(const VectorDouble& k) const
    {
      return (int(k.getComp(0) * 255 * 255) + int(k.getComp(1) * 255) + int(k.getComp(2)));
    }
  };

}

/**
class ProductVectorDouble : public VectorDouble {
public:
    ProductVectorDouble() {}
    ProductVectorDouble(std::vector<double> val) : VectorDouble(val) {}
    ProductVectorDouble(double comp, unsigned int dim) : VectorDouble(comp, dim) {}
    ProductVectorDouble(const cv::Vec3b& rgbVal, std::pair<unsigned int, unsigned int> idx);
    std::vector<double> transformFromVec3bAndIndex(const cv::Vec3b& rgbVal, std::pair<unsigned int, unsigned int> idx);
    ProductVectorDouble& operator=(const ProductVectorDouble& other);
};
*/
#endif
