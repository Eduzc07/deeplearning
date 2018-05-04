#ifndef _OPENCVKMEANSCLUSTERER_
#define _OPENCVKMEANSCLUSTERER_

#include "OpenCVClusterer.h"

/**
 * Uses of opencv function to cluster RGB Image
 */
class OpenCVKMeansClusterer : public OpenCVClusterer
{
public:
    OpenCVKMeansClusterer(cv::Mat& inputImage, int clusterNo): OpenCVClusterer(inputImage, clusterNo) {}

private:
    /**
    * Given a RGB image it extracts the colors from it and clusters them into the desired number of classes with the kMeans algorithm.
    * Returns false when the input image contains less colors than the number of clusters.
    * @param[in]:
    * @param[out]: labels is the result of the clustering a map giving for each color in the image the associated label
    * @param[in]: clusterNo is the number of classes used to cluster the colors
    */
    bool clusterColors() override;
};

/**
 * Uses KMeansClustering to perform the RGB color clustering and 
 * connected component labelling to perform image segmentation
 */


#endif