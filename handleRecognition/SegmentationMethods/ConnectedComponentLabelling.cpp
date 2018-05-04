#include "ConnectedComponentLabelling.h"

#include <QTime>
#include <QDebug>

void ConnectedComponentLabelling::execute(std::map <cv::Vec3b, ImageSegment, LessVec3b>& segmentsMap, cv::Mat& output)
{
    QTime* t2 = new QTime();
    t2->start();

    //build temporary matrix used to segment
    cv::Mat tempLabels = cv::Mat::zeros(m_ClusteredImage.size(), CV_32S);
    //initialize label counter
    unsigned int nextLabel = 1;
    //used to manage the equivalence classes of labels
    //see http://www.cse.msu.edu/~stockman/Book/2002/Chapters/ch3.pdf page 73
    std::vector<unsigned int> labelParent;
    labelParent.push_back(0);
    //saves the correspondence between the kmeanslabels and the labels obtained in the segmentation
    std::vector<cv::Vec3b> originalLabels;
    originalLabels.push_back(cv::Vec3b(0, 0, 0));


    //8-connectivity neighbourhood for segmentation
    std::vector<cv::Point> mask;
    mask.push_back(cv::Point(-1, -1));
    mask.push_back(cv::Point(-1, 0));
    mask.push_back(cv::Point(0, -1));
    mask.push_back(cv::Point(-1, 1));


    unsigned int minNeighbLabel = 100000000;
    std::set<unsigned int> neighbLabels;
    cv::Vec3b currentLabelCluster = cv::Vec3b(0, 0, 0);
    bool repeat = false;
    int counter = 0;

    int mini = 0, minj = 0, maxi = 0, maxj = 0;

    if (!m_BoundingRect.width() || !m_BoundingRect.height()) {
        maxi = m_InitialImage.rows;   //m_ClusteredImage might bigger than m_InitialImage because it is a mask obtained from a cv::floodFill call
        maxj = m_InitialImage.cols;
        mini = 0;
        minj = 0;
    } else {
        maxi = m_BoundingRect.bottom();
        maxj = m_BoundingRect.right();
        mini = m_BoundingRect.top();
        minj = m_BoundingRect.left();
    }

    int maxNeighbLabelSize = 0;

    //first pass (of two) of the segmentation
    for (int i = mini; i < maxi; i++)
        for (int j = minj; j < maxj; j++) {
            //for each pixel
            cv::Vec3b clusterLabel = m_ClusteredImage.at<cv::Vec3b>(i, j);

            if (j != minj && clusterLabel == currentLabelCluster)
                repeat = true;
            else
                repeat = false;

            currentLabelCluster = clusterLabel;
            //get the list of pixels around him with a label and with the same value
            getNeighbours(tempLabels, i, j, currentLabelCluster, mask, neighbLabels, minNeighbLabel, repeat, counter);

            if ((int)neighbLabels.size() > maxNeighbLabelSize)
                maxNeighbLabelSize = neighbLabels.size();

            if (neighbLabels.empty()) {
                //if no such neighbours
                //initialize equivalence class of labels
                labelParent.push_back(0);
                originalLabels.push_back(clusterLabel);
                //give the element the current label
                tempLabels.at<int>(i, j) = nextLabel;
                nextLabel++;
            } else {
                //assign the element the minimum label of its neighbours
                tempLabels.at<int>(i, j) = minNeighbLabel;
                //join the equivalency class containing the chosen label and the labels of the neighbours
                setUnion(minNeighbLabel, neighbLabels, labelParent);
            }
        }

//     qDebug() << "Max min neighb label size" << maxNeighbLabelSize;

    //intermediary data analysis
    //conversion between the labels obtained in the first pass of the segmentation and colors
    std::vector<cv::Vec3b> labelConversion;
    labelConversion.push_back(cv::Vec3b(0, 0, 0));
    //the indices of the labels obtained at the end of the segmentation (after the union of the connected components)
    std::vector<int> finalLabels;
    //map between the colors chosen and the indices of the segmented images
    std::map<cv::Vec3b, int, LessVec3b> finalColorsOnScreen;

    //for every label given in the first step of the segmentation
    for (unsigned int k = 1; k < labelParent.size(); k++) {
        //if the label has no parents (was corresponds to correspond to a segment)
        if (!labelParent[k]) {
            //choose a color for it
            cv::Vec3b temp(rand() % 220 + 35, rand() % 220 + 35 , rand() % 220 + 35);
            //save the color in the labelConversion vector
            labelConversion.push_back(temp);
            //save the good label in a vector
            finalLabels.push_back(k);
            //save the correspondence between the color and the label
            finalColorsOnScreen[temp] = k;
            //build a segment for this label
            ImageSegment newSegment;
            newSegment.m_IndexLabelSegmentation = k;
            newSegment.m_ColorInitialImage = originalLabels[k];
            newSegment.m_ColorSegImage = temp;
            newSegment.m_AvgColor = cv::Vec3f(0.0, 0.0, 0.0);
            //save the association between the chosen color and the segment
            segmentsMap[temp] = newSegment;
        } else {
            //choose not a color for this label (it will be discarded at the end)
            labelConversion.push_back(cv::Vec3b(0, 0, 0));
        }
    }

    // for (int i=0; i<finalLabels.size(); i++)
    // qDebug() << finalLabels[i];

    //choose colors for labels that will be discarded in the second part of the segmentation algorithm
    for (unsigned int i = 1; i < labelParent.size(); i++) {
        if (labelConversion[i] == cv::Vec3b(0, 0, 0))
            labelConversion[i] = labelConversion[setFind(i, labelParent)];
    }

    //final pass connected component labelling
    output = cv::Mat(tempLabels.size(), CV_8UC3);

    for (int i = mini; i < maxi; i++)
        for (int j = minj; j < maxj; j++) {
            //set the colors for each pixel according to the label conversion calculated earlier
            cv::Vec3b temp = labelConversion[tempLabels.at<int>(i, j)];
            output.at<cv::Vec3b>(i, j) = temp;

            //get the segment for this color
            if (segmentsMap.find(temp) == segmentsMap.end())
                printf("Color %d %d %d not found in segment map\n", temp[0], temp[1], temp[2]);

            ImageSegment& seg = segmentsMap[temp];

            //update position of the segment
            if (j < seg.m_Left)
                seg.m_Left = j;

            if (j > seg.m_Right)
                seg.m_Right = j;

            if (i < seg.m_Top)
                seg.m_Top = i;

            if (i > seg.m_Bottom)
                seg.m_Bottom = i;

            //count the number of points in the segment
            seg.m_NoPoints++;

            if (m_ComputeAvgColor) {
                cv::Vec3b pixelColor = m_InitialImage.at<cv::Vec3b>(i, j);
                double color0 = double(pixelColor[0]) / 255.0;
                double color1 = double(pixelColor[1]) / 255.0;
                double color2 = double(pixelColor[2]) / 255.0;
                seg.m_AvgColor += cv::Vec3f(color0, color1, color2);
            }
        }

    int duration = t2->elapsed();
    qDebug() << "Total time segmentation " << duration;
}

void ConnectedComponentLabelling::getNeighbours(const cv::Mat& labels, int row, int col, const cv::Vec3b& currentLabelCluster, const std::vector< cv::Point >& mask, std::set< unsigned int >& neighbLabels, unsigned int& minNeighbLabel, bool repeat, int& counter)
{
    int startLoop = mask.size() - 1;

    if (!repeat || counter >= 10) {
        minNeighbLabel = 100000000;
        neighbLabels.clear();
        startLoop = 0;
        counter = 0;
    } else {
        counter++;
    }


    for (int i = startLoop; i < (int)mask.size(); i++) {
        cv::Point currCoord = mask[i] + cv::Point(row, col);

        if (mask[i].x != 0 || mask[i].y != 0) {
            if (!row || !col) {
                if (currCoord.x >= m_BoundingRect.bottom() || currCoord.x < 0 || currCoord.y >= m_BoundingRect.right() || currCoord.y < 0)
                    continue;
            }

            unsigned int curLabel = labels.at<int>(currCoord.x, currCoord.y);

            if (curLabel && m_ClusteredImage.at<cv::Vec3b>(currCoord.x, currCoord.y) == currentLabelCluster) {
                neighbLabels.insert(curLabel);

                if (curLabel < minNeighbLabel)
                    minNeighbLabel = curLabel;
            }
        }
    }
}


void ConnectedComponentLabelling::setUnion(unsigned int label, const std::set< unsigned int >& label_set, std::vector< unsigned int >& parentLabels)
{
    std::set<unsigned int>::iterator it = label_set.begin();

    for (; it != label_set.end() ; ++it)
        setUnion(label, *it, parentLabels);
}

void ConnectedComponentLabelling::setUnion(unsigned int label1, unsigned int label2, std::vector< unsigned int >& parentLabels)
{
    unsigned int i = label1;
    unsigned int j = label2;

    while (parentLabels[i] != 0)
        i = parentLabels[i];

    while (parentLabels[j] != 0)
        j = parentLabels[j];

    if (i != j)
        parentLabels[j] = i;
}

unsigned int ConnectedComponentLabelling::setFind(unsigned int label, const std::vector< unsigned int >& parentLabels)
{
    unsigned int i = label;

    while (parentLabels[i] != 0)
        i = parentLabels[i];

    return i;
}

void ConnectedComponentLabelling::printPointInfo(const cv::Mat& labels, int row, int col, const std::vector< cv::Point >& mask)
{
    for (int k = 0; k < (int)mask.size(); k++) {
        if (row + mask[k].x < 0 || row + mask[k].x >= m_ClusteredImage.rows || col + mask[k].y < 0 || col + mask[k].y >= m_ClusteredImage.cols)
            printf(" -1 ");
        else
            printf(" %d ", m_ClusteredImage.at<cv::Vec3b>(row + mask[k].x, col + mask[k].y)[0]);

        if ((k + 1) % 3 == 0)
            printf("\n");
    }

    printf("-----------\n");

    for (int k = 0; k < (int)mask.size(); k++) {
        if (row + mask[k].x < 0 || row + mask[k].x >= m_ClusteredImage.rows || col + mask[k].y < 0 || col + mask[k].y >= m_ClusteredImage.cols)
            printf(" -1 ");
        else
            printf(" %d ", labels.at<int>(row + mask[k].x, col + mask[k].y));

        if ((k + 1) % 3 == 0)
            printf("\n");
    }

    printf("-----------\n");
}
