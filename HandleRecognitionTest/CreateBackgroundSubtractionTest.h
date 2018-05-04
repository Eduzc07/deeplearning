#ifndef _CREATEBKGRNDSUBTRACTION_TEST
#define _CREATEBKGRNDSUBTRACTION_TEST

#include <vector>
#include <QString>
#include <QTextStream>
#include <QFile>

/***
 * Performs background subtraction and saves the results in a mirror of 
 * /home/cucu/HandleRecognition/SegmenteForHandleRecognition
 */

class CreateBackgroundSubtractionTest {
public:    
    CreateBackgroundSubtractionTest();
    void execute(const QString& classFilter);
    
private:
    
    /***
     * Processes folders with the following structure
     * TopFolder(with name "folder")
     *     - CrateID1
     *         - img11.ppm
     *         - img21.ppm
     *         ........... 
     *     - CrateID2
     *         - img12.ppm
     *         - img22.ppm
     *         ............ 
     *     .....
     */
    
    void processTestFolder(const QString& folder);


    /***
     * Processes folders with the following structure
     *     - CrateID
     *         - img1.ppm
     *         - img2.ppm
     *         ........... 
     */
    
    void processImagesFolder(const QString& testFolder, const QString& folder);


    
private:
    std::vector<QString> m_DataFolders;
//     QString m_RootPath = "/home/cucu/HandleRecognition/SegmenteForHandleRecognition/";
    QString m_RootPath = "/storage/cucu/MachineLearning/";
    QString m_TestFolder = "PreprocessedSegments/";
    QRegExp m_ClassFilter;
    
    QString m_MomentsFileName = "moments";
    QFile* m_MomentsFile = nullptr;
    QTextStream* m_MomentsStream = nullptr;
};
























#endif
