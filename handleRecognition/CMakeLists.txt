include_directories(
    ./
    ${CMAKE_SOURCE_DIR}/
    SegmentationMethods/
    DataBaseStructures/
    ${CMAKE_SOURCE_DIR}/lib/FileIO/
    ${QT_INCLUDES}
    ${OpenCV_INCLUDE_DIRS}
)

set(HandleRecognition_SRCS
    CrateHandle.cpp
    CrateHandleClassifier.cpp
    CrateHandleExtractor.cpp
    FloodFillCrateHandleSegmenter.cpp
    KMeansCrateHandleSegmenter.cpp
    BackgroundCrateHandleSegmenter.cpp
    HandleRecognitionSegment.cpp
    SegmentationMethods/ImageSegment.cpp
    SegmentationMethods/VectorDouble.cpp
    SegmentationMethods/RGBImageClusterer.cpp
    SegmentationMethods/RGBImageClusteringSegmenter.cpp
    SegmentationMethods/ConnectedComponentLabelling.cpp
    SegmentationMethods/OpenCVKMeansClusterer.cpp
    SegmentationMethods/OpenCVClusterer.cpp
    SegmentationMethods/OpenCVEMClusterer.cpp
    SegmentationMethods/MostRepresentedColors.cpp
    DataBaseStructures/logohandle.cpp
    DataBaseStructures/VertHorizHistoDB.cpp)

add_library(
    HandleRecognition STATIC
    ${HandleRecognition_SRCS}
)

if (USE_Qt5)
    QT5_USE_MODULES(HandleRecognition Core Widgets)
endif (USE_Qt5)

target_link_libraries(
    HandleRecognition
    opencv_imgproc
    ${QT_QTCORE_LIBRARY}
)

set_target_properties(HandleRecognition PROPERTIES AUTOMOC TRUE)
