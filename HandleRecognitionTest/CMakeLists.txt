include_directories(
        ./
        ${CMAKE_SOURCE_DIR}/lib/FileIO
        ${CMAKE_SOURCE_DIR}/lib/Logging
        ${CMAKE_SOURCE_DIR}/lib/Version
        ${CMAKE_SOURCE_DIR}/lib/Defs
        ${CMAKE_SOURCE_DIR}/_PlayGround/handleRecognition
        ${CMAKE_SOURCE_DIR}/_PlayGround/handleRecognition/SegmentationMethods
        ${CMAKE_SOURCE_DIR}/_PlayGround/handleRecognition/DataBaseStructures
        ${QT_INCLUDES}
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
  )

set(SRCS
      main.cpp
      MainWidget.cpp
      ImagesCanvas.cpp
      ImageIconGraphicsItem.cpp
      KMeansIconGraphicsItem.cpp
      BackgroundExtractionGraphicsItem.cpp
      VertHorizHistoGraphicsItem.cpp
      CreateBackgroundSubtractionTest.cpp
      ClusterMoments.cpp
      CVMatQImageConversion.cpp
      TestWithTensorFlowModel.cpp
      ArgumentList.cpp
      ${UIS}
 )

 add_executable(
     FileViewer
     ${SRCS}
     ${QMS}
     ${RES}
 )

 QT5_USE_MODULES(FileViewer Core Widgets)
 target_link_libraries(
        FileViewer
        FileIO
        Logging
        Version
        CrateRecognition
        HandleRecognition
        opencv_imgproc
        opencv_core
        opencv_highgui
        opencv_ml
        opencv_dnn
 )

set_target_properties(FileViewer PROPERTIES AUTOMOC TRUE)


