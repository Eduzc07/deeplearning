import os


dataPath_Freilassing_orig="/storage/cucu/MachineLearning/Freilassing1/"
dataPath_apolda_links_orig="/storage/cucu/MachineLearning/apolda-links/"
dataPath_herrieden_rechts_orig="/storage/cucu/MachineLearning/herrieden-rechts/"
dataPath_Freilassing_32="/storage/cucu/MachineLearning/Freilassing1_classWithMin100_32X32_png/"
dataPath_Freilassing_48="/storage/cucu/MachineLearning/Freilassing1_classWithMin100_48X48_png/"
dataPath_All_orig="/shared/developer/Crate-Logos_Sorted/Ready/"
dataPath_All_48="/storage/cucu/MachineLearning/All_48X48_jpeg/"
dataPath_Freilassing1_All="/storage/cucu/MachineLearning/Freilassing1_All_48X48_png/"
dataPath_All_227="/storage/cucu/MachineLearning/All_227X227_png/"
dataPath_All_224="/storage/cucu/MachineLearning/All_224X224_png/"
dataPath_All_96="/storage/cucu/MachineLearning/All_96X96_png/"

def get_immediate_subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def get_immediate_files(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name))]

def get_immediate_files_with_label(a_dir):
    return [(os.path.join(a_dir, name),os.path.basename(a_dir)) for name in os.listdir(a_dir)
            if os.path.isfile(os.path.join(a_dir, name))]

def getClassesIDs(dataPath):
    folderPaths = get_immediate_subdirectories(dataPath)
    classesIDs = []
    for fp in folderPaths:
        className = os.path.basename(fp)
        classesIDs.append(className)
    print(sorted(classesIDs))
    return sorted(classesIDs)

def getTrainingPath(modelName):
    if ("alexnet" in modelName):
        return dataPath_All_227
    if ("vgg16" in modelName):
        return dataPath_All_224
    if ("ciresan" in modelName):
        return dataPath_Freilassing1_All
    if ("vgglike" in modelName):
        return dataPath_All_96


def getTrainingModel(modelName):
    if ("alexnet" in modelName):
        return "network_alexnet"
    if ("vgg16" in modelName):
        return "network_vgg16"
    if ("ciresan" in modelName):
        return "network_ciresan"
    if ("vgglike" in modelName):
        return "network_vgglike"

def getImageSize(modelName):
    if ("alexnet" in modelName):
        return 227
    if ("vgg16" in modelName):
        return 224
    if ("ciresan" in modelName):
        return 48
    if ("vgglike" in modelName):
        return 96
  

