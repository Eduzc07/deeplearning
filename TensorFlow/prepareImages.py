#converts images in training folder to .png format

import os
from scipy.misc import imread
from scipy.misc import imsave
from scipy.misc import imresize

from common import get_immediate_subdirectories
from common import get_immediate_files
from common import getClassesIDs

from common import dataPath_Freilassing_48, dataPath_Freilassing_orig, dataPath_All_orig, dataPath_Freilassing1_All


logoGrouping = { 1: (183, 250), 7: (174, 1622), 8: (297,), 17: (500,), 18: (923,), 21 : (623,), 24: (40, 104, 292, 690), 9 : (239, 473), 25: (202, 233, 248), 36: (278, 867, 1234), 67: (1501,), 100: (1714,), 101: (1862,), 103 : (367,), 110 : (672,), 122 : (638, 1598), 127: (224,),  218 : (490, 392, 396), 74: (94, 411), 62: (172,), 176: (1152,), 177: (270, 295, 848, 918), 184 : (888, 1464, 1465), 199: (574, 688, 686, 1329),  204: (339,), 207: (208, 375, 376, 546), 234: (308, 1458), 279 : (402,), 305: (436,), 343: (1685,), 360: (640,), 360: (640,),  392: (465,), 418: (419, 593), 445: (446,), 450 : (451, 458, 459, 466, 1731), 454: (455, 456, 457), 460 : (461, 462, 463, 1143, 1182, 1334), 486: (1504,),  512: (1043,), 523: (608,),  558: (1936,),560: (668,), 581: (1593,), 599: (1618,), 606: (609,),658: (666,),  660: (711,), 1693: (1721,), 1788: (1789,) }

def getLogoGroupId(logoId, logoGrouping1):
    return logoGrouping1.get(logoId, logoId)


#reads path with trainings data and creates the coding from folder name to trainings class
#reads logo-groupings and assign more folder to the same class where appropiate
def createClassEncodings(*dataSources):
    classIDs = set()    

    #get the list of all class folders in all the data sources folders
    for i in range(len(dataSources)):
        print("Path: "+ dataSources[i])

        folderPaths = get_immediate_subdirectories(dataSources[i])
        for folder in folderPaths:
            filePaths_ext = get_immediate_files(folder)
            filePaths = [s for s in filePaths_ext if "ignore" not in s]
            if len(filePaths) == 0:
                continue           
 
            classDirName = os.path.basename(folder)
            try: 
            	int(classDirName)
            except ValueError:
                print(classDirName + " not integer")
                continue
            classID = int(classDirName)
            classIDs.add(classID)
            print("ClassID " + str(classID))

    classesList=list(classIDs)		

    print("Final classes")
    print(classesList)

    logoGrouping1 = {}
    for key,val in logoGrouping.items():
        if key in classesList:
            for v in val:
                logoGrouping1[v] = key
        else:
            key1 = key
            for v in val:
                if v in classesList:
                    key1 = v
                    break
            if key1 in classesList:
                logoGrouping1[key] = key1
                for v in val:
                    if v != key1:
                        logoGrouping1[v] = key1         
                    

    groupList = set()
    for c in classesList:
        groupList.add(getLogoGroupId(c, logoGrouping1))
    groupList = list(groupList)

    classEncodings = {}
    for c in classesList:
        classEncodings[c] = groupList.index(getLogoGroupId(c, logoGrouping1))

    for c, val in sorted(classEncodings.items()):
        print(str(c) + " " + str(val))

    return classEncodings

def writeClassEncodingsToFile(classEnc, toFolder):
    dictFilePath = os.path.join(toFolder, "ClassEncoding.txt")
    dictFile = open(dictFilePath, "w")
    for c, val in sorted(classEnc.items()):
        dictFile.write(str(c) + "-" + str(val) + "\n")  
    dictFile.close()

def prepareImages(rootPath, targetPath, resizeTo, formatIn, formatOut, classEncodings):
    if not os.path.exists(targetPath):
        os.makedirs(targetPath)

    #dictFilePath = os.path.join(targetPath, "ClassEncoding.txt")
    #dictFile = open(dictFilePath, "w")

    #filename_queue = []
    folderPaths = get_immediate_subdirectories(rootPath)
    for folder in folderPaths:
        filePaths_ext = get_immediate_files(folder)
        filePaths = [s for s in filePaths_ext if "ignore" not in s]

        if len(filePaths) == 0:
            continue
        
        classDirName = os.path.basename(folder)
        try:
            int(classDirName)
        except ValueError:
            continue
        
        classID = int(classDirName) 
        print(classDirName)
        #dictFile.write(str(fCount) + "-" + classDirName + "\n")
        
        for fp in filePaths:
           fileName = os.path.basename(fp)  
           print(fp + " - " + fileName)
           newClassDirName = classEncodings[classID]  
           
           destPath = os.path.join(targetPath, str(newClassDirName), fileName).replace(formatIn, formatOut)
           print(destPath)           
           if os.path.exists(destPath):
               continue           
           dirDestPath = os.path.join(targetPath, str(newClassDirName))
           if not os.path.exists(dirDestPath):
               os.makedirs(dirDestPath)  

           img = imread(fp, flatten=False, mode='RGB')
           img = imresize(img, [resizeTo, resizeTo])
           imsave(destPath, img)

    #dictFile.close()

classEnc = createClassEncodings(dataPath_Freilassing_orig, dataPath_All_orig)
writeClassEncodingsToFile(classEnc, dataPath_Freilassing1_All)
prepareImages(dataPath_Freilassing_orig, dataPath_Freilassing1_All, 48, "ppm", "png", classEnc)
prepareImages(dataPath_All_orig, dataPath_Freilassing1_All, 48, "jpg", "png", classEnc)
