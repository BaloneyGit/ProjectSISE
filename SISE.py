from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
import pickle

def unpickle(file): # use for unpickling pickled images from your dataset
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class FeatureExtractorSISE:
    def __init__(self):
        base_model = VGG19(weights='imagenet')
        self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
        
    def extractSISEQ(self, img): # for query image 
        z = np.resize(img, (3, 224, 224))
        z1 = z.transpose(1, 2, 0)
        z2 = np.expand_dims(z1, axis=0)
        z3 = preprocess_input(z2)                
        feature = self.model.predict(z3)[0]       # (1, 4096) -> (4096,)
        return feature/np.linalg.norm(feature)   # Normalizing
    
    def extractSISED(self, dataset_img): # for dataset images
        dataset_img = np.resize(dataset_img, (3, 224, 224))
        dataset_img = dataset_img.transpose(1, 2, 0)
        exp_dims_datasetimgs = np.expand_dims(dataset_img, axis=0)
        pp_input_datasetimgs = preprocess_input(exp_dims_datasetimgs)                
        feature_datasetimgs = self.model.predict(pp_input_datasetimgs)[0]       # (1, 4096) -> (4096,)
        return feature_datasetimgs/np.linalg.norm(feature_datasetimgs)   # Normalizing
    
class SISE_functions:
    def findQueryImageFeatures(test_image):
        return FeatureExtractorSISE().extractSISEQ(test_image)

    def findDatasetImageFeatures(training_batch): # training_batch: singular batch of training images
        featr = [(ind, FeatureExtractorSISE().extractSISED(feat)) for ind, feat in enumerate(training_batch)] # (index, dataset image feature)
        return np.array(featr)

    def findDist(queryImgFeatr, datasetImgFeatr, topLimit): # queryImgFeatr: output of findQueryImageFeatures();
                                                            # datasetImgFeatr: output of findDatasetImageFeatures();
                                                            # topLimit: no. of desired top result images, eg: 'topLimit = 8' gives the top 8 images   
        distIndex = [(i[0], np.linalg.norm(i[1]-queryImgFeatr)) for i in datasetImgFeatr] # (index, dist between dataset img feature & query image feature)

        dist = [i[1] for i in distIndex] # only distances
        dist = np.array(dist)

        sortedDistsIndices = np.argsort(dist) # indices of sorted distances
        sortedDists = dist[sortedDistsIndices] # sorted distances

        topResults = []
        for i in range(0, topLimit):
            topResults.append((sortedDistsIndices[i], sortedDists[i]))

        return topResults

    def displayOutputImages(dataset, topResultsList): # dataset: entire training images dataset; topResultsList: output of findDist() func
        rows = 2
        columns = len(topResultsList)
        fig = plt.figure(figsize=(20, 7))
        for i in range(0, len(topResultsList)):
            imagePlot = dataset[topResultsList[i][0]]
            imagePlot = imagePlot.reshape(3, 32, 32)
            imagePlot = imagePlot.transpose(1, 2, 0)
            fig.add_subplot(rows, columns, (i+1))
            plt.imshow(imagePlot)
            plt.axis('off')