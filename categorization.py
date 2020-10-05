import cv2
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
import glob
from sklearn import svm
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class Visual_BOW():
    def __init__(self, k=20, dictionary_size=50):
        self.k = k  # number of SIFT features to extract from every image
        self.dictionary_size = dictionary_size  # size of your "visual dictionary" (k in k-means)
        self.n_tests = 10  # how many times to re-run the same algorithm (to obtain average accuracy)

    def extract_sift_features(self):
        '''
        To Do:
            - load/read the Caltech-101 dataset
            - go through all the images and extract "k" SIFT features from every image
            - divide the data into training/testing (70% of images should go to the training set, 30% to testing)
        Useful:
            k: number of SIFT features to extract from every image
        Output:
            train_features: list/array of size n_images_train x k x feature_dim
            train_labels: list/array of size n_images_train
            test_features: list/array of size n_images_test x k x feature_dim
            test_labels: list/array of size n_images_test
        '''

        labels= []
        features = []
        for x in glob.glob('101_ObjectCategories/*'):
            for y in glob.glob(x+'/*'):
                labels.append(x[21:len(x)])
        for item in glob.glob('101_ObjectCategories/*/*'):
            img = cv2.imread(item)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            sift = cv2.xfeatures2d.SIFT_create(nfeatures= self.k)
            kp, des = sift.detectAndCompute(gray,None)
            if (des is None):
                del labels[len(features)]
            if (des is not None):
                features.append(des)

        train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=0.7,test_size=0.3)

        return train_features, train_labels, test_features, test_labels

    def create_dictionary(self, features):
        '''
        To Do:
            - go through the list of features
            - flatten it to be of size (n_images x k(20)) x feature_dim(128) (from 3D to 2D)
            - use k-means algorithm to group features into "dictionary_size" groups
        Useful:
            dictionary_size: size of your "visual dictionary" (k in k-means)
        Input:
            features: list/array of size n_images x k x feature_dim
        Output:
            set k = 8 to make it work
            kmeans: trained k-means object (algorithm trained on the flattened feature list)
        '''
        flattened = []
        for x in range(0,len(features)):
            for y in range(0, len(features[x])):
                flattened.append(features[x][y])
        kmeans = MiniBatchKMeans(n_clusters= self.dictionary_size).fit(flattened)
        return kmeans

    def convert_features_using_dictionary(self, kmeans, features):
        '''
        To Do:
            - go through the list of features (images)
            - for every image go through "k" SIFT features that describes it
            - every image will be described by a single vector of length "dictionary_size"
            and every entry in that vector will indicate how many times a SIFT feature from a particular
            "visual group" (one of "dictionary_size") appears in the image. Non-appearing features are set to zeros.
        Input:
            features: list/array of size n_images x k x feature_dim
        Output:
            features_new: list/array of size n_images x dictionary_size
        '''

        features_new = []
        for x in features:
            x = x.astype(np.float64)
            temp_means = kmeans.predict(x)
            corrected = [0] * self.dictionary_size
            for i in temp_means:
                corrected[i] = corrected[i]+1
            features_new.append(corrected)
        return features_new

    def train_svm(self, inputs, labels):
        '''
        To Do:
            - train an SVM classifier using the data
            - return the trained object
        Input:
            inputs: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        Output:
            clf: trained svm classifier object (algorithm trained on the inputs/labels data)
        '''
        x = svm.SVC(kernel='linear')
        clf = x.fit(inputs, labels)
        return clf

    def test_svm(self, clf, inputs, labels):
        '''
        To Do:
            - test the previously trained SVM classifier using the data
            - calculate the accuracy of your model
        Input:
            clf: trained svm classifier object (algorithm trained on the inputs/labels data)
            inputs: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        Output:
            accuracy: percent of correctly predicted samples
        '''
        label_predict = clf.predict(inputs)

        accuracy = 0
        for x in range(0,len(label_predict)):
            if (label_predict[x]==labels[x]):
                accuracy = accuracy + 1

        accuracy = accuracy/len(labels)

        return accuracy

    def save_plot(self, features, labels):
        '''
        To Do:
            - perform PCA on your features
            - use only 2 first Principle Components to visualize the data (scatter plot)
            - color-code the data according to the ground truth label
            - save the plot
        Input:
            features: new features (converted using the dictionary) of size n_images x dictionary_size
            labels: list/array of size n_images
        '''
        color_guide = []
        t = 2
        for m in glob.glob('101_ObjectCategories/*'):
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            color_guide.append([m[21:len(m)],color])
        colors = []
        for l in labels:
            for w in color_guide:
                if(l == w[0]):
                    colors.append(w[1])

        pca = PCA(n_components=2)
        pca.fit(features)
        features = pca.transform(features)
        one = []
        two = []

        for x in features:
            one.append(x[0])
            two.append(x[1])
        plt.scatter(one,two, c = colors, s = 5)
        plt.title("Plot for k = "+ str(self.k) + " and Dictionary Size = "+ str(self.dictionary_size))
        plt.savefig('plot'+str(self.k)+'x'+str(self.dictionary_size)+'.png')


############################################################################
################## DO NOT MODIFY ANYTHING BELOW THIS LINE ##################
############################################################################

    def algorithm(self):
        # This is the main function used to run the program
        # DO NOT MODIFY THIS FUNCTION
        accuracy = 0.0
        for i in range(self.n_tests):
             train_features, train_labels, test_features, test_labels = self.extract_sift_features()
             kmeans = self.create_dictionary(train_features)
             train_features_new = self.convert_features_using_dictionary(kmeans, train_features)
             classifier = self.train_svm(train_features_new, train_labels)
             test_features_new = self.convert_features_using_dictionary(kmeans, test_features)
             accuracy += self.test_svm(classifier, test_features_new, test_labels)
             self.save_plot(test_features_new, test_labels)
        accuracy /= self.n_tests
        return accuracy

if __name__ == "__main__":
    alg = Visual_BOW(k=20, dictionary_size=50)
    accuracy = alg.algorithm()
    print("Final accuracy of the model is:", accuracy)

