'''kmeans.py
Performs K-Means clustering
YOUR NAME HERE
CS 252 Mathematical Data Analysis Visualization, Spring 2022
'''
from dis import dis
from re import X
from tkinter import N
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors
from scipy import misc


class KMeans():
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        ''' 
        return np.array(self.data, copy = True)

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        dist = pt_2 - pt_1
        dist = np.square(dist)
        dist = np.sum(dist)
        dist = np.sqrt(dist)
        return dist

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        d = centroids-pt
        d = np.square(d)
        d = np.sum(d,axis=1)
        d = np.sqrt(d)
        
        return d

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        shape = self.data.shape
        a = np.random.choice(shape[0],size=k)
        
        ds = self.data
        
        self.k = k
        self.centroids = ds[a,:]
        
        return self.centroids
        
    def initialize_plusplus(self, k):
        '''Initializes K-means by setting the initial centroids (means) according to the K-means++
        algorithm

        (LA section only)

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        TODO:
        - Set initial centroid (i = 0) to a random data sample.
        - To pick the i-th centroid (i > 0)
            - Compute the distance between all data samples and i-1 centroids already initialized.
            - Create the distance-based probability distribution (see notebook for equation).
            - Select the i-th centroid by randomly choosing a data sample according to the probability
            distribution.
        '''
        pass

    def cluster(self, k=2, tol=1e-5, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all 
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        self.initialize(k)
        self.data_centroid_labels = self.assign_labels(self.centroids)
        
        for i in range (max_iter-1):
            newCen,dif = self.update_centroids(k,self.data_centroid_labels,self.centroids)
            self.centroids=newCen
            self.data_centroid_labels = self.assign_labels(newCen)
            
            if abs(np.min(dif)) < tol:
                break
                 
        if verbose:
            print(i+1)
            
        inertia = self.compute_inertia()
        self.inertia = inertia
            
        return inertia,i

    def cluster_batch(self, k=2, n_iter=1, verbose=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        centroids = []
        labels = []
        inertia = []
        
        for i in range (n_iter):
            iner,v = self.cluster(k=k,max_iter=10,verbose=verbose)
            centroids.append(self.centroids)
            labels.append(self.data_centroid_labels)
            inertia.append(iner)
            
        inertia = np.array(inertia)
        lowIner = np.argmin(inertia)
        
        self.centroids = centroids[lowIner]
        self.data_centroid_labels = labels[lowIner]
        self.inertia = inertia[lowIner]
        return inertia
            
    def assign_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray. shape=(self.num_samps,). Holds index of the assigned cluster of each data sample

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        shape = self.data.shape
        labels = []
        
        ds = self.data
        
        for i in range(shape[0]):
            dist = self.dist_pt_to_centroids(ds[i,:], centroids)
            labels.append(np.argmin(dist))
            
        labels = np.array(labels)
            
        return labels

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster.
        
        The basic algorithm is to loop through each cluster and assign the mean value of all 
        the points in the cluster. If you find a cluster that has 0 points in it, then you should
        choose a random point from the data set and use that as the new centroid.

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values
        '''
        NPdata = self.data
        nCentroids = []
        
        for i in range(k):
            pt = NPdata[data_centroid_labels == i]
            ptMean = np.mean(pt,axis=0)
            nCentroids.append(ptMean)
            
        dif = (nCentroids-prev_centroids)
        
        nCentroids = np.array(nCentroids)
        
        return nCentroids, dif
            
    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Parameters:
        -----------
        None

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        NPdata = self.data
        distSum = 0
        
        for i in range (len(self.centroids)):
            pts = (NPdata[self.data_centroid_labels == i])
            cen = np.array(self.centroids)
            cen = cen[i,:]
            
            dis = pts - cen
            dis = np.square(dis)
            dis = np.sum(dis, axis=1)
            #dis = np.sqrt(dis)
            distSum += np.sum(dis)
        
        shape = NPdata.shape
        return(distSum/shape[0])

    def plot_clusters(self):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        
        dataNP = self.data
        
        for i in range (self.k):
            pt = dataNP[self.data_centroid_labels == i]
            plt.scatter(pt[:,0],pt[:,1], label=i)
        
        #print(self.centroids)  
        plt.scatter(self.centroids[:,0],self.centroids[:,1],label = 'Centroids', c = 'black')
        print(self.inertia)
        plt.xlabel("X")
        plt.ylabel("Y")
        #plt.legend()
        plt.show()

    def elbow_plot(self, max_k, n_iter=3):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        inertia = []
        x = []
        
        for i in range(max_k):
            #print(i)
            self.cluster_batch(k=i+1,n_iter=n_iter)
            inertia.append(self.inertia)
            x.append(i+1)

        plt.title('Elbow Plot')

        #print(inertia) 
        #print(x)  
        plt.plot(x, inertia,'bx-')
        plt.xlabel("K")
        plt.ylabel("inertia")
        plt.show()

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        new_data = np.zeros((self.num_samps, self.num_features))
        
        for i in range(self.num_samps):
            new_data[i] = self.centroids[self.data_centroid_labels[i]]

        self.data = new_data