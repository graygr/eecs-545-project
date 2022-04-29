import matplotlib.pyplot as plt
import csv
import numpy as np

# simple bg
naive = np.array(list(csv.reader(open('simple-bg-naive-class_stats.csv')))).astype('float').astype('int')
kmeans = np.array(list(csv.reader(open('simple-bg-kmeans-class_stats.csv')))).astype('float').astype('int')
kernel = np.array(list(csv.reader(open('simple-bg-kmeans_kernel-class_stats.csv')))).astype('float').astype('int')
gmm = np.array(list(csv.reader(open('simple-bg-gmm-class_stats_newfeature.csv')))).astype('float').astype('int')
gt = np.array(list(csv.reader(open('simple-bg-naive-gt_interp.csv')))).astype('float').astype('int')

plt.subplot(2,2,1)
plt.plot(naive[:, 0], naive[:, 2], label="Naive", color='black')
plt.plot(kmeans[:, 0], kmeans[:, 2], label="KMeans", color='green')
plt.plot(kernel[:, 0], kernel[:, 2], label="KMeans with Kernel", color='grey')
plt.plot(gmm[:, 0], gmm[:, 2], label="GMM", color='purple')
plt.plot(gt[:, 0], gt[:, 2], label="Ground Truth", color='red')
plt.title('Predicted Debris Objects, simple bg')
plt.ylabel('# Debris Objects')
plt.ylim((0,60))
plt.grid()
plt.legend()

# simple fg
naive = np.array(list(csv.reader(open('simple-fg-naive-class_stats.csv')))).astype('float').astype('int')
kmeans = np.array(list(csv.reader(open('simple-fg-kmeans-class_stats.csv')))).astype('float').astype('int')
kernel = np.array(list(csv.reader(open('simple-fg-kmeans_kernel-class_stats.csv')))).astype('float').astype('int')
gmm = np.array(list(csv.reader(open('simple-fg-gmm-class_stats_newfeature.csv')))).astype('float').astype('int')
gt = np.array(list(csv.reader(open('simple-fg-naive-gt_interp.csv')))).astype('float').astype('int')

plt.subplot(2,2,2)
plt.plot(naive[:, 0], naive[:, 2], label="Naive", color='black')
plt.plot(kmeans[:, 0], kmeans[:, 2], label="KMeans", color='green')
plt.plot(kernel[:, 0], kernel[:, 2], label="KMeans with Kernel", color='grey')
plt.plot(gmm[:, 0], gmm[:, 2], label="GMM", color='purple')
plt.plot(gt[:, 0], gt[:, 2], label="Ground Truth", color='red')
plt.title('Predicted Debris Objects, simple fg')
plt.ylabel('# Debris Objects')
plt.ylim((0,120))
plt.grid()

# complex bg
naive = np.array(list(csv.reader(open('complex-bg-naive-class_stats.csv')))).astype('float').astype('int')
kmeans = np.array(list(csv.reader(open('complex-bg-kmeans-class_stats.csv')))).astype('float').astype('int')
kernel = np.array(list(csv.reader(open('complex-bg-kmeans_kernel-class_stats.csv')))).astype('float').astype('int')
gmm = np.array(list(csv.reader(open('complex-bg-gmm-class_stats_newfeature.csv')))).astype('float').astype('int')
gt = np.array(list(csv.reader(open('complex-bg-naive-gt_interp.csv')))).astype('float').astype('int')

plt.subplot(2,2,3)
plt.plot(naive[:, 0], naive[:, 2], label="Naive", color='black')
plt.plot(kmeans[:, 0], kmeans[:, 2], label="KMeans", color='green')
plt.plot(kernel[:, 0], kernel[:, 2], label="KMeans with Kernel", color='grey')
plt.plot(gmm[:, 0], gmm[:, 2], label="GMM", color='purple')
plt.plot(gt[:, 0], gt[:, 2], label="Ground Truth", color='red')
plt.title('Predicted Debris Objects, complex bg')
plt.xlabel('Frame')
plt.ylabel('# Debris Objects')
plt.ylim((0,60))
plt.grid()

# complex fg
naive = np.array(list(csv.reader(open('complex-fg-naive-class_stats.csv')))).astype('float').astype('int')
kmeans = np.array(list(csv.reader(open('complex-fg-kmeans-class_stats.csv')))).astype('float').astype('int')
kernel = np.array(list(csv.reader(open('complex-fg-kmeans_kernel-class_stats.csv')))).astype('float').astype('int')
gmm = np.array(list(csv.reader(open('complex-fg-gmm-class_stats_newfeature.csv')))).astype('float').astype('int')
gt = np.array(list(csv.reader(open('complex-fg-naive-gt_interp.csv')))).astype('float').astype('int')

plt.subplot(2,2,4)
plt.plot(naive[:, 0], naive[:, 2], label="Naive", color='black')
plt.plot(kmeans[:, 0], kmeans[:, 2], label="KMeans", color='green')
plt.plot(kernel[:, 0], kernel[:, 2], label="KMeans with Kernel", color='grey')
plt.plot(gmm[:, 0], gmm[:, 2], label="GMM", color='purple')
plt.plot(gt[:, 0], gt[:, 2], label="Ground Truth", color='red')
plt.title('Predicted Debris Objects, complex fg')
plt.xlabel('Frame')
plt.ylabel('# Debris Objects')
plt.ylim((0,80))
plt.grid()
plt.show()