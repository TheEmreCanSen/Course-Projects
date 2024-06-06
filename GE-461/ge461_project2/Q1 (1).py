import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

labels = np.loadtxt("labels.txt")
digits = np.loadtxt("digits.txt")
copied_labels = labels.copy()
copied_digits = digits.copy()

#######Question 1.1#########
digits_mean = np.mean(digits, axis=0)
digits_center = digits - digits_mean

#######Question 1.2########
x_train, x_test, y_train, y_test = train_test_split(digits_center, labels, test_size = 0.5, random_state = 916, shuffle = True)

digits_pca = PCA(n_components=400)
digits_pca.fit_transform(x_train)
digits_pca_components = digits_pca.components_
digits_pca_vectors = digits_pca.explained_variance_
pca_Mean = digits_pca.mean_
pcaPercentageOfVariance = digits_pca.explained_variance_ratio_
print("PCA calculations end.")

plt.plot(digits_pca_vectors)
plt.xlabel("Components")
plt.ylabel("Eigen Values")
plt.title("Eigen Values")
plt.show()

#####Question 1.3######
xorg_train, xorg_test, yorg_train, yorg_test = train_test_split(copied_digits, copied_labels, test_size = 0.5, random_state = 916, shuffle = True)
orgdigits_pca = PCA(n_components=400)
orgdigits_pca.fit_transform(xorg_train)
digits_pca_components = orgdigits_pca.components_
digits_pca_vectors = orgdigits_pca.explained_variance_
pca_Mean = orgdigits_pca.mean_
pcaPercentageOfVariance = orgdigits_pca.explained_variance_ratio_
x_copied_mean = orgdigits_pca.mean_.reshape(20,20).T
image2 = plt.imshow(x_copied_mean)
plt.title("X Train Data Mean Transposed")
plt.show()

for i in range(70):    
    plt.subplot(10, 10, i+1)
    plt.axis("off")
    plt.imshow(digits_pca_components[i].reshape(20,20).T)
plt.suptitle("Eigenvectors of Training Data Sample Mean (not centered)")
plt.show()

#####Question 1.4#####

component_count = 200
train = np.zeros((200, 2))
test = np.zeros((200, 2))

for i in range(component_count):
    i = i+1
    pcaResult = PCA(n_components = i, random_state=310).fit(x_train) 

    x_train_transform = pcaResult.transform(x_train)
    x_test_transform = pcaResult.transform(x_test)
   
    gaussian = GaussianNB() 
    gaussian.fit(x_train_transform, (y_train.T)) 
    x_train_prediction = gaussian.predict(x_train_transform) 
    x_train_accuracy = metrics.accuracy_score((y_train.T), x_train_prediction)
    x_train_error = 1 - x_train_accuracy 
    x_test_prediction = gaussian.predict(x_test_transform) 
    x_test_accuracy = metrics.accuracy_score((y_test.T), x_test_prediction)
    x_test_error = 1 - x_test_accuracy
    train[i-1, 1] = x_train_error 
    train[i-1, 0] = i 
    test[i-1, 1] = x_test_error 
    test[i-1, 0] = i 

#####Question 1.5#####
plt.plot(train[:, 0], train[:, 1], label="training")
plt.plot(test[:, 0], test[:, 1], label="test")
plt.legend()
plt.title("Training vs Test")
plt.xlabel("Number of Components")
plt.ylabel("Classification Error")
plt.show()

plt.plot(train[:, 0], train[:, 1], label="training") 
plt.title("Training Set Results")
plt.xlabel("Number of Components")
plt.ylabel("Classification Error")
plt.show()

plt.plot(test[:, 0], test[:, 1], label="test")
plt.title("Test Set Results")
plt.xlabel("Number of Components")
plt.ylabel("Classification Error")
plt.show()



