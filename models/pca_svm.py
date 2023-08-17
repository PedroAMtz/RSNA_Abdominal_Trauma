from sklearn.decomposition import PCA
from sklearn import svm

def pca_model(input_array, num_components=3):
	pca = PCA(num_components=num_components)
	x = pca.fit(input_array)
	return x

def svm_classifier(features, target, kernel="linear"):
	clf = svm.SVC(kernel=kernel)
	x = clf.fit(features, target)
	return x

#if __name__ == "__main__":
	
