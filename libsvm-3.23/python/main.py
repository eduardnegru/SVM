import sys
import math
# sys.path.append('/home/adrian/Desktop/ml/SVM/')
from svmutil import *
from matplotlib import pyplot as plt

f=open("output", "a")

# https://medium.com/all-things-ai/in-depth-parameter-tuning-for-svc-758215394769

# degree is a parameter used when kernel is set to ‘poly’. 
# It’s basically the degree of the polynomial used to find the hyperplane to split the data.

# C is the penalty parameter of the error term. It controls the trade off between smooth 
# decision boundary and classifying the training points correctly.

# gamma is a parameter for non linear hyperplanes. The higher the gamma value it tries 
# to exactly fit the training data set


#0 -- linear: u'*v
# 1 -- polynomial: (gamma*u'*v + coef0)^degree
# 2 -- radial basis function: exp(-gamma*|u-v|^2)
# 3 -- sigmoid: tanh(gamma*u'*v + coef0)

def confusion_matrix_get(predicted, real):
	
	tn, tp, fn, fp = 0, 0, 0, 0

	#skin_noskin 1 or 2
	for index, pred in enumerate(predicted):
		if int(pred) == int(real[index]):
			if int(pred) == 1:
				tp += 1
			else:
				tn += 1
		else:
			if int(pred) == 1:
				fp += 1
			else:
				fn += 1
	
	return {
		"tn": tn,
		"tp": tp,
		"fn": fn,
		"fp": fp,
	}


kernel_types = [0, 1, 2, 3]


kernel_parameters = {
	"1": {
		"d": [5, 6],
		"g": [0.1, 1, 10, 100]
	},
	"2": {
		"c": [0.1, 1, 10, 100, 1000],
		"d": [1, 2, 3, 4, 5, 6],
		"g": [0.1, 1, 10, 100]
	},
	"3": {
		"c": [0.1, 1, 10, 100, 1000],
		"d": [1, 2, 3, 4, 5, 6],
		"g": [0.1, 1, 10, 100]
	}
}
# kernel_parameters = {
# 	"0": {
# 		# no parameters to vary for linear kernels
# 	},
# 	"1": {
# 		"c": [0.1, 1, 10, 100, 1000],
# 		"d": [1, 2, 3, 4, 5, 6],
# 		"g": [0.1, 1, 10, 100]
# 	},
# 	"2": {
# 		"c": [0.1, 1, 10, 100, 1000],
# 		"d": [1, 2, 3, 4, 5, 6],
# 		"g": [0.1, 1, 10, 100]
# 	},
# 	"3": {
# 		"c": [0.1, 1, 10, 100, 1000],
# 		"d": [1, 2, 3, 4, 5, 6],
# 		"g": [0.1, 1, 10, 100]
# 	}
# }

dataset_name = "skin_noskin"
# Read data in LIBSVM format
y, x = svm_read_problem(dataset_name)
data_size = len(x)
train_size = math.ceil(0.8 * data_size)

for kernel_type in kernel_parameters:
	if len(kernel_parameters[kernel_type].keys()) == 0:
		print ("Training with default parameters", file=f)
	else:
		for parameter in kernel_parameters[kernel_type]:
			error = {}
			for parameter_value in kernel_parameters[kernel_type][parameter]:
				
				print("\n\nTraining SVM kernel type " + str(kernel_type) + " " + str(parameter) + " " + str(parameter_value), file=f)
				strParameters = "-q -t " + str(kernel_type) + " -" + str(parameter) + " " + str(parameter_value)
				strFileName = str(kernel_type) + "_" + str(parameter) + "_" + str(parameter_value)

				m = svm_train(y[:train_size], x[:train_size], strParameters)
				# print("Finished training")
				print("Support vectors", len(m.get_SV()), file=f)
				svm_save_model("./skin/" + dataset_name + "_" + strFileName + ".model", m)

				p_label, p_acc, p_val = svm_predict(y[train_size:], x[train_size:], m)
				confusion_matrix_train = confusion_matrix_get(p_label, y[train_size:])
				print(confusion_matrix_train, file=f)
				error[parameter_value] = p_acc[1]

				p_label, p_acc, p_val = svm_predict(y[:train_size], x[:train_size], m)
				confusion_matrix_test = confusion_matrix_get(p_label, y[:train_size])
				print(confusion_matrix_test, file=f)

			plt.suptitle("Dataset=Skin_noskin, Kernel=" + str(kernel_type) + ", Param=" + str(parameter))
			plt.xlabel(parameter)
			plt.ylabel("error")
			plt.plot(error.keys(), error.values(), marker='o')
			plt.savefig("./skin/skin_noskin" + "_" + str(kernel_type) + "_" + str(parameter))
			# plt.show()

f.close()