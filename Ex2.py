import numpy as np
import matplotlib.pyplot as plt

trainingVector = []
realData = []
realDataLabels = []
learningRate = 0.08
w = np.zeros((1, 3)).flatten()
b = np.zeros(3).flatten()


def calcRealLabelProb(inp):
	labels = []
	for x in range(1, 4):
		labels.append(NormalDistFunction(inp, 2 * x))
	return labels


def createRealData():
	global realData,realDataLabels
	xVec = np.linspace(0, 10, 100)
	for x in xVec:
		labels = calcRealLabelProb(x)
		realDataLabels.append(labels[0] / sum(labels))
		realData.append(x)


def createTrainingData():
	global trainingVector
	for x in range(1, 4):
		vecX = np.random.normal(2 * x, 1, 100)
		vecY = 100 * [x]
		trainingVector.extend(zip(vecX, vecY))


def NormalDistFunction(x, a):
	numerator = np.exp((-(((x - a) ** 2) / 2)))
	denominator = np.sqrt(2 * np.pi)
	return numerator / denominator


def calcSoftMaxresultForX(x):
	res = 0
	global w, b, learningRate
	temp = (w * x) + b
	softMaxResult = np.exp(temp) / np.sum(np.exp(temp))
	first = softMaxResult[0]
	for index, classNum in enumerate(softMaxResult):
		if (classNum > first):
			res = index
			first = classNum
	return (res, softMaxResult)


def calcNew_W_And_B(softMaxVec, trueClassNum, sampleValue):
	global w, b, learningRate
	softMaxVec[trueClassNum - 1] -= 1
	temp = learningRate * sampleValue * softMaxVec
	for i in range(len(w)):
		w[i] = w[i] - (temp[i])
		b[i] = b[i] - learningRate * softMaxVec[i]


def ModelTrainer():
	for sample in trainingVector:
		result = calcSoftMaxresultForX(sample[0])
		calcNew_W_And_B(result[1], sample[1], sample[0])


def plotGraphs():
	modelLabel = []

	for x in realData:
		temp = calcSoftMaxresultForX(x)
		modelLabel.append(temp[1][0])

	plt.plot(realData, modelLabel, "r", label='Model')
	plt.plot(realData, realDataLabels, "b--", label='Real')
	plt.legend(('Model', 'Real'))
	plt.show()
	return


def main():
	createTrainingData()
	for x in range(0, 10):
		np.random.shuffle(trainingVector)
		ModelTrainer()

	createRealData()
	plotGraphs()


if __name__ == "__main__":
	main()
