from random import uniform
import numpy as np
import matplotlib.pyplot as plt

trainingVector = []
realData = []
learningRate = 0.1
w = np.zeros((1, 3)).flatten()
b = np.zeros((1, 3)).flatten()


def calcRealLabel(input):
	return NormalDistFunction()


def createRealData():
	for x in range(0, 50):
		temp = uniform(0, 9)
		label = calcRealLabel(temp)
		sample = (temp, label)
		realData.append(sample)


def createTestDistForGivenY(mu, sigma):
	vecX = np.random.normal(mu, sigma, 100)
	vecY = 100 * [mu / 2]
	trainingVector.extend(zip(vecX, vecY))


def createTrainingData():
	for x in range(1, 4):
		createTestDistForGivenY(2 * x, 1)
	np.random.shuffle(trainingVector)


def NormalDistFunction(x, a):
	denominator = 1 / np.sqrt(2 * np.pi)
	numerator = np.exp(-((x - 2 * a) ** 2) / 2)
	return numerator * denominator


def calcModelresultForX(x):
	temp = (w * x) + b
	temp = np.exp(temp) / np.sum(np.exp(temp))
	return (np.unravel_index(temp.argmax(), temp.shape)[0], temp)


def calcNew_W_And_B(modelClassVec, originClassNum, sampleValue):
	modelClassVec[originClassNum - 1] -= 1
	temp = learningRate * sampleValue * modelClassVec
	global w, b
	w = w - temp
	b = b - learningRate * modelClassVec


def leranTheModel():
	for sample in trainingVector:
		result = calcModelresultForX(sample[0])
		calcNew_W_And_B(result[1], sample[1], sample[0])


def plotGraphs():
	# plt.plot(, , "r--", label='Model')
	# plt.show()
	return


def main():
	createTrainingData()
	for x in range(0, 5):
		leranTheModel()
	createRealData()
	createRealData()
	plotGraphs()


if __name__ == "__main__":
	main()
