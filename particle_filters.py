import numpy as np
import extra
import targets

class bootStrapPF(object):
	"""Bootstrap particle filter"""
	
	model = []; #Standard
	nPart = [];
	target = [];
	data = [];
	filtMean = [];


	def setup(self, model, nPart, data, target):
		self.model = model
		self.nPart = nPart
		self.data = data
		self.target = target

	def run(self):
		tt = 0;
		xPart = np.zeros((self.nPart,self.model.Xdim)) # Current
		xPartN = np.zeros((self.nPart,self.model.Xdim)) # New
		weights = np.zeros(self.nPart) # Current
		weightsN = np.zeros(self.nPart) # New
		self.filtMean = np.zeros((self.target.dimension,self.data.T))
		# Starting the bootstrap particle filter here:

		xPartN = self.model.propagate(xPart, self.data.yt[tt], tt, self.nPart)
		weightsN = self.model.weightFun(xPartN, self.data.yt[tt], tt)
		print(np.shape(weightsN))
		print(np.shape(self.target.fun(xPartN,self.data.yt[tt],xPart,0,tt,self.nPart)))
		self.filtMean[:,0] = np.average(self.target.fun(xPartN,self.data.yt[tt],xPart,0,tt,self.nPart),
			weights = (np.exp(weightsN)/np.sum(np.exp(weightsN))),axis = 1)
		xPart = xPartN
		weights = weightsN
		for tt in range(1,self.data.T):
			print(tt)
			# Resample
			indX = extra.randoind(np.exp(weights),self.nPart)
			# Propagate
			xPartN = self.model.propagate(xPart[indX], self.data.yt[tt], tt, self.nPart)
			weightsN = self.model.weightFun(xPartN, self.data.yt[tt], tt)
			
			self.filtMean[:,tt] = np.average(self.target.fun(xPartN,self.data.yt[tt],xPart,
				self.data.yt[tt-1],tt,self.nPart), weights = np.exp(weightsN)/np.sum(np.exp(weightsN)), 
				axis = 1)

			# Set
			xPart = xPartN
			weights = weightsN

class PaRIS(object):
	model = [] #Standard
	nPart = []
	target = []
	data = []
	filtMean = []
	nBackDraws = []

	def setup(self, model, nPart, nBackDraws, data, target):
		self.model = model
		self.nPart = nPart
		self.nBackDraws = nBackDraws
		self.data = data
		self.target = target

	def run(self):
		tt = 0;
		xPart = np.zeros((self.nPart,self.model.Xdim)) # Current
		xPartN = np.zeros((self.nPart,self.model.Xdim)) # New
		weights = np.zeros(self.nPart) # Current
		weightsN = np.zeros(self.nPart) # New
		self.filtMean = np.zeros((self.target.dimension,self.data.T))
		tStat = np.zeros((self.target.dimension,self.nPart))
		# Starting the bootstrap particle filter here:

		xPartN = self.model.propagate(xPart, self.data.yt[tt], tt, self.nPart)
		weightsN = self.model.weightFun(xPartN, self.data.yt[tt], tt)


		xPart = xPartN
		weights = weightsN
		for tt in range(1,self.data.T):
			#print(tt)
			# Resample
			indX = extra.randoind(np.exp(weights),self.nPart)
			# Propagate
			xPartN = self.model.propagate(xPart[indX], self.data.yt[tt], tt, self.nPart)
			weightsN = self.model.weightFun(xPartN, self.data.yt[tt], tt)

			tStatN = np.zeros(np.shape(tStat))
			# Backward draws
			for j in range(0,self.nBackDraws):
				bInd = extra.backwardDraws(weights,xPart,xPartN,self.data,tt,self.model, 
					int(np.ceil(np.sqrt(self.nPart))))
				tStatN = tStatN + (tStat[:,bInd] + self.target.fun(xPartN,self.data.yt[tt],
					xPart[bInd],self.data.yt[tt-1],tt,self.nPart))/self.nBackDraws
			tStat = tStatN
			self.filtMean[:,tt] = np.average(tStat, weights = np.exp(weightsN)/np.sum(np.exp(weightsN)),
				axis = 1)
			# Set
			xPart = xPartN
			weights = weightsN

	def __str__(self):
		return('PaRIS algorithm \n N = {0}, K = {1}'.format(self.nPart, self.nBackDraws))



