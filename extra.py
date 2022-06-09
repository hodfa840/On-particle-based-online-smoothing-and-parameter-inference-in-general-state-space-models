import numpy as np
from scipy.stats import norm 
def logNormPdf(x,mu,s):
	#result = - 0.5 * ((x - mu)**2)*(s**-2) - 0.5 * np.log(2*np.pi*(s**2))
	return norm.logpdf(x, loc = mu, scale = s).flatten()

def randoind(w,n):
	# Assume that we are giving weights (not log weights)
	wn = w/np.sum(w) # <- Normalize, can we do this better, can we assume it? 
	#indX = np.zeros(n,dtype='int')
	#wc = np.cumsum(w)
	#R = np.random.rand(n)
	#if n == 1:
	indX = np.random.choice(np.size(wn), n, p = wn)
	return indX
		
def backwardDraws(w, xPart, xPartN, data, tt, model, maxIter):
	#print('bd')
	if np.abs(np.sum(w) - 1) > 1e-4:
		wgt = expWgt(w)
	else:
		wgt = w
	#print(wgt.shape)
	#print np.sum(wgt)
	L = np.asarray(range(len(wgt)), dtype = 'int')

	backIndices = np.empty(len(wgt), dtype ='int')
	for i in range(0,maxIter):
		n = len(L)
		#print("n = {n}".format(n=n))
		indX = randoind(wgt,n)
		U = np.log(np.random.uniform(size = (n,1)))
		#acc = U < (logNormPdf(xPartN[L],model.q(xPart[indX],data.yt[tt],tt),
		#	model.qv(xPart[indX],data.yt[tt],tt) ) - model.maxPdf)
		temp =  model.logTransProb(xPartN[L], xPart[indX], data.yt[tt], tt)
		temp = temp.reshape(-1,1)
		#print('Ushape{Ushape}== '.format(Ushape=U.shape))
		#print('tempshape{Tshape}=='.format(Tshape=temp.shape))
		#test = U.shape == temp.shape

		#print(test)
		#print("tempShape: {Tsha}".format(Tsha=temp.shape))
		acc = U < model.logTransProb(xPartN[L], xPart[indX], data.yt[tt], tt)
		#acc = U < temp
		#print U
		#print (logNormPdf(xPartN[L],model.q(xPart[indX],data.yt[tt],tt),
		#	model.qv(xPart[indX],data.yt[tt],tt) ) - maxPdf)
		#print acc
		#print L.shape
		#print acc.shape
		#print('acc shape{accShape}'.format(accShape=acc.shape))
		acc = acc[:,0]
		print(sum(acc)/len(acc))
		#print(backIndices)
		#print acc.shape
		if len(L[acc]) > 0:
			backIndices[L[acc]] = indX[acc]
			#print(L[acc])
			#print('-------------------------------------------------------------------------------')

			L = L[~acc]

		#print len(L[acc])
		if len(L) == 0:
			#print backIndices
			return backIndices

	# Full draw
	for l in L:
		#print(l)
		#print(w.shape)
		temp = model.logTransProb(xPartN[l], xPart, data.yt[tt], tt)
		#print('tempSape {Tshape}'.format(Tshape=temp.shape))
		probs = wgt * np.exp(model.logTransProb(xPartN[l], xPart, data.yt[tt], tt))
		#print(probs.shape)
		#probs = probs[:,0]
		backIndices[l] = randoind(probs,1)
			#ogNormPdf(xPartN[l],model.q(xPart[:,0],data.yt[tt],tt),model.qv(xPart[:,0],data.yt[tt],tt) ) ) ,1)
	#print backIndices
	return backIndices



def expWgt(logW):
	'''Returns normalized weights in a computationally safe way.'''
	const = np.max(logW)
	W = np.exp(logW - const)
	sumofweights = np.sum(W)
	W = W / sumofweights
	return W
