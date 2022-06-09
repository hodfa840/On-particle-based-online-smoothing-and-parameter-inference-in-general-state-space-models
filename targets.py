import numpy as np

class lgssSuffStat():
	"""docstring for lgssSuffStat"""
	dimension = 4

	def fun(self,xt,yt,xtprev,ytprev,tt,nPart):
		val = np.zeros((self.dimension,nPart))
		val[0,:] = (xtprev**2).squeeze()
		val[1,:] = (xt**2).squeeze()
		val[2,:] = (xt*xtprev).squeeze()
		val[3,:] = ((yt - xt)**2).squeeze()
		return val

class meanFun():
	"""docstring for meanFun"""
	dimension = 1
	
	def fun(self,xt,yt,xtprev,ytprev,tt,nPart):
		val = np.zeros((self.dimension,nPart))
		val[0,:] = xt[:,0]
		return val

class testFun():
	"""docstring for testFun"""
	dimension = 3
	
	def fun(self, xt, yt, xtprev, ytprev, tt, nPart):
		val = np.zeros((self.dimension,nPart))
		val[0,:] = (xt**2).squeeze()
		val[1,:] = xt.squeeze()
		val[2,:] = np.multiply(xt, xtprev).squeeze()
		return val

     
        
        
        
