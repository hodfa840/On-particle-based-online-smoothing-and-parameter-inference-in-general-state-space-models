import numpy as np
import extra
from scipy.stats import multivariate_normal as mvn
from scipy.stats import binom

class lgss(object):
	"""Linear Gaussian state space model"""

	# X(t+1) = q(x(t),y(t),t) + qv(x(t),y(t),t)*v(t) 
	# Y(t) = g(x(t),y(t),t) + gv(x(t),y(t),t)*e(t)
	#

	nPar	= 4;
	par		= np.zeros(4);
	maxPdf	= 0;
	Xdim	= 1;
	def updateMaxPdf(self):
		#self.maxPdf = - 0.5 * np.log(2*np.pi*self.par[1]**2)
		self.maxPdf = extra.logNormPdf(0,0,self.par[1])

	def q(self, xt, yt, tt):
		return self.par[0]*xt;

	def qv(self, xt, yt, tt):
		return self.par[1]*np.ones_like(xt)

	def g(self, xt, yt, tt):
		return self.par[2]*xt;

	def gv(self, xt, yt, tt):
		return self.par[3]*np.ones_like(xt);

	def weightFun(self, xt, yt, tt):
		weights = extra.logNormPdf(yt, self.g(xt, yt, tt), self.gv(xt, yt, tt));
		return weights

	def propagate(self, xt, yt, tt, nPart):
		q_part = self.q(xt,yt,tt)
		qv_part = self.qv(xt,yt,tt)
		w_part = np.random.randn(nPart,self.Xdim)
		#print q_part.shape
		#print qv_part.shape
		#print w_part.shape
		
		#print type(q_part)
		#print type(qv_part)
		
		#print type(w_part)
		x_new = q_part + qv_part*w_part
		#print x_new.shape
		return x_new

	def logTransProb(self, xtN, xtO, yt, tt):
		return extra.logNormPdf(xtN, self.q(xtO, yt, tt), self.qv(xtO, yt, tt)) - self.maxPdf

	def __str__(self):
		return "Linear Gaussian SSM \n X(t+1) = {0} * X(t) + {1} * v(t+1) \n Y(t) = {2} * X(t) + {3} * e(t)".format(
				self.par[0], self.par[1], self.par[2], self.par[3])


class stochVol(object):
	"""docstring for stochVol"""
	
	# X(t+1) = q(x(t),y(t),t) + qv(x(t),y(t),t)*v(t) 
	# Y(t) = g(x(t),y(t),t) + gv(x(t),y(t),t)*e(t)
	#

	nPar	= 3;
	par		= np.zeros(3);
	maxPdf	= 0;
	Xdim	= 1;

	def updateMaxPdf(self):
		#self.maxPdf = - 0.5 * np.log(2*np.pi*self.par[1]**2)
		self.maxPdf = extra.logNormPdf(0,0,self.par[1])

	def q(self, xt, yt, tt):
		return self.par[0]*xt;

	def qv(self, xt, yt, tt):
		return self.par[1]*np.ones_like(xt)

	def g(self, xt, yt, tt):
		return np.zeros_like(xt)

	def gv(self, xt, yt, tt):
		return self.par[2]*np.exp(0.5*xt);

	def weightFun(self, xt, yt, tt):
		weights = extra.logNormPdf(yt, self.g(xt, yt, tt), self.gv(xt, yt, tt));
		return weights#[:,0]

	def propagate(self, xt, yt, tt, nPart):
		return self.q(xt,yt,tt) + self.qv(xt,yt,tt)*np.random.randn(nPart,self.Xdim);

	def logTransProb(self, xtN, xtO, yt, tt):
		return extra.logNormPdf(xtN, self.q(xtO, yt, tt), self.qv(xtO, yt, tt)) - self.maxPdf

	def __str__(self):
		return "Stochastic volatility model \n X(t+1) = {0} * X(t) + {1} * v(t+1) \n Y(t) = {2} * exp(0.5 * X(t)) * e(t)".format(
				self.par[0], self.par[1], self.par[2])


