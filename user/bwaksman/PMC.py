# Ben Waksman
#
# Implementation of Population Monte Carlo (Adaptive Importance Sampling)
# For RVChallengeYale

# Import all relevant packages 
import sys
import csv
import numpy as np
from scipy.stats import multivariate_normal
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.signal import lombscargle
from scipy.optimize import newton
import scipy.stats as stats


np.set_printoptions(threshold='nan')

# mixture model of multivariate normal distributions given by a list of
# means and covariance matrices, and weighted by a given list of weights
class model:

	def __init__(self, weights, mus, covars):

		# normalize the weights so sum is 1
		self.weights = np.array(weights, dtype='f') / np.sum(weights)

		self.mus = np.array(mus, dtype='f')
		self.covars = np.array(covars, dtype='f')

		# sanity check for matching sizes of input data
		assert (len(mus) == len(weights) == len(covars))
		self.n = len(mus)

		# used to choose which multivariate normal distribution to sample from
		self.bins = np.cumsum(self.weights)


	# update the distributions in the mixture
	def update(self, weights, mus, covars):
		self.weights = np.array(weights, dtype='f') / np.sum(weights)
		self.mus = np.array(mus, dtype='f')
		self.covars = np.array(covars, dtype='f')

		assert (len(mus) == len(weights) == len(covars))
		self.n = len(mus)

		self.bins = np.cumsum(weights)


	# return a random sample of size npop
	# apply the function fiddle to the data before returning
	def sample(self, npop):

		# Simulate drawing from a true mixture model by drawing a number
		# uniformly at random from [0,1) and then determining which 
		# multivariate normal distribution to draw from, where the chance of
		# drawing from a distriubtion is determined by its relative weight

		# generate uniform random variables
		rand = np.random.random_sample(npop)
		# determine which of the mixed distributions to draw from
		use = np.digitize(rand, self.bins)

	 	xx = list(map((lambda i: np.random.multivariate_normal(self.mus[i], self.covars[i])), use))

	 	return xx

	# return the PDF of each element of xx
	def getPPS(self, xx):

		# To calculate the PDF of a value in the mixed model, calculate its PDF
		# for each individual multivariate normal distribution and take the 
		# weighted sum

	 	pps = map((lambda i: multivariate_normal.pdf(xx, self.mus[i], self.covars[i],allow_singular=True) * self.weights[i]), range(0, self.n))

	 	return pps

	# return the current set of means
	def getMus(self):
		return self.mus

	# return the current set of weights
	def getWts(self):
		return self.weights

	# return the current set of covariance matrices
	def getCovars(self):
		return self.covars

	# return the current number of models in the mixture
	def getN(self):
		return self.n


# Implement Population Monte Carlo (adaptive importance sampling)
# as described in Wraith et al. (2009)
def iterPMC(npop, model, eps):

	perplex = 0.0 				# normalized perplexity of the model
	reg = 1.0 					# regulation term for widening cov matrix at early steps

	# Loop until perplexity approaches 1
	while perplex < .8:

		# Take npop samples from the mixture model
		xx = model.sample(npop)
		# Compute PDF for mixture
		pps = model.getPPS(xx)

		# calculate fit of each random sample to the data
		lik = map(chi2, map(fiddle, xx))
		lik = lik - min(lik)
		lik = np.exp(-lik/2)

		totpp = np.sum(pps, 0)

		for i in range(len(totpp)):
			if totpp[i] == 0.0:
				print xx[i]

		rhos = map((lambda y: y/totpp), pps)

		# Calculate importance weigts
		wts = lik / totpp
		wts = wts / np.sum(wts)

		# Calculate perplexity at current step
		posWts = [wt for wt in wts if wt > 0]
		perplex = np.exp(-np.sum(posWts * np.log(posWts))) / npop

		print perplex

		# Update model terms
		a = np.array(map((lambda rho: np.sum(wts*rho)), rhos))
		# remove models that are assigned 0 weight
		temp  = a > 0.0
		a = a[temp]

		rhos = np.array(rhos)[temp]
		
		mu = map((lambda d: np.array(np.matrix(wts*rhos[d])*xx)[0] / a[d]), range(0, len(a)))
		mu = np.array(map(fiddle, mu))
		xx = np.array(map(fiddle, xx))
		dx = map((lambda u: xx - u), mu)
		sig = map((lambda d: np.sum(map((lambda n: np.dot(np.matrix(dx[d][n]).T, np.matrix(dx[d][n])) * wts[n] * rhos[d][n]), range(0, npop)), 0) / a[d] + reg * eps), range(0,len(a)))

		# ensure symmetry in each covariance matrix
		sig = map((lambda s: (np.array(s) + np.array(s).T) / 2), sig)

		reg = reg / 1.2

		for i in range(len(sig)):
			if np.linalg.det(sig[i]) == 0.0:
				print sig[i]
				sig[i] += eps


		# print current model info
		model.update(a, mu, sig)
		print mu
		if len(mu) == 1:
			print sig
		print a



	return (perplex, xx, model)


# Group together parameters for different planets to apply predictFull()
def chi2(x, returnFit=True):
	phis = []
	Ps = []
	logKs = []
	es = []
	omegas = []
	for i in range((len(x)-2)/5):
		phis.append(x[2 + i * 5])
		Ps.append(x[3 + i * 5])
		logKs.append(x[4 + i * 5])
		es.append(x[5 + i * 5])
		omegas.append(x[6 + i * 5])

	return predictFull(x[0], x[1], phis, Ps, logKs, es, omegas, returnFit)


# Adjust the data so that it stays within the allowable ranges
def fiddle(x):

	for i in range((len(x)-2)/5):

		# wrap phi into range [0, 1)
		x[2 + i * 5] = np.mod(x[2 + i * 5], 1)
	
		# cutoff e in range [0,1)
		x[5 + i * 5] = max(0.0, min(.9999,x[5 + i * 5]))
	
		# wrap omega into range [0, 2*pi)
		x[6 + i * 5] = np.mod(x[6 + i * 5], 2 * np.pi)

	return x


# Calculate predicted RV signal for the general case of eccentric orbits
# given the parameter values. Return either the fit of the predicted data
# or the predicted data itself depending on the value of returnFit
# phi in range [0,1] and omega in radians
def predictFull(linear, fwhm, phi, P, logK, e, omega, returnFit=True):
	global data

	rv = [0]*len(data['jdb'])

	for j in range(len(phi)):


		M = 2.0 * np.pi * (np.array([x / P[j] for x in data['jdb']]) + phi[j])

		# Solve Kepler's equation using Newton-Raphson
		func = (lambda E, m: E - e[j] * np.sin(E) - m)
		E = map((lambda m: newton(func, m, args=(m,), maxiter=200)), M) 

		# calculate the true anomaly
		f = np.arccos((np.cos(E) - e[j])/(1-e[j]*np.cos(E)))

		# correct for fact that f is [0, 2pi] but arccos(z) is [0, pi]
		for i in range(len(f)):
			if np.modf(data['jdb'][i]/P[j] + phi[j])[0] > 0.5:
				f[i] = 2 * np.pi - f[i]

		rv += np.exp(logK[j]) * (np.cos(omega[j] + f) + e[j] * np.cos(omega[j]))

	rv += fwhm*np.array(data['fwhm']) + linear
		
	if returnFit:
		return fit(rv, data)
	return rv

# Calculate & return the fit of the prediction to the actual data
def fit(prediction, data):

	temp = map((lambda x: x**2), data['rv'] - prediction)

	chi = []
	for i in range(len(data['sig2'])):
		chi.append(temp[i] / (data['sig2'][i] + data['activity_noise']))

	return sum(chi)

# Carry out the necessary initialization steps to run the PMC algorithm
# and then call the algorithm with the initialized mixture model
# and print out summary results
# If a partial_model is given, then use that to initialize PMC and add
#  parameters for another planet
def runPMC(data, peaks, partial_model=None):

	npop = 1000		# number of samples at each iteration of PMC

	wts = []
	mus = []
	covars = []
	n = 0
	old_covars = np.array([[0]*5]*5)

	# If a partial_model is given, then use the old covariance matrix
	#  to generate a new one for a larger model
	if partial_model != None:
		old_mus = partial_model.getMus()[0]
		n = len(old_mus)
		partial_covars = partial_model.getCovars()[0]
		
		old_covars = np.zeros((n+5),(n+5))
		for i in range(len(partial_covars)):
			for j in range(len(partial_covars[i])):
				old_covars[i][j] = partial_covars[i][j]



	# Initial guesses at values, periods taken from periodogram
	phis = [1.0/3.0, 2.0/3.0]
	logks = [-1/3.0, 2/3.0]
	es = [1.0/3.0, 2.0/3.0]
	omegas = [0, np.pi]

	# Add guassians at each combination of parameter values
	if partial_model == None:

		# Make the initial guess for the linear and FWHM terms 
		#  from the OLS best fit
		coef = np.polyfit(data['fwhm'], data['rv'], 1)
		
		for i in range(len(peaks)):
			for phi in phis:
				for logk in logks:
					for e in es:
						for omega in omegas:
							wts.append(1)
							mus.append([coef[1], coef[0], phi, peaks[i], logk, e, omega])
							covars.append(np.diag([.000001, 0.00001, .5,0.1, .2, 0.2,1.0]))

		# used to increase variance of each parameter value in early iterations of PMC
		eps = np.diag([0.000001,0.000001, 0.0001, 0.000001, 0.0001, 0.000001, 0.000001])

		
	# If partial_model given, then augment with additional parameters
	else:
		for i in range(len(peaks)):
			for phi in phis:
				for logk in logks:
					for e in es:
						for omega in omegas:
							wts.append(1)
							mus.append(np.append(old_mus, [phi, peaks[i], logk, e, omega]))
							covars.append(old_covars + np.diag([0]*n + [.5,0.1, .2, 0.2,1.0]))

		# used to increase variance of each parameter value in early iterations of PMC
		eps = np.diag([0.000001,0.01] + [0.0001, 0.000001, 0.0001, 0.000001, 0.000001]*((n-2)/5+1))	

		

	mod = model(wts, mus, covars)

	
	# examine the initial distribution of each variable
	xx = map(fiddle, mod.sample(npop))
	makePlots(xx)

	print mod.getMus()
	print mod.getWts()
	print mod.getCovars()


	[perplexity, xx, mod] = iterPMC(npop, mod, eps)

	makePlots(xx)

	print perplexity
	print mod.getMus()
	print mod.getWts()
	print mod.getCovars()

	return mod


# Make histograms of each of the variables and a scatterplot of period vs phase
def makePlots(xx):

	xlabs = ('Phase', 'Period (days)', 'K (m/s)')
	
	plt.hist([np.mod(x[2],1) for x in xx], bins=20)
	plt.xlabel('Phase')
	plt.ylabel('N')
	plt.show()

	plt.hist([x[3] for x in xx], bins=20)
	plt.xlabel('Period (days)')
	plt.ylabel('N')
	plt.show()

	plt.hist([np.exp(x[4]) for x in xx], bins=20)
	plt.xlabel(xlabs[2])
	plt.ylabel('N')
	plt.show()

	plt.plot([np.mod(x[2],1) for x in xx],[x[3] for x in xx], '.')
	plt.xlabel('Phase')
	plt.ylabel('Period')
	plt.show()


# Return a list of periods for which the periodogram of the rv data
# has a peak significant beyond the given fap (false alarm probability)
# 
# The threshold for false alarms is determined by permuting the data over
# the time domain, generating periodograms and building a CDF for the 
# highest power peak
def findPeaks(times, rv, fap, oversample):

	ntrials = 200

	min_int = float("inf")
	for i in range(len(times)-1):
		min_int = min(min_int, times[i+1] - times[i])

	# Identify frequencies at which to evaluate the periodogram
	n0 = len(times)
	T = max(times)-min(times)
	k = np.linspace(1.0/oversample, T / 1.5, T/1.5 *oversample)
	omegas = 2 * np.pi * k / T

	# For each trial, randomly permute the data and generate a periodogram
	maxes = []

	for i in range(ntrials):
		noise = np.random.permutation(rv)
		psd = lombscargle(np.array(times), noise, omegas)
		maxes.append(max(psd))

	z = range(int(max(maxes)*100 + 1))

	# Calculate CDF for each z value
	cdf = []
	for i in z:
		cdf.append(len([x for x in maxes if x > i/100.0])/float(ntrials))

	#plot the CDF: P(max_peak > Z)
	#plt.plot(np.array(z)/100.0, cdf)
	#plt.show()

	# Find the threshold frequency associated the given FAP
	for i in range(len(cdf)):
		if cdf[i] > fap:
			idx = i
	thres = z[idx+1]/100.0

	psd = lombscargle(np.array(times), np.array(rv), omegas)

	plt.plot(1/omegas * 2 * np.pi, psd)
	plt.axhline(thres, color='r')
	plt.xlabel('Period (days)')
	plt.ylabel('Power')
	plt.show()

	# find the periods for which the data has significant peaks
	peaks = [1/omegas[x] * 2 * np.pi for x in range(len(psd)) if psd[x] > thres]

	print len(peaks)

	return peaks


def main():

	# Read in data from file given in command line
	# Extract time data 'jdb', combined 'rv' data, and variance
	global data
	data = defaultdict(lambda:[])
	try:
		# Open .rdb file, break into table using first line as headers
		with open(sys.argv[1], 'rb') as csvfile:
			reader = csv.DictReader(csvfile, delimiter='\t')
			# skip the second line (all dashes)
			next(reader, None)

			for row in reader:

				data['jdb'].append(float(row['jdb']))
				
				try:
					data['rv'].append((float(row['vrad']))*1000)
					data['sig2'].append((float(row['svrad'])*1000)**2)
					data['rhk'].append(float(row['rhk'])*1000)

				# Catch different column headers in training data
				except KeyError:
					data['rv'].append((float(row['rv_planet'])  + float(row['rv_activity'])  + float(row['rv_osc_and_gran']) + float(row['rv_inst_noise']))*1000)
					data['sig2'].append((float(row['sig_rv'])*1000)**2)

				data['fwhm'].append(float(row['fwhm'])*1000)
				

				
	except IndexError:
		print "Expected input file in command line args"
		exit()


	print len(data['rv'])

	data['jdb_full'] = data['jdb']
	data['rv_full'] = data['rv']
	data['fwhm_full'] = data['fwhm']
	
	# Look at only those measurements for which FWHM < 10
	# --an attempt to remove effects of stellar activity beyond decorrelating FWHM--
	idx = [i for i in range(len(data['fwhm'])) if data['fwhm'][i] < 10]
	data['fwhm'] = [data['fwhm'][x] for x in idx]
	data['rv'] = [data['rv'][x] for x in idx]
	data['jdb'] = [data['jdb'][x] for x in idx]
	data['sig2'] = [data['sig2'][x] for x in idx]


	
	# Plot RV vs Time
	plt.plot(data['jdb'], data['rv'], '.')
	plt.xlabel('Time')
	plt.ylabel('Radial Velocity (m/s)')
	plt.show()


	# Fit FWHM to RV signal
	coef = np.polyfit(data['fwhm'], data['rv'], 1)	
	print coef

	# Plot FWHM vs RV with best fit shown
	plt.plot(data['fwhm'], data['rv'], '.')
	plt.plot(data['fwhm'], coef[1] + np.array(data['fwhm'])*coef[0], color='r')
	plt.xlabel("FWHM")
	plt.ylabel('Radial Velocity')
	plt.show()

	print np.corrcoef(data['fwhm'], data['rv'])

	# Examine normality of residuals
	plt.hist(data['rv'] - coef[1] - np.array(data['fwhm'])*coef[0], bins=30)
	plt.xlabel('(Adjusted) Radial Velocity (m/s)')
	plt.ylabel('Count')
	plt.show()
	print stats.shapiro(data['rv'] - coef[1] - np.array(data['fwhm'])*coef[0])
	stats.probplot(data['rv'] - coef[1] - np.array(data['fwhm'])*coef[0], plot=plt)
	plt.show()


	plt.plot(data['jdb'], data['rv'] - coef[1] - np.array(data['fwhm'])*coef[0],'.')
	plt.show()


	# Calculate extra noise term to include in chi-square computation
	data['activity_noise'] = np.std(data['rv'] - coef[1] - np.array(data['fwhm'])*coef[0])**2


	# Identify peaks with significant power
	peaks = findPeaks(data['jdb'],data['rv'] - coef[1] - np.array(data['fwhm'])*coef[0] , .01, 1)
	print peaks

	# run PMC for a 1planet case
	mod = runPMC(data, peaks)


	# iterate increaseing number of planets
	# Subtract the previous model, take the periodogram, then run PMC with an additional planet
	#  using the previous model while intializing
	for i in range(2,6):
		print i
		peaks = findPeaks(data['jdb'], data['rv'] - chi2(mod.getMus()[0], False), 0.01, 1)
		print peaks
		mod = runPMC(data, peaks, mod)




# Call main() when program is called from terminal
if __name__ == "__main__":
    main()




