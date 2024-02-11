import matplotlib.pyplot as plt
import numpy as np
import pickle
import multiprocessing
from multiprocessing.pool import ThreadPool
import subprocess
import os
import time

class Evolve():

	'''
	Evolve is a an evolutionary algorithm which is designed be be independent from model execution and evaluation to provide maximum flexibility in cost function design. 
	
	Evolve interacts with models/cost functions through the class method RunCellScript(), which must accept command line arguments which correspond to parameters (global or local) which are used to update the model prior to execution. The model/cost function script name is currently hard coded within this class method but should correspond to wherever you built your model and evaluate its performance. Your model file/cost function script should output parameters used and model score to a file named 'parallelresults.txt' using a threadsafe append type of function so that ENBO can safely read them.
	
	
	Evolve expects the following parameters (required):
	
	modelscript: str - pathname of file which runs and evaluates your model
	param_ranges: list - [[min,max,increment],[min,max,increment],...]
	
	But also accepts the following parameters (optional):
	
	threshold_type: str, 'pool','individual' - this parameter designates whether it is pool or individual performances which triggers ENBO termination criteria
	score_threshold: int/float - this parameter should correspond to your desired cost function performance.
	generation_size: int - the number of individuals to be evaluated in parallel in each generation. Should not be greater than the number of processors
	mating_pool_size: int - the size of your mating pool should be adjusted according to the size/complexity of your open parameter space
	generation_limit: int - the number of generations (*generation_size) that will be evaluated before ENBO timeout
	dynamic_mutation: boolean - should mutation rate be ajusted according to mean pool score - mutation rate determined by class method update_mutation_rate() and ranges from 0.5-0.0
	mutation_rate: float - initial mutation rate. Must be a value between 1.0-0.0
	seed_generation: li - List of list of parameters equal in length to mating_pool_size which may be used to initialize Evolve.
	
	Evolve creates and manages two datafiles:
	
	'MatingPool.pickle' - stores the current mating pool
	'best_individual.txt' - following each generation ENBO stores the parameters for the best performing individual in the pool and the mean pool score
	'''
	
	def __init__(self,modelscript=None, param_ranges = [],outputdir=os.getcwd(),threshold_type='pool',score_threshold=1,generation_size=19,mating_pool_size=95,generation_limit=100,dynamic_mutation=True,mutation_rate=0.5,seed_generation=None):
		start_time = time.time()
		self.modelscript = modelscript
		self.outputdir = outputdir
		if self.modelscript is None:
			print('No modelscript provided')
			return()
		
		self.threshold_type = threshold_type
		self.threshold=False
		self.max_pool_size = mating_pool_size
		self.score_threshold=score_threshold
		self.check_set_generation_size(generation_size)
		self.generation_limit = generation_limit
		self.generations = 0
		self.param_ranges = param_ranges
		if self.param_ranges == []:
			return()
		
		self.initialize_mating_pool(seed_generation=seed_generation)
		print('Pool initialized')
		end_time = time.time()
		self.time_it(start_time)
		self.original_pool_mean = np.mean(self.mating_pool_evaluations)
		print('Original pool mean score: ',self.original_pool_mean)
		self.dynamic_mutation = dynamic_mutation
		if self.dynamic_mutation:
			self.mutation_rate = 0.5
		else:
			self.mutation_rate = mutation_rate	
		
		print('Original mutation rate: ',self.mutation_rate)	
		
	def check_set_generation_size(self,gen_size):
		if gen_size > multiprocessing.cpu_count():
			gen_size = multiprocessing.cpu_count()
			print('It is not recommended to have generation size greater than number of processors on machine. Reduced generation size to cpu count -1.')
		
		self.generation_size = gen_size
	
	def initialize_mating_pool(self,seed_generation=None):
		if seed_generation is not None:
			self.mating_pool = seed_generation
		else:
			self.mating_pool = []
			print('Initialized mating pool with '+str(self.max_pool_size)+' random entries')
		
		while len(self.mating_pool) < self.max_pool_size:
			self.mating_pool.append(self.get_random_individual())
		
		
		self.mating_pool_evaluations = []
		for group in self.divide_chunks(self.mating_pool,self.generation_size):
			self.mating_pool_evaluations+=self.evaluate_generation(group)
		
	
	def sort_children_by_varbs(self,children,varbs,spikes):
		newspikes = []
		newvarbs = []
		for child in children:
			which = varbs.index(child)
			newvarbs.append(varbs[which])
			newspikes.append(spikes[which])
		
		return(newvarbs,newspikes)
	
	def update_threshold(self,scores):
		if self.threshold_type == 'pool' and self.mating_pool_evaluations != []:
			if np.mean(self.mating_pool_evaluations) < self.score_threshold:
				self.threshold=True
		
		else:
			for score in scores:
				if score[0] < self.threshold:
					self.threshold = True
	
	def evaluate_generation(self,children):
		open('parallelresults.txt','w').close()
		pool = ThreadPool(self.generation_size)
		pool.map(self.RunCellScript,children)
		varbs,scores = self.load_parallel_results()
		varbs,scores = self.sort_children_by_varbs(children,varbs,scores)
		print(scores)
		self.update_threshold(scores)
		return([score[0] for score in scores])
	
	def divide_chunks(self, l, n):
		for i in range(0,len(l),n):
			yield l[i:i+n]
	
	def check_add_to_mating_pool(self,children,scores):
		for c,child in enumerate(children):
			pool_max = np.max(self.mating_pool_evaluations)
			if scores[c] < pool_max:
				individual_to_remove = self.mating_pool_evaluations.index(pool_max)
				self.mating_pool_evaluations[individual_to_remove] = scores[c]
				self.mating_pool[individual_to_remove] = child

	def RunCellScript(self,params):
		args = 'python3 '+self.modelscript+' '
		for p in params:
			args+=str(p)+' '
		
		subprocess.call(args,shell=True)
	
	def parse_results_line(self,line):
		return([float(item) for item in line.strip('[]\n').split(',')])

	def load_parallel_results(self):
		with open('parallelresults.txt','r') as f:
			results = [self.parse_results_line(item) for item in f.readlines()]
		
		varbs = results[::2]
		sptimes = results[1::2]
		return(varbs,sptimes)
	
	def mating_pool_snapshot(self):
		pool_min = np.min(self.mating_pool_evaluations)
		best_individual_index = self.mating_pool_evaluations.index(pool_min)
		best_score = self.mating_pool_evaluations[best_individual_index]
		best_individual = self.mating_pool[best_individual_index]
		try:
			with open(self.outputdir+'best_individual.txt','a') as f:
				f.write(str(best_score)+'\n')
				f.write(str(best_individual)+'\n')		
				f.write(str(np.mean(self.mating_pool_evaluations))+'\n')
				print(best_individual,'best_individual',best_score,'best_score')
		
		except:
			with open(self.outputdir+'best_individual.txt','w') as f:
				f.write(str(best_score)+'\n')
				f.write(str(best_individual)+'\n')
				f.write(str(np.mean(self.mating_pool_evaluations))+'\n')
		
		with open(self.outputdir+'MatingPool.pickle','wb') as f:
			pickle.dump([self.mating_pool,self.mating_pool_evaluations],f)
	
	def get_random_individual(self):
		A = []
		for pr in self.param_ranges:
			A.append(np.random.choice(np.arange(pr[0],pr[1],pr[2])))
		
		return(A)
	
	def get_new_generation(self):
		children = []
		while len(children) < self.generation_size:
			mom,dad = [self.mating_pool[i] for i in np.random.choice(range(self.max_pool_size),2)]
			children+=self.crossover(mom,dad)
		
		return([self.mutate(child) for child in children[:self.generation_size]])
	
	def crossover(self,varsA,varsB):
		crossover_point = np.random.choice(range(len(varsA)))
		if np.random.random() > 0.5:
			newA = varsA[:crossover_point]+varsB[crossover_point:]
			newB = varsB[:crossover_point]+varsA[crossover_point:]
		
		else:
			newB = varsA[:crossover_point]+varsB[crossover_point:]
			newA = varsB[:crossover_point]+varsA[crossover_point:]
		
		newC = [(varsA[i]+varsB[i])/2.0 for i in range(len(varsA))]
		return(newA,newB,newC)
	
	def mutate(self,varbs):
		for v,var in enumerate(varbs):
			if np.random.random() < self.mutation_rate:
				varbs[v]=np.random.uniform(self.param_ranges[v][0],self.param_ranges[v][1])
				if varbs[v] < self.param_ranges[v][0]:
					varbs[v] = self.param_ranges[v][0]
				
				if varbs[v] > self.param_ranges[v][1]:
					varbs[v] = self.param_ranges[v][1]
		
		return(varbs)
	
	def update_mutation_rate(self):
		pool_mean = np.mean(self.mating_pool_evaluations)
		print('New pool mean score: ',pool_mean)
		self.mutation_rate = 0.5/(1+np.exp(0.5+(-1*pool_mean)/25))		#0.5/(1+np.exp(1+(-1*pool_mean)/25))
		print('New mutation rate: ',self.mutation_rate)
	
	def time_it(self,s):
		e = time.time()
		print('elapsed time: '+str(round((e-s)/60.0,1))+' minutes')
	
	def run(self):
		start_time = time.time()
		self.threshold=False
		while not self.threshold and self.generations < self.generation_limit:
			self.mating_pool_snapshot()
			children = self.get_new_generation()
			scores = self.evaluate_generation(children)
			self.check_add_to_mating_pool(children,scores)
			if self.dynamic_mutation:
				self.update_mutation_rate()
			
			self.time_it(start_time)
			self.generations+=1
		
		self.mating_pool_snapshot()


if __name__ == "__main__":
	params = [
					[0,1,0.01], #interbouton gkbar_axnodeX 'k_ib'
					[0,1,0.01], #bouton gkbar_axnodeX 'k_b'
					[0.5,7.0,0.1], #interbouton gnabar_axnodeX 'na_ib'
					[0.5,7.0,0.1], #bouton gnabar_axnodeX 'na_b'
					[0,0.1,0.01], #interbouton gnapbar_axnodeX 'nap_ib'
					[0,0.1,0.01], #bouton gnapbar_axnodeX 'nap_b'
					[0,0.1,0.001],#interbouton gl_axnodeX 'gl_ib'
					[0,0.1,0.001],#bouton gl_axnodeX 'gl_b'
					[1.88*0.5,1.88*1.5,1.88/100.0], # 'amA'
					[21.4*0.5,21.4*1.5,21.4/100.0],# 'amB'
					[10.3*0.5,10.3*1.5,10.3/100.0],# 'amC'
					[0.086*0.5,0.086*1.5,0.086/100.0],# 'bmA'
					[25.7*0.5,25.7*1.5,25.7/100.0],# 'bmB'
					[9.16*0.5,9.16*1.5,9.16/100.0],# 'bmC'
					[0.062*0.5,0.062*1.5,0.062/100.0],# 'ahA'
					[114.0*0.5,114.0*1.5,114.0/100.0],# 'ahB'
					[11.0*0.5,11.0*1.5,11.0/100.0],# 'ahC'
					[0.3*0.5,0.3*1.5,0.3/100.0],# 'asA'
					[-27*1.5,-27*0.5,27/100.0],# 'asB'
					[-5*1.5,-5*0.5,5/100.0],# 'asC'
					[1000.0*0.25,1000.0*1.5,1000.0/100.0],# 'gbar_ib'
					[1000.0*0.25,1000.0*1.5,1000.0/100.0],# 'gbar_b'
					[0,1,0.01], #node gkbar_axnode 'k_nb'
					[0.5,7.0,0.1], #node gnabar_axnode 'na_nb'
					[0,0.1,0.01], #node gnapbar_axnode 'gnap_nb'
					[0,0.1,0.001],#node gl_axnode 'gl_nb'
					[1.88*0.5,1.88*1.5,1.88/100.0], # 'amA_n'
					[21.4*0.5,21.4*1.5,21.4/100.0], # 'amB_n'
					[10.3*0.5,10.3*1.5,10.3/100.0], # 'amC_n'
					[0.086*0.5,0.086*1.5,0.086/100.0], # 'bmA_n'
					[25.7*0.5,25.7*1.5,25.7/100.0], # 'bmB_n'
					[9.16*0.5,9.16*1.5,9.16/100.0], # 'bmC_n'
					[0.062*0.5,0.062*1.5,0.062/100.0], # 'ahA_n'
					[114.0*0.5,114.0*1.5,114.0/100.0], # 'ahB_n'
					[11.0*0.5,11.0*1.5,11.0/100.0], # 'ahC_n'
					[0.3*0.5,0.3*1.5,0.3/100.0], # 'asA_n'
					[-27*1.5,-27*0.5,27/100.0], # 'asB_n'
					[-5*1.5,-5*0.5,5/100.0], # 'asC_n'
			]
	
				
	use_seed = False
	diameter = 1.5
	targetdir = os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/'
	if use_seed:
		with open('MatingPool_reserve.pickle','rb') as f:
			seed = pickle.load(f)
		evol = Evolve(modelscript='RunBiophysics.py',param_ranges=params,outputdir=targetdir,threshold_type='pool',score_threshold=1,generation_limit=4000,seed_generation=seed[0])
	else:
		evol = Evolve(modelscript='RunBiophysics.py',param_ranges=params,outputdir=targetdir,threshold_type='pool',score_threshold=1,generation_limit=4000)
	evol.run()


