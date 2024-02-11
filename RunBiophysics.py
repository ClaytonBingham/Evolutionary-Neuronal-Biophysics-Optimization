from CellManager import CellManager
from neuron import h
import os, sys
from multiprocessing import Process
from time import sleep
import numpy as np
import time
from scipy.spatial.distance import cdist
import pandas as pd
import random
import matplotlib.pyplot as plt
from CostFunctions import ConductionVelocity
import pickle

def ez_record(h,var='v',sections=None,order=None,targ_names=None,cust_labels=None,level='section'):
	"""
	Records state variables across segments

	Args:
		h = hocObject to interface with neuron
		var = string specifying state variable to be recorded.
		      Possible values are:
		          'v' (membrane potential)
		          'cai' (Ca concentration)
		sections = list of h.Section() objects to be recorded
		targ_names = list of section names to be recorded; alternative
		             passing list of h.Section() objects directly
		             through the "sections" argument above.
		cust_labels =  list of custom section labels
		level = 'section' or 'segment', determines if one or many positions are recorded for each section
	
	Returns:
		data = list of h.Vector() objects recording membrane potential
		labels = list of labels for each voltage trace
	"""
	data, labels = [], []
	for i in range(len(sections)):
		sec = sections[i]
		if level=='section':
			# record data
			data.append(h.Vector())
			if var == 'v':
				data[-1].record(sec(0.5)._ref_v)
			elif var == 'cai':
				data[-1].record(sec(0.5)._ref_cai)
			# determine labels
			if cust_labels is None:
				lab = sec.name()+'_'+str(round(0.5,5))
			else: 
				lab = cust_labels[i]+'_'+str(round(0.5,5))
			labels.append(lab)
		else:
			positions = np.linspace(0,1,sec.nseg+2)
			for position in positions[1:-1]:
				# record data
				data.append(h.Vector())
				if var == 'v':
					data[-1].record(sec(position)._ref_v)
				elif var == 'cai':
					data[-1].record(sec(position)._ref_cai)
				# determine labels
				if cust_labels is None:
					lab = sec.name()+'_'+str(round(position,5))
				else: 
					lab = cust_labels[i]+'_'+str(round(position,5))
				labels.append(lab)
	return (data, labels)

def ez_convert(data):
	"""
	Takes data, a list of h.Vector() objects filled with data, and converts
	it into a 2d numpy array, data_clean. This should be used together with
	the ez_record command. 
	"""
	data_clean = np.empty((len(data[0]),len(data)))
	for (i,vec) in enumerate(data):
		data_clean[:,i] = vec.to_python()
	return data_clean

def stimulator(sec,amp):
	stim = h.IClamp(sec(0.5))
	vec = h.Vector()
	netcon = h.NetCon(sec(0.5)._ref_v, None, sec=sec)
	netcon.record(vec)
	stim.dur = 0.5
	stim.amp = amp
	stim.delay = 4
	return(stim,vec)

def output_sptime_params(params,score):
	with open('parallelresults.txt','a') as f:
		f.write(str(params)+'\n'+str(score)+'\n')

def plot_v_traces(clean_data,timestop,labels):
	import matplotlib.pyplot as plt
	fig = plt.figure()
	plt.title('r-ais/hil/nd; g-pnd; k-internd; c-bt; m-interbt')
	colors = dict(zip(['soma','hillock','iseg','apical','basal','node','paranode1','paranode2','internode','bouton','interbouton','tuft'],['y','r','r','y','y','r','g','g','k','c','m','y']))
	for i in range(0,len(clean_data[0]),1)[::-1]:
		if labels[i] != 'node':
			continue
		plt.plot(np.arange(0,timestop+h.dt,h.dt),[item[i] for item in clean_data],color=colors[labels[i]],label='Section '+str(i))
		plt.xlabel('ms')
		plt.ylabel('mV')
		plt.pause(0.005)
	
	plt.show()

def run_cell_modifybiophys(stimamp_scalar,fnames,plot=False,biophysics={},CCFdiam=10.0):
	h.load_file("stdrun.hoc") #this loads standard simulation initialized variables
	stimamp = list(np.array([-100.0,20,0])*stimamp_scalar)
	stimtimevec = [4.0,4.125,4.750]
	h.tstop = 35.0
	h_stimtimevec = h.Vector(stimtimevec)
	h.celsius = 37
	h.v_init = -80
	h.finitialize(h.v_init)
	h.stdinit()
	h.steps_per_ms = 64
	h.dt = 0.015625
	single_axon = CellManager(fnames[0],fnames[1],fnames[2],CCF_outer=CCFdiam)
	if biophysics is not {}:
		single_axon.modify_biophysics(biophysics)
	data,labels = ez_record(h,var='v',sections=list(single_axon.node+single_axon.paranode1+single_axon.paranode2+single_axon.internode+single_axon.bouton+single_axon.interbouton))
#	stim = stimulator(single_axon.bouton[int(len(single_axon.bouton)/2)],stimamp_scalar)
	stim = stimulator(single_axon.node[0],stimamp_scalar)
	h.run()
	clean_data = ez_convert(data)
	if plot:
		sectypes = ['node' for i in range(len(single_axon.node))]+['paranode1' for i in range(len(single_axon.paranode1))]+['paranode2' for i in range(len(single_axon.paranode2))]+['internode' for i in range(len(single_axon.internode))]+['bouton' for i in range(len(single_axon.bouton))]+['interbouton' for i in range(len(single_axon.interbouton))]
		plot_v_traces(clean_data,h.tstop,sectypes)
	
	return(single_axon,clean_data,stim)

def get_biophys(fname_inc=0,diameter=6.0):
	biophys = dict(zip(['k_ib','k_b','na_ib','na_b','nap_ib','nap_b','gl_ib','gl_b','amA','amB','amC','bmA','bmB','bmC','ahA','ahB','ahC','asA','asB','asC','gbar_ib','gbar_b','k_nb','na_nb','gnap_nb','gl_nb','amA_n','amB_n','amC_n','bmA_n','bmB_n','bmC_n','ahA_n','ahB_n','ahC_n','asA_n','asB_n','asC_n'],[float(arg) for arg in sys.argv[1:]]))
	cell_fnames = [os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/dlPFC_cp.0.py',os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/dlPFC_cp.0_section_labels_'+str(diameter)+'_.csv',os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/dlPFC_cp.0_section_lengths_'+str(diameter)+'_.csv']
	
	if biophys == {}:
#		params = [0.3786779680372039, 0.2454379788422686, 2.0120317781791606, 3.8009554823648424, 0.04577337565270471, 0.09969086819711157, 0.04971760828179261, 0.06510062712243338, 1.5311019929679026, 27.51194796693157, 9.605554796854745, 0.10807442500405887, 24.17245394548994, 8.847221189599594, 0.0661187318506897, 110.21173782937765, 10.334434840341622, 0.4055599479217065, -24.546247782860583, -2.8230041360491365, 604.1707023838201, 1351.7658943845518, 0.5460183943999131, 5.358739296362057, 0.05789512211187473, 0.018739897103155333, 2.7415979962984522, 27.528843928578393, 8.849878685384171, 0.11439275787312242, 33.621849572473494, 9.76169520640639, 0.07771360324098142, 102.05043126176139, 10.701961931582668, 0.2626151680153638, -31.155234829456653, -4.310180043126254]
#		params = [0.20899129699422536, 0.16595831336859967, 4.249999999999999, 5.43060098311585, 0.049750222507862256, 0.07003869152098229, 0.0528146892130205, 0.077936879084383, 1.9736787150428527, 27.05195586847118, 7.621631104057206, 0.08908132990633746, 22.656400891840267, 13.20770235675871, 0.07356567458801208, 135.94500000000005, 12.972931936260858, 0.4301675102590633, -21.7173520505139, -6.281943612512677, 966.9925133242368, 338.62981673047375, 0.24039673074682139, 5.399999999999999, 0.09850273163189396, 0.028006895848316574, 2.190865968360954, 19.871000057792305, 9.96524999999999, 0.0774488649243355, 22.80874999999999, 8.573354433686063, 0.08651093689055933, 90.06342823767467, 11.299318569539672, 0.30736540289357756, -32.07873034685426, -3.564057961818525]
		with open(os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/'+'MatingPool.pickle','rb') as f:
			mpool = pickle.load(f)
		cellnum = 0
		params = mpool[0][cellnum]
		print(cellnum)
		#
		
		biophys = dict(zip(['k_ib','k_b','na_ib','na_b','nap_ib','nap_b','gl_ib','gl_b','amA','amB','amC','bmA','bmB','bmC','ahA','ahB','ahC','asA','asB','asC','gbar_ib','gbar_b','k_nb','na_nb','gnap_nb','gl_nb','amA_n','amB_n','amC_n','bmA_n','bmB_n','bmC_n','ahA_n','ahB_n','ahC_n','asA_n','asB_n','asC_n'],params))
		cell_fnames = [os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/'+fname for fname in os.listdir(os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/') if '.py' in fname and 'swc2py' not in fname]
		cell_fnames = [cell_fnames[fname_inc]]
		cell_fnames.append(cell_fnames[0][:-3]+'_section_labels_'+str(diameter)+'_.csv')
		cell_fnames.append(cell_fnames[0][:-3]+'_section_lengths_'+str(diameter)+'_.csv')
	return(cell_fnames,biophys)

if __name__ == "__main__":
	#10.75: 

	diameter = 1.5
	cell_fnames,biophys = get_biophys(0,diameter)
#	cell_number = cell_fnames[0].split('/')[-1].split('_')[2]
	print('biophysics '+str(50),cell_fnames[0],0)
	cell,on_data,stim = run_cell_modifybiophys(2.0,[cell_fnames[0],cell_fnames[1],cell_fnames[2]],plot=True,biophysics=biophys,CCFdiam=diameter)
	on_data = np.transpose(on_data)
	
	###################################
	#Conduction Velocity Cost Function#
	###################################
	cv=ConductionVelocity()
	myelinatedCV = cv.calculate_CVs(cell,on_data,h.dt)
	cost = cv.calculate_cost(myelinatedCV,8.5,h.dt,4.0625,on_data)
	output_sptime_params(list(biophys.values()),cost)


