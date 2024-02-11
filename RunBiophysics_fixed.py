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

def get_fnames(fname_inc=0,diameter=6.0):
	cell_fnames = [os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/'+fname for fname in os.listdir(os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/') if '.py' in fname and 'swc2py' not in fname]
	cell_fnames = [cell_fnames[fname_inc]]
	cell_fnames.append(cell_fnames[0][:-3]+'_section_labels_'+str(diameter)+'_.csv')
	cell_fnames.append(cell_fnames[0][:-3]+'_section_lengths_'+str(diameter)+'_.csv')
	return(cell_fnames)

def get_biophys(fname_inc=0,diameter=6.0):
	biophys = dict(zip(['k_ib','k_b','na_ib','na_b','nap_ib','nap_b','gl_ib','gl_b','amA','amB','amC','bmA','bmB','bmC','ahA','ahB','ahC','asA','asB','asC','gbar_ib','gbar_b','k_nb','na_nb','gnap_nb','gl_nb','amA_n','amB_n','amC_n','bmA_n','bmB_n','bmC_n','ahA_n','ahB_n','ahC_n','asA_n','asB_n','asC_n'],[float(arg) for arg in sys.argv[1:]]))
	cell_fnames = [os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/dlPFC_cp.0.py',os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/dlPFC_cp.0_section_labels_'+str(diameter)+'_.csv',os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/dlPFC_cp.0_section_lengths_'+str(diameter)+'_.csv']
	
	if biophys == {}:
#		params = [0.3786779680372039, 0.2454379788422686, 2.0120317781791606, 3.8009554823648424, 0.04577337565270471, 0.09969086819711157, 0.04971760828179261, 0.06510062712243338, 1.5311019929679026, 27.51194796693157, 9.605554796854745, 0.10807442500405887, 24.17245394548994, 8.847221189599594, 0.0661187318506897, 110.21173782937765, 10.334434840341622, 0.4055599479217065, -24.546247782860583, -2.8230041360491365, 604.1707023838201, 1351.7658943845518, 0.5460183943999131, 5.358739296362057, 0.05789512211187473, 0.018739897103155333, 2.7415979962984522, 27.528843928578393, 8.849878685384171, 0.11439275787312242, 33.621849572473494, 9.76169520640639, 0.07771360324098142, 102.05043126176139, 10.701961931582668, 0.2626151680153638, -31.155234829456653, -4.310180043126254]
#		params = [0.20899129699422536, 0.16595831336859967, 4.249999999999999, 5.43060098311585, 0.049750222507862256, 0.07003869152098229, 0.0528146892130205, 0.077936879084383, 1.9736787150428527, 27.05195586847118, 7.621631104057206, 0.08908132990633746, 22.656400891840267, 13.20770235675871, 0.07356567458801208, 135.94500000000005, 12.972931936260858, 0.4301675102590633, -21.7173520505139, -6.281943612512677, 966.9925133242368, 338.62981673047375, 0.24039673074682139, 5.399999999999999, 0.09850273163189396, 0.028006895848316574, 2.190865968360954, 19.871000057792305, 9.96524999999999, 0.0774488649243355, 22.80874999999999, 8.573354433686063, 0.08651093689055933, 90.06342823767467, 11.299318569539672, 0.30736540289357756, -32.07873034685426, -3.564057961818525]
		with open(os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/'+'MatingPool.pickle','rb') as f:
			mpool = pickle.load(f)
#		cellnum = 0
		cells = [93, 28, 49, 28, 38, 88, 27, 16]
		cellnum = cells[[1.5,2.0,3.0,4.0,6.0,9.0,10.25,10.75].index(diameter)]
		params = mpool[0][cellnum]
		print(cellnum)
		#
		
		biophys = dict(zip(['k_ib','k_b','na_ib','na_b','nap_ib','nap_b','gl_ib','gl_b','amA','amB','amC','bmA','bmB','bmC','ahA','ahB','ahC','asA','asB','asC','gbar_ib','gbar_b','k_nb','na_nb','gnap_nb','gl_nb','amA_n','amB_n','amC_n','bmA_n','bmB_n','bmC_n','ahA_n','ahB_n','ahC_n','asA_n','asB_n','asC_n'],params))
		cell_fnames = [os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/'+fname for fname in os.listdir(os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/') if '.py' in fname and 'swc2py' not in fname]
		cell_fnames = [cell_fnames[fname_inc]]
		cell_fnames.append(cell_fnames[0][:-3]+'_section_labels_'+str(diameter)+'_.csv')
		cell_fnames.append(cell_fnames[0][:-3]+'_section_lengths_'+str(diameter)+'_.csv')
	return(cell_fnames,biophys)

def polyfit(x,y,degree,weights=None):
	results = {}
	coeffs = np.polyfit(x,y,degree,w=weights)
	results['polynomial'] = coeffs.tolist()
	correlation = np.corrcoef(x, y)[0,1]
	results['correlation'] = correlation
	 # r-squared
	results['determination'] = correlation**2
	return(results)

def find_polyfit_degree(x,y):
	r2 = []
	for i in range(5):
		results = polyfit(x,y,i)
		r2.append(results['determination'])
	
	return(r2.index(np.max(r2)))

def get_polynomial_biophys(diameter):
	all_bphys = []
	diams = [1.5,2.0,3.0,4.0,6.0,9.0,10.25,10.75]
	for diam in diams:
		fs,bphys = get_biophys(0,diam)
		all_bphys.append(list(bphys.values()))
	
	newvals = []
	coefficients = []
	for i in range(len(all_bphys[0])):
		newvals.append([cell[i] for cell in all_bphys])
		ideal_order = find_polyfit_degree(diams,newvals[-1])+3
		print(i,ideal_order)
		sigma = np.ones(len(diams))
		sigma/=10
		sigma[[0, -1]] = 1.0
		coeffs = polyfit(diams,newvals[-1],ideal_order,weights=sigma)
		coefficients.append(coeffs['polynomial'])
		p = np.poly1d(coeffs['polynomial'])
		newvals[-1] = [p(diam) for diam in diams][diams.index(diameter)]
#		newvals[-1] = [np.polynomial.polynomial.polyval(diam,coeffs['polynomial']) for diam in diams][diams.index(diameter)]
	return(dict(zip(list(bphys.keys()),newvals)),coefficients)

def get_average_biophys():
	all_bphys = []
	for diam in [1.5,2.0,3.0,4.0,6.0,9.0,10.25,10.75]:
		fs,bphys = get_biophys(0,diam)
		all_bphys.append(list(bphys.values()))
	
	newvals = []
	for i in range(len(all_bphys[0])):
		newvals.append(np.mean([cell[i] for cell in all_bphys]))
	return(dict(zip(list(bphys.keys()),newvals)))

if __name__ == "__main__":
	#10.75: 

	#[67, 50, 31, 72, 40, 54, 72, 55]
	#from set -1.5[7.63],2.0[13.0],3.0[16.51],4.0[22.06],6.0[33.73],9.0[48.68],10.25[67.46],10.75[98.37]
	#averaged -1.5[7.33],2.0[10.73],3.0[17.68],4.0[24.98],6.0[39.38],9.0[61.32],10.25[68.43],10.75[72.64]

	#[89, 58, 21, 54, 56, 75, 88, 46]
	#from set -1.5[7.14],2.0[12.33],3.0[17.3],4.0[22.38],6.0[32.34],9.0[48.18],10.25[68.43],10.75[98.37]
	#averaged -1.5[7.28],2.0[10.66],3.0[17.55],4.0[24.72],6.0[39.02],9.0[60.54],10.25[67.46],10.75[72.64]

	#[92, 50, 44, 90, 17, 43, 52, 62]
	#from set -1.5[7.3],2.0[13.0],3.0[17.82],4.0[21.96],6.0[31.48],9.0[46.29],10.25[70.48],10.75[98.37]
	#averaged -1.5[7.23],2.0[10.59],3.0[17.43],4.0[24.59],6.0[38.7],9.0[60.54],10.25[67.46],10.75[72.64]

	#[44, 13, 31, 22, 27, 44, 74, 55]
	#from set -1.5[7.67],2.0[11.86],3.0[16.51],4.0[23.49],6.0[34.98],9.0[47.22],10.25[69.44],10.75[98.37]
	#averaged -1.5[7.34],2.0[10.76],3.0[17.75],4.0[24.98],6.0[39.35],9.0[61.32],10.25[68.43],10.75[72.64]

	#[77, 94, 6, 66, 21, 9, 45, 46]
	#from set -1.5[7.64],2.0[11.63],3.0[17.17],4.0[22.17],6.0[33.97],9.0[47.7],10.25[69.44],10.75[98.37]
	#averaged -1.5[7.46],2.0[10.93],3.0[17.95],4.0[25.39],6.0[40.0],9.0[62.13],10.25[69.44],10.75[73.78]

	#[93, 28, 49, 28, 38, 88, 27, 16]
	#from set -1.5[7.59],2.0[11.66],3.0[17.82],4.0[23.38],6.0[33.97],9.0[48.68],10.25[71.54],10.75[98.37]
	#averaged -1.5[7.44],2.0[10.88],3.0[17.89],4.0[25.25],6.0[39.68],9.0[62.13],10.25[69.44],10.75[73.78]
	#polyfit4_0.01 -1.5[7.59],2.0[broken],3.0[18.44],4.0[25.39],6.0[29.7],9.0[48.18],10.25[71.54],10.75[98.37]
	#polyfit4_0.5 -1.5[7.72],2.0[11.54],3.0[18.37],4.0[25.25],6.0[29.89],9.0[48.68],10.25[71.54],10.75[broken]
	#polyfit3 -1.5[7.63],2.0[11.75],3.0[19.19],4.0[25.25],6.0[32.56],9.0[51.32],10.25[72.64],10.75[broken]
	#polyfit3_0.1 -1.5[7.6],2.0[11.75],3.0[19.19],4.0[25.39],6.0[32.56],9.0[50.77],10.25[74.95],10.75[98.37]
	
	#2.5[15.64]
	#3.5[22.59]
	#5.0[29.89]
	#7.5[36.6]
	
	coefficients = [[-0.012356072057956973, 0.21656735890971254, -1.0323598118851582, 1.7155618229050826], [0.005743335614250873, -0.10813007328148706, 0.5816899763021781, -0.42265189328613745], [0.014803572451992672, -0.21522829909254113, 0.8309005212703641, 3.2524061387547665], [-0.009151145062082455, 0.1703983330332589, -1.0204120957882052, 5.617278700347728], [-6.545507840547101e-07, -0.000854720939570393, 0.007461946511450563, 0.031766168725249275], [0.0004226958108641459, -0.006756688694784406, 0.030082761409328647, -0.0003275471785293335], [0.00019947106597494093, -0.005335321916048501, 0.040329634305130035, -0.04412069296415507], [0.0002342952237988151, -0.005345256401971129, 0.034024145370508935, 0.011420737033593916], [-0.0015484188880430375, 0.07610030240896447, -0.7503480656961908, 3.211180852282207], [-3.3811363295646454e-05, -0.21202547161950636, 2.732686992172091, 12.501996096937425], [-0.029249812283428437, 0.5462444130257578, -2.625402389321036, 12.884623338205762], [0.0004730513607001158, -0.00853156877673857, 0.03854466323571184, 0.06604508716787859], [0.10145543333163272, -2.0791551283251826, 12.91272167704535, -1.728051500146552], [0.06390914473898548, -1.266168798548518, 7.568872659869069, -1.8876098168235513], [-1.074100932704553e-05, -0.00029156415283636787, 0.002452287994963917, 0.06286156654673099], [-0.8580802830626045, 18.100199840049314, -107.55220392264197, 259.21761518288656], [0.07214758904432403, -1.6178266431891724, 10.561028225833502, -6.587413122707197], [0.00034242176032506564, -0.007832091216749817, 0.049318535706094915, 0.2386774086390712], [0.0002135666515577409, -0.09959125008475513, 1.2959283297624866, -25.377186619820442], [-0.02609828878607266, 0.5205271675821085, -3.0845399201754686, -0.13653563231976867], [-0.6746432735475212, -2.3291073177245503, 118.79759658161747, 325.41110450056965], [2.4573508709743312, -40.255740564112735, 218.55503868958698, 618.2759622055509], [0.004804289246293937, -0.0809104155402136, 0.35328359273137633, 0.3358942798113312], [0.012068612999162797, -0.17165290138136857, 0.4907379224777041, 6.0828997782175955], [0.0003485007541102794, -0.006057390980305004, 0.02907099942701871, 0.010480601031340079], [-0.0004981016199367013, 0.00894170256840447, -0.04947951809595276, 0.13095100445874908], [0.01007899762812045, -0.17175132210783234, 0.8352937186799848, 0.948876854182836], [0.007135149218561226, -0.22375076794009024, 1.8902252615674209, 23.148890328178922], [0.0011471035400897381, 0.0032422801159672603, -0.38718515025565764, 11.435826043061963], [0.0003870517840065457, -0.006583985530779513, 0.031330945445500656, 0.06260512417352183], [-0.18999594326562372, 3.5858072345613405, -19.06869897678736, 49.540379659191515], [0.039452201037691696, -0.8549375062006664, 5.504769053753002, -0.17536547873581043], [-0.00018177915425726717, 0.003909138372488437, -0.02542121931985463, 0.11246645549854177], [0.05182188349895396, -2.6833715146709194, 23.190468731294306, 81.83564674006158], [0.07060820523517895, -1.4508346879822127, 8.42559681946022, -1.6279982180907213], [0.00046874543457928533, -0.005278741962679222, 4.792673138782425e-05, 0.38871095718888804], [0.04533026487771644, -0.8656913696986116, 5.5121364065561504, -46.36261088811552], [0.002832892518888237, -0.017377455442490967, -0.10155783072625718, -3.879287531126866]]
	
	
	
	#polyfit2 -1.5[broken],2.0[],3.0[],4.0[],6.0[],9.0[],10.25[],10.75[]
	
	#[]
	#from set -1.5[],2.0[],3.0[],4.0[],6.0[],9.0[],10.25[],10.75[]
	#averaged -1.5[],2.0[],3.0[],4.0[],6.0[],9.0[],10.25[],10.75[]


#	MRGCVs = [None,None,None,None,33.49,51.89,58.29,62.96]
#	FirminCVs = [range(7,11),range(10,16),range(15,21),range(20,31),range(30,46),range(45,61),range(60,81),range(80,111)]

	diameter = 7.5
	if diameter in [1.5,2.0,3.0,4.0,6.0,9.0,10.25,10.75]:
		cell_fnames,biophys = get_biophys(0,diameter)
	#	biophys = get_average_biophys()	
		biophys,coefficients = get_polynomial_biophys(diameter)
		if diameter == 1.5:
			print(coefficients)
	
	else:
		cell_fnames = get_fnames(0,diameter)
		params = []
		for co in coefficients:
			p = np.poly1d(co)
			params.append(p(diameter))
		biophys = dict(zip(['k_ib','k_b','na_ib','na_b','nap_ib','nap_b','gl_ib','gl_b','amA','amB','amC','bmA','bmB','bmC','ahA','ahB','ahC','asA','asB','asC','gbar_ib','gbar_b','k_nb','na_nb','gnap_nb','gl_nb','amA_n','amB_n','amC_n','bmA_n','bmB_n','bmC_n','ahA_n','ahB_n','ahC_n','asA_n','asB_n','asC_n'],params))
			
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


