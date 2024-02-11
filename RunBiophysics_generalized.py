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
import argparse

def ez_record(h,var='v',sections=None,order=None,targ_names=None,cust_labels=None,level='section'):
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



if __name__ == "__main__":
	##############################################################
	#       reference data and fixed variables/coefficients      #
	##############################################################
	#polyfit3_0.1 -1.5[7.6],2.0[11.75],2.5[15.64],3.0[19.19],3.5[22.59],4.0[25.39],5.0[29.89],6.0[32.56],7.5[36.6],9.0[50.77],10.25[74.95],10.75[98.37]
	
	'''
	k_nb = 0.004804x^3 - 0.08091x^2 + 0.3533x + 0.3359
	na_nb = 0.01207x^3 - 0.1717x^2 + 0.4907x + 6.083
	gnap_nb = 0.0003485x^3 - 0.006057x^2 + 0.02907x + 0.01048
	gl_nb = -0.0004981x^3 + 0.008942x^2 - 0.04948x + 0.131
	amA_n = 0.01008x^3 - 0.1718x^2 + 0.8353x + 0.9489
	amB_n = 0.007135x^3 - 0.2238x^2 + 1.89x + 23.15
	amC_n = 0.001147x^3 + 0.003242x^2 - 0.3872x + 11.44
	bmA_n = 0.0003871x^3 - 0.006584x^2 + 0.03133x + 0.06261
	bmB_n = -0.19x^3 + 3.586x^2 - 19.07x + 49.54
	bmC_n = 0.03945x^3 - 0.8549x^2 + 5.505x - 0.1754
	ahA_n = -0.0001818x^3 + 0.003909x^2 - 0.02542x + 0.1125
	ahB_n = 0.05182x^3 - 2.683x^2 + 23.19x + 81.84
	ahC_n = 0.07061x^3 - 1.451x^2 + 8.426x - 1.628
	asA_n = 0.0004687x^3 - 0.005279x^2 + 4.793e-05x + 0.3887
	asB_n = 0.04533x^3 - 0.8657x^2 + 5.512x - 46.36
	asC_n = 0.002833x^3 - 0.01738x^2 - 0.1016x - 3.879
	'''
	coefficients = [[-0.012356072057956973, 0.21656735890971254, -1.0323598118851582, 1.7155618229050826], 
					[0.005743335614250873, -0.10813007328148706, 0.5816899763021781, -0.42265189328613745], 
					[0.014803572451992672, -0.21522829909254113, 0.8309005212703641, 3.2524061387547665], 
					[-0.009151145062082455, 0.1703983330332589, -1.0204120957882052, 5.617278700347728], 
					[-6.545507840547101e-07, -0.000854720939570393, 0.007461946511450563, 0.031766168725249275], 
					[0.0004226958108641459, -0.006756688694784406, 0.030082761409328647, -0.0003275471785293335], 
					[0.00019947106597494093, -0.005335321916048501, 0.040329634305130035, -0.04412069296415507], 
					[0.0002342952237988151, -0.005345256401971129, 0.034024145370508935, 0.011420737033593916], 
					[-0.0015484188880430375, 0.07610030240896447, -0.7503480656961908, 3.211180852282207], 
					[-3.3811363295646454e-05, -0.21202547161950636, 2.732686992172091, 12.501996096937425], 
					[-0.029249812283428437, 0.5462444130257578, -2.625402389321036, 12.884623338205762], 
					[0.0004730513607001158, -0.00853156877673857, 0.03854466323571184, 0.06604508716787859], 
					[0.10145543333163272, -2.0791551283251826, 12.91272167704535, -1.728051500146552], 
					[0.06390914473898548, -1.266168798548518, 7.568872659869069, -1.8876098168235513], 
					[-1.074100932704553e-05, -0.00029156415283636787, 0.002452287994963917, 0.06286156654673099], 
					[-0.8580802830626045, 18.100199840049314, -107.55220392264197, 259.21761518288656], 
					[0.07214758904432403, -1.6178266431891724, 10.561028225833502, -6.587413122707197], 
					[0.00034242176032506564, -0.007832091216749817, 0.049318535706094915, 0.2386774086390712], 
					[0.0002135666515577409, -0.09959125008475513, 1.2959283297624866, -25.377186619820442], 
					[-0.02609828878607266, 0.5205271675821085, -3.0845399201754686, -0.13653563231976867], 
					[-0.6746432735475212, -2.3291073177245503, 118.79759658161747, 325.41110450056965], 
					[2.4573508709743312, -40.255740564112735, 218.55503868958698, 618.2759622055509], 
					[0.004804289246293937, -0.0809104155402136, 0.35328359273137633, 0.3358942798113312], 
					[0.012068612999162797, -0.17165290138136857, 0.4907379224777041, 6.0828997782175955], 
					[0.0003485007541102794, -0.006057390980305004, 0.02907099942701871, 0.010480601031340079], 
					[-0.0004981016199367013, 0.00894170256840447, -0.04947951809595276, 0.13095100445874908], 
					[0.01007899762812045, -0.17175132210783234, 0.8352937186799848, 0.948876854182836], 
					[0.007135149218561226, -0.22375076794009024, 1.8902252615674209, 23.148890328178922], 
					[0.0011471035400897381, 0.0032422801159672603, -0.38718515025565764, 11.435826043061963], 
					[0.0003870517840065457, -0.006583985530779513, 0.031330945445500656, 0.06260512417352183], 
					[-0.18999594326562372, 3.5858072345613405, -19.06869897678736, 49.540379659191515], 
					[0.039452201037691696, -0.8549375062006664, 5.504769053753002, -0.17536547873581043], 
					[-0.00018177915425726717, 0.003909138372488437, -0.02542121931985463, 0.11246645549854177], 
					[0.05182188349895396, -2.6833715146709194, 23.190468731294306, 81.83564674006158], 
					[0.07060820523517895, -1.4508346879822127, 8.42559681946022, -1.6279982180907213], 
					[0.00046874543457928533, -0.005278741962679222, 4.792673138782425e-05, 0.38871095718888804], 
					[0.04533026487771644, -0.8656913696986116, 5.5121364065561504, -46.36261088811552], 
					[0.002832892518888237, -0.017377455442490967, -0.10155783072625718, -3.879287531126866]
					]
	
	variables = ['k_ib','k_b','na_ib','na_b','nap_ib','nap_b','gl_ib','gl_b','amA','amB','amC','bmA','bmB','bmC',
				'ahA','ahB','ahC','asA','asB','asC','gbar_ib','gbar_b','k_nb','na_nb','gnap_nb','gl_nb','amA_n',
				'amB_n','amC_n','bmA_n','bmB_n','bmC_n','ahA_n','ahB_n','ahC_n','asA_n','asB_n','asC_n']
	offvariables = ['k_ib','na_ib','gl_ib','gbar_ib','nap_ib','k_b','na_b','gl_b','gbar_b','nap_b','amA','amB','amC','bmA','bmB','bmC','ahA','ahB','ahC','asA','asB','asC']
	onvariables = [var for var in variables if var not in offvariables]
	MRGCVs = [None,None,None,None,33.49,51.89,58.29,62.96]
	FirminCVs = [range(7,11),range(10,16),range(15,21),range(20,31),range(30,46),range(45,61),range(60,81),range(80,111)]
	
	##############################################################
	#    handle CLI arguments and build dependent variables      #
	##############################################################
	ap = argparse.ArgumentParser()
	ap.add_argument("-d", "--diameter", required = False, type=float,default = 10.75)
	args = vars(ap.parse_args())
	if args["diameter"] < 1.5 or args["diameter"] > 10.75:
		print('diameter requested is out of the range studied and model performance may be unstable. Studied range was limited to 1.5-10.75 microns outer diameter.')
	cell_fnames = get_fnames(0,args["diameter"])
	params = []
	for c,co in enumerate(coefficients):
		p = np.poly1d(co)
		if variables[c] in onvariables:
			print(variables[c],np.poly1d(p))
		
		params.append(p(args["diameter"]))
	biophys = dict(zip(['k_ib','k_b','na_ib','na_b','nap_ib','nap_b','gl_ib','gl_b','amA','amB','amC','bmA','bmB','bmC','ahA','ahB','ahC','asA','asB','asC','gbar_ib','gbar_b','k_nb','na_nb','gnap_nb','gl_nb','amA_n','amB_n','amC_n','bmA_n','bmB_n','bmC_n','ahA_n','ahB_n','ahC_n','asA_n','asB_n','asC_n'],params))
		
	print('biophysics '+str(50),cell_fnames[0],0)

	###########################
	#       run model         #
	###########################
	cell,on_data,stim = run_cell_modifybiophys(2.0,[cell_fnames[0],cell_fnames[1],cell_fnames[2]],plot=True,biophysics=biophys,CCFdiam=args["diameter"])
	on_data = np.transpose(on_data)
	
	###################################
	#Conduction Velocity Cost Function#
	###################################
	cv=ConductionVelocity()
	myelinatedCV = cv.calculate_CVs(cell,on_data,h.dt)
	cost = cv.calculate_cost(myelinatedCV,8.5,h.dt,4.0625,on_data)
	output_sptime_params(list(biophys.values()),cost)
