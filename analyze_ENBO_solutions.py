import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib
matplotlib.rcParams.update({'font.size': 22})
#sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})



def parse_cvstring(cvstring):
	cvs = cvstring.split(',')
	keys = []
	values = []
	for c,cv in enumerate(cvs):
		keys.append(int(cv.split('[')[0]))
		values.append(float(cv.split('[')[1][:-1]))
	return(keys,values)

def load_mpool_cell(diameter,cellnum):
	with open(os.getcwd()+'/morphs_varyingdiameter/'+str(diameter)+'/'+'MatingPool.pickle','rb') as f:
		mpool = pickle.load(f)
	
	return(mpool[0][cellnum])

def scatter_variable(data,var,varind,varrange):
	ys = []
	xs = []
	for diam in data.keys():
		for cell in data[diam].keys():
			ys.append(data[diam][cell][varind])
			xs.append(diam)
	
	plt.scatter(xs,ys,label='ENBO Values')
	plt.ylabel('Value')
	plt.xlabel('Outer Diameter (um)')
	plt.ylim(varrange[0],varrange[1])
	plt.plot([diam for diam in data.keys()],[varrange[2] for diam in data.keys()],color='r',linestyle='dashed',label='MRG Value')
	plt.legend()
	plt.title(var)
	plt.show()

def scatter_CVs(data,refdata,refdata1):
	ys = []
	xs = []
	for diam in data.keys():
		ys.append([])
		for cell in data[diam].keys():
			ys[-1].append(data[diam][cell])
			xs.append(diam)
	
	plt.boxplot(refdata1,labels=[diam for diam in data.keys()],vert=True,meanline=False,showfliers=False,showmeans=False,showbox=False,autorange=True,whiskerprops=dict(linewidth=0),capprops=dict(color='#5C95FF',lw=2.0),medianprops=dict(linewidth=0))
	plt.boxplot(ys,labels=[diam for diam in data.keys()],vert=True,boxprops=dict(color='#2E1F27'),medianprops=dict(color='#2E1F27'),showfliers=False)
	mcintyre = plt.scatter(range(1,len(list(data.keys()))+1),refdata,label='Original Model Values (McIntyre et al., 2002)',color='#F87575')
#	plt.scatter(range(1,len(list(data.keys()))+1),refdata1,label='Firmin et al., 2014',color='r')
	plt.ylabel('Conduction Velocity (m/s)')
	plt.xlabel('Outer Diameter (\u03BCm)')
#	plt.legend(['Firmin et al., 2014','ENBO Values','McIntyre et al., 2002'])
	p1 = mpatch.Patch(color='#5C95FF',label='Experimental Bounds (Firmin et al., 2014)')
	p2 = mpatch.Patch(color='#0A090C',label='ENBO Values')
	plt.legend(handles=[p2,p1,mcintyre])
	plt.title('Fitting Conduction Velocity with ENBO')
	plt.show()

def almost_zero(varrange):
	if varrange[0] == 0:
		varrange[0] = 0.001
	if varrange[1] == 0:
		varrange[1] = 0.001
	return(varrange)

def scatter_variables_ratio(data,var1,varind1,varrange1,var2,varind2,varrange2):
	ys = []
	xs = []
	for diam in data.keys():
		for cell in data[diam].keys():
			try:
				ys.append(data[diam][cell][varind1]/data[diam][cell][varind2])
				xs.append(diam)
			except:
				continue
	
	fig = plt.figure(figsize=(5,5),dpi=150,tight_layout=True)
	plt.scatter(xs,ys,label=var1+':'+var2)
	varrange1 = almost_zero(varrange1)
	varrange2 = almost_zero(varrange2)
	plt.ylim(varrange1[0]/varrange2[0],varrange1[1]/varrange2[1])
	plt.plot([diam for diam in data.keys()],[varrange1[2]/varrange2[2] for diam in data.keys()],color='r',linestyle='dashed',label='MRG Value')
	plt.ylabel('Value')
	plt.xlabel('Outer Diameter (um)')
	plt.legend()
	plt.title(var1+':'+var2)
	plt.savefig(var1+'_'+var2+'.png')
	del(fig)
	plt.close('all')

if __name__ == "__main__":
	CVdict = {}
	for diam in [1.5,2.0,3.0,4.0,6.0,9.0,10.25,10.75]:
		CVdict[diam] = {}
	
	
	CVstrings = dict(zip([1.5,2.0,3.0,4.0,6.0,9.0,10.25,10.75],[
			'1[7.28],3[9.83],5[7.58],15[7.26],16[7.39],26[7.64],27[7.09],33[7.45],36[7.5],43[7.21],44[7.67],51[7.47],53[7.26],59[7.26],60[7.58],62[7.54],67[7.63],68[7.1],73[7.24],76[7.57],77[7.64],82[7.58],84[7.94],89[7.14],92[7.3],93[7.59]',
			'1[12.39],3[11.66],4[11.83],7[11.63],13[11.86],22[11.89],25[12.23],28[11.66],39[11.78],42[11.83],48[13.3],50[13.0],51[11.57],52[12.52],54[12.43],56[11.69],58[12.33],65[11.6],66[12.08],78[11.72],86[11.66],94[11.63]',
			'0[17.11],1[16.8],4[16.45],5[17.75],6[17.17],7[17.11],8[16.92],10[16.69],15[16.51],16[17.68],17[16.69],19[16.8],20[16.4],21[17.3],22[16.45],23[16.4],25[16.8],26[17.11],27[17.62],30[17.62],31[16.51],36[17.49],37[16.45],41[16.45],43[19.43],44[17.82],45[16.63],46[16.4],47[16.51],49[17.82],50[16.86],52[16.92],54[17.3],57[16.92],58[17.3],60[16.45],61[16.74],63[16.92],65[16.92],66[17.11],67[16.86],69[18.89],70[17.05],71[18.1],72[16.63],73[18.44],75[16.45],76[17.95],78[16.8],79[16.4],80[16.63],82[16.4],83[18.23],84[16.86],85[16.51],86[16.34],87[16.51],88[16.63],89[16.63],92[17.75],93[16.99],94[17.11]',
			'2[21.56],9[23.26],10[22.38],11[26.38],12[25.66],13[23.15],14[24.85],15[21.86],16[24.98],18[28.44],20[23.5],21[22.27],22[23.49],27[23.49],28[23.38],31[22.92],37[21.86],38[26.1],40[25.12],41[23.38],43[23.03],45[22.38],47[22.81],48[22.59],52[23.03],53[22.49],54[22.38],57[25.25],60[22.49],62[25.12],66[22.17],67[21.86],72[22.06],73[28.11],77[23.26],80[22.81],84[23.97],85[22.06],87[23.61],88[23.73],89[23.38],90[21.96],91[24.21],92[21.76],94[24.85]',
			'1[35.77],2[34.72],4[34.72],6[32.79],7[36.89],9[34.47],10[33.25],11[31.69],12[31.48],13[33.73],14[31.9],15[34.22],16[33.49],17[31.48],18[35.77],19[37.18],20[33.73],21[33.97],23[31.48],24[32.12],25[33.73],26[34.72],27[34.98],28[36.89],29[34.47],30[34.72],31[35.24],32[34.72],33[35.24],34[31.69],36[34.47],37[31.7],38[33.97],39[34.72],40[33.73],41[31.48],42[31.48],43[35.24],44[36.04],45[32.56],46[31.48],47[33.02],48[31.69],49[33.25],51[34.72],52[34.72],53[34.47],54[35.24],56[32.34],57[32.56],59[36.89],60[34.22],61[36.04],63[32.34],64[31.48],66[31.48],67[33.02],68[32.79],69[36.89],70[38.08],71[32.56],73[38.39],75[34.98],76[37.47],78[32.79],79[33.97],80[33.97],81[35.77],82[33.02],83[32.56],84[37.77],85[33.02],86[31.9],88[37.47],90[34.72],91[33.02],92[34.72],93[31.48],94[34.22]',
			'0[51.89],1[48.18],2[53.05],3[50.77],4[49.7],5[47.22],6[47.7],7[47.7],8[52.46],9[47.7],10[50.77],12[48.7],13[53.05],15[48.18],16[47.7],17[46.75],19[51.89],20[47.22],21[50.23],22[47.22],24[47.22],25[51.89],26[53.05],28[46.29],30[47.7],31[50.77],32[48.68],34[49.19],35[46.75],37[47.22],38[53.66],39[51.89],40[47.22],41[50.77],42[53.66],43[46.29],44[47.22],46[48.18],47[50.23],48[51.89],49[49.7],50[52.46],51[53.66],52[50.23],53[53.66],54[48.68],55[51.32],56[46.75],57[51.89],58[51.32],59[52.46],60[50.77],61[49.7],63[49.7],64[40.77],65[49.19],67[46.75],69[50.77],71[49.19],73[51.89],75[48.18],77[48.18],78[49.19],79[47.22],81[48.68],82[49.19],83[51.32],84[50.23],85[50.77],86[49.7],87[53.66],88[48.68],89[50.23],90[50.77],91[53.05],93[52.46],94[52.47]',
			'1[70.48],2[72.64],3[70.48],5[72.64],7[71.54],8[72.64],10[72.64],11[66.5],12[68.43],14[71.54],18[72.64],20[66.5],27[71.54],29[68.43],30[66.5],31[66.5],32[72.64],40[67.46],42[71.54],44[70.48],45[69.44],46[72.64],48[67.46],51[69.44],52[70.48],54[67.46],55[66.5],58[66.5],63[70.48],64[66.5],67[72.64],68[70.48],69[69.44],72[67.46],73[67.46],74[69.44],78[70.48],79[69.44],82[70.48],83[70.48],88[68.43],89[66.5],92[67.46],94[68.43]',
			'16[98.37],24[98.37],34[98.37],40[98.37],46[98.37],55[98.37],61[98.37],62[98.37],65[98.37],89[96.36]'
			]))
	
	variables = ['k_ib','k_b','na_ib','na_b','nap_ib','nap_b','gl_ib','gl_b','amA','amB','amC','bmA','bmB','bmC',
				'ahA','ahB','ahC','asA','asB','asC','gbar_ib','gbar_b','k_nb','na_nb','gnap_nb','gl_nb','amA_n',
				'amB_n','amC_n','bmA_n','bmB_n','bmC_n','ahA_n','ahB_n','ahC_n','asA_n','asB_n','asC_n']
	
	offvariables = ['k_ib','na_ib','gl_ib','gbar_ib','nap_ib','k_b','na_b','gl_b','gbar_b','nap_b','amA','amB','amC','bmA','bmB','bmC','ahA','ahB','ahC','asA','asB','asC']
	
	onvariables = [var for var in variables if var not in offvariables]
	onvar_range = [ [0,1,0.08], #node gkbar_axnode 'k_nb'
					[0.5,7.0,3.0], #node gnabar_axnode 'na_nb'
					[0,0.1,0.005], #node gnapbar_axnode 'gnap_nb'
					[0,0.1,0.007],#node gl_axnode 'gl_nb'
					[1.88*0.5,1.88*1.5,1.86], # 'amA_n'
					[21.4*0.5,21.4*1.5,21.4], # 'amB_n'
					[10.3*0.5,10.3*1.5,10.3], # 'amC_n'
					[0.086*0.5,0.086*1.5,0.086], # 'bmA_n'
					[25.7*0.5,25.7*1.5,25.7], # 'bmB_n'
					[9.16*0.5,9.16*1.5,9.16], # 'bmC_n'
					[0.062*0.5,0.062*1.5,0.062], # 'ahA_n'
					[114.0*0.5,114.0*1.5,114.0], # 'ahB_n'
					[11.0*0.5,11.0*1.5,11.0], # 'ahC_n'
					[0.3*0.5,0.3*1.5,0.3], # 'asA_n'
					[-27*1.5,-27*0.5,-27], # 'asB_n'
					[-5*1.5,-5*0.5,-5.0], # 'asC_n'
					]
	
	MRGCVs = [None,None,None,None,33.49,51.89,58.29,62.96]
	FirminCVs = [8.5,13.0,18.0,25.0,35.0,50.0,70.0,100.0] #[7-10,10-15,15-20,20-30,30-45,45-60,60-80,80-110]
	FirminCVs = [range(7,11),range(10,16),range(15,21),range(20,31),range(30,46),range(45,61),range(60,81),range(80,111)]
	physdict = {}
	for diam in CVdict.keys():
		physdict[diam] = {}
	
	for diam in CVdict.keys():
		keys,values = parse_cvstring(CVstrings[diam])
		for i in range(len(keys)):
			CVdict[diam][keys[i]] = values[i]
			physdict[diam][keys[i]] = [item for it,item in enumerate(load_mpool_cell(diam,i)) if variables[it] not in offvariables]
		
	scatter_CVs(CVdict,MRGCVs,FirminCVs)
#	for i in range(len(onvariables)):
#		scatter_variable(physdict,onvariables[i],i,onvar_range[i])
	
#	from itertools import combinations
#	combs = [item for item in combinations(range(len(onvariables)),2)]
#	for comb in combs:
#		scatter_variables_ratio(physdict,onvariables[comb[0]],comb[0],onvar_range[comb[0]],onvariables[comb[1]],comb[1],onvar_range[comb[1]])
	
#	rphysdict = {}
#	cvdata = []
#	gnew = []
#	for diam in CVdict.keys():
#		cvdata.append([CVdict[diam][cell] for cell in CVdict[diam].keys()])
