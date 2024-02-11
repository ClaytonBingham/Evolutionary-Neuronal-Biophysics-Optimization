import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib
import random
matplotlib.rcParams.update({'font.size': 22})
#sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

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

def pick_random_models(physdict,onvar_range):
	picks = []
	cells = []
	for diam in physdict.keys():
		choice = random.choice(list(physdict[diam].keys()))
		picks.append(physdict[diam][choice])
		cells.append(choice)
	score = evaluate_random_models(picks,onvar_range)
	return(score,picks,cells)

def evaluate_random_models(choices,onvar_range):
	variances = []
	for i in range(len(choices[0])):
		variances.append(np.var([choice[i]/(onvar_range[i][1]-onvar_range[i][0]) for choice in choices]))
	return(np.mean(variances))

def polyfit(x,y,degree):
	results = {}
	coeffs = np.polyfit(x,y,degree)
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

def get_polynomial_biophys(all_bphys,diameter):
	diams = [1.5,2.0,3.0,4.0,6.0,9.0,10.25,10.75]
	newvals = []
	coefficients = []
	for i in range(len(all_bphys[0])):
		newvals.append([cell[i] for cell in all_bphys])
		ideal_order = find_polyfit_degree(diams,newvals[-1])
		print(i,ideal_order)
		coeffs = polyfit(diams,newvals[-1],ideal_order)
		coefficients.append(coeffs['polynomial'])
#		newvals[-1] = [np.polynomial.polynomial.polyval(diam,coeffs['polynomial']) for diam in diams][diams.index(diameter)]
		newvals[-1] = [np.polynomial.polynomial.polyval(diam,coeffs['polynomial']) for diam in diams]
	return(dict(zip(list(bphys.keys()),newvals)),coefficients)

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
	FirminCVs = [range(7,11),range(10,16),range(15,21),range(20,31),range(30,46),range(45,61),range(60,81),range(80,111)]
	physdict = {}
	for diam in CVdict.keys():
		physdict[diam] = {}
	
	for diam in CVdict.keys():
		keys,values = parse_cvstring(CVstrings[diam])
		for i in range(len(keys)):
			CVdict[diam][keys[i]] = values[i]
			physdict[diam][keys[i]] = [item for it,item in enumerate(load_mpool_cell(diam,i)) if variables[it] not in offvariables]
	
	
	
	
	low_score = 100
	best_picks = None
	choice = None
	i = 0
	while i < 10000:
		score, picks,choices = pick_random_models(physdict,onvar_range)
		if score < low_score:
			low_score = score
			best_picks = picks
			choice = choices

			print(low_score,'low score')
		i+=1
	
	print(choices,'choices')
	with open('picks.pickle','wb') as f:
		pickle.dump([choice,best_picks],f)
	
	with open('picks.pickle','rb') as f:
		choices, best_picks = pickle.load(f)
	
	means = []
	for i in range(len(best_picks[0])):
		ys = [pick[i] for pick in best_picks]
		means.append(np.mean(ys))
		x = [1.5,2.0,3.0,4.0,6.0,9.0,10.25,10.75]
		plt.plot(x,ys,label=onvariables[i])
		coeffs = polyfit(x,ys,5)['polynomial']
		polys = [np.polynomial.polynomial.polyval(diam,coeffs[::-1]) for diam in x]
		plt.plot(x,polys,label=onvariables[i]+' polyline')
		plt.legend()
		plt.xlabel('Outer Diameter (\u03BCm)')
		plt.ylabel('Parameter Value')
		plt.show()
#	with open('bestpick.pickle','wb') as f:
#		pickle.dump(dict(zip(onvariables,means)),f)
#	picks_to_dict = []
#	for pick in best_picks:
#		picks_to_dict.append(dict(zip(onvariables,pick)))
#	with open('picksdict.pickle','wb') as f:
#		pickle.dump(picks_to_dict,f)
	
	
	
	
	
