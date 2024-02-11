#from roots.microstructures import Microstructures
#from iterativemicrostructures import IterativeMicrostructures
from roots.root2neuron import Root2Py
from roots.swcToolkit import swcToolkit
from roots.LinearAddMicrostructures import LinearAddMicrostructures
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#scale
label_scalars = {}
label_scalars['node'] = 1
label_scalars['internode'] = 8
label_scalars['paranode1'] = 8
label_scalars['paranode2'] = 8
label_scalars['interbouton'] = 2
label_scalars['bouton'] = 8

def write_section_centers(arbor,centers = [],target='sectionCenters.csv'):
	if centers == []:
		print('no centers passed, finding now and writing to '+target)
		for branch in arbor.keys():
			for section in arbor[branch]:
				centers.append(section[int(len(section)/2)])
	
	else:
		labelli = []
		for branch in centers.keys():
			for label in centers[branch]:
				labelli.append(label)
		
		centers=labelli
	
	df = pd.DataFrame()
	df['sectionCenters'] = range(len(centers))
	df['x'] = [center[0] for center in centers]
	df['y'] = [center[1] for center in centers]
	df['z'] = [center[2] for center in centers]
	df.to_csv(target,index=False)

def write_labels_to_csv(labels,target='sectionType.csv'):
#	labelli = {}
#	for branch in labels.keys():
#		for label in labels[branch]:
#			labelli[len(labelli.keys())] = label
	
	labelli = []
	for branch in labels.keys():
		for label in labels[branch]:
			labelli.append(label)
	
	print(len(labelli),'len labels')
	df = pd.DataFrame()
	df['sectionList'] = range(len(labelli)) #list(labelli.keys())
	df['sectionType'] = labelli #list(labelli.values())
	df.to_csv(target,index=False)

def write_lengths_to_csv(labels,target='sectionLengths.csv'):
#	labelli = {}
#	for branch in labels.keys():
#		for label in labels[branch]:
#			labelli[len(labelli.keys())] = label
	
	labelli = []
	for branch in labels.keys():
		for label in labels[branch]:
			labelli.append(round(label,2))
	
	print(len(labelli),'len lengths')
	df = pd.DataFrame()
	df['sectionList'] = range(len(labelli)) #list(labelli.keys())
	df['sectionLength'] = labelli #list(labelli.values())
	df.to_csv(target,index=False)



class GetMicrostructureGeometry():
	def __init__(self):
		pass
	
	def interpolate_fiber_dep_vars(self,fiberD,outerdiams):
		for d in range(len(outerdiams[:-1])):
			if fiberD>= outerdiams[d] and fiberD<=outerdiams[d+1]:
				return(d,d+1,float((fiberD-outerdiams[d])/(outerdiams[d+1]-outerdiams[d])))
		
		return(None)

	def calculate_new_dep_var(self,a,b,prop):
		return(a+(b-a)*prop)
	
	def get_values_from_fitted_curve(self,xreal,yreal,xnew):
		z = np.polyfit(xreal,yreal,2)
		f = np.poly1d(z)
		ynew = f(xnew)
		return(ynew)

	def dependent_vars(self,fiberD):
		ddict = {}
		ddict['outerdiams'] = [5.7,7.3,8.7,10.0,11.5,12.8,14.0,15.0,16.0]
		ddict['gs'] = [0.605,0.630,0.661,0.690,0.700,0.719,0.739,0.767,0.791]
		ddict['axonDs'] = [3.4,4.6,5.8,6.9,8.1,9.2,10.4,11.5,12.7]
		ddict['nodeDs'] = [1.9,2.4,2.8,3.3,3.7,4.2,4.7,5.0,5.5]
		ddict['paraD1s']=[1.9,2.4,2.8,3.3,3.7,4.2,4.7,5.0,5.5]
		ddict['paraD2s']=[3.4,4.6,5.8,6.9,8.1,9.2,10.4,11.5,12.7]
		ddict['deltaxs']=np.array([500,750,1000,1150,1250,1350,1400,1450,1500])
		ddict['paralength2s']=np.array([35,39,40,46,50,54,56,58,60])
		ddict['nls'] = [80,100,110,120,130,135,140,145,150]
		if fiberD < ddict['outerdiams'][0] or fiberD > ddict['outerdiams'][-1]:
			prop = None
		else:
			prop = self.interpolate_fiber_dep_vars(fiberD,ddict['outerdiams'])
		if prop is None:
			print('requested fiber diameter is out of range')
			dep_vars = []
			for key in ddict.keys():
				if key == 'outerdiams':
					dep_vars.append(fiberD)
				else:
					dep_vars.append(self.get_values_from_fitted_curve(ddict['outerdiams'],ddict[key],fiberD))
			return(dep_vars)
		
		else:
			dep_vars = []
			for key in ddict.keys():
				dep_vars.append(self.calculate_new_dep_var(ddict[key][prop[0]],ddict[key][prop[1]],prop[2]))
		
		return(dep_vars)

	def calculate_morph_vars(self,fiberD):
		dep_vars = self.dependent_vars(fiberD)
		interlen = int((dep_vars[6]-1-(2*3)-(2*dep_vars[7]))/6)
		if interlen < 0:
			interlen = 1.0
		return([dep_vars[3],dep_vars[4],dep_vars[5],dep_vars[0],dep_vars[5],dep_vars[4]],[1,3,int(dep_vars[7]),interlen,int(dep_vars[7]),3])


'''
proc dependent_var() {
	if (fiberD==5.7) {g=0.605 axonD=3.4 nodeD=1.9 paraD1=1.9 paraD2=3.4 deltax=500 paralength2=35 nl=80}
	if (fiberD==7.3) {g=0.630 axonD=4.6 nodeD=2.4 paraD1=2.4 paraD2=4.6 deltax=750 paralength2=38 nl=100}
	if (fiberD==8.7) {g=0.661 axonD=5.8 nodeD=2.8 paraD1=2.8 paraD2=5.8 deltax=1000 paralength2=40 nl=110}
	if (fiberD==10.0) {g=0.690 axonD=6.9 nodeD=3.3 paraD1=3.3 paraD2=6.9 deltax=1150 paralength2=46 nl=120}
	if (fiberD==11.5) {g=0.700 axonD=8.1 nodeD=3.7 paraD1=3.7 paraD2=8.1 deltax=1250 paralength2=50 nl=130}
	if (fiberD==12.8) {g=0.719 axonD=9.2 nodeD=4.2 paraD1=4.2 paraD2=9.2 deltax=1350 paralength2=54 nl=135}
	if (fiberD==14.0) {g=0.739 axonD=10.4 nodeD=4.7 paraD1=4.7 paraD2=10.4 deltax=1400 paralength2=56 nl=140}
	if (fiberD==15.0) {g=0.767 axonD=11.5 nodeD=5.0 paraD1=5.0 paraD2=11.5 deltax=1450 paralength2=58 nl=145}
	if (fiberD==16.0) {g=0.791 axonD=12.7 nodeD=5.5 paraD1=5.5 paraD2=12.7 deltax=1500 paralength2=60 nl=150}
	Rpn0=(rhoa*.01)/(PI*((((nodeD/2)+space_p1)^2)-((nodeD/2)^2)))
	Rpn1=(rhoa*.01)/(PI*((((paraD1/2)+space_p1)^2)-((paraD1/2)^2)))
	Rpn2=(rhoa*.01)/(PI*((((paraD2/2)+space_p2)^2)-((paraD2/2)^2)))
	Rpx=(rhoa*.01)/(PI*((((axonD/2)+space_i)^2)-((axonD/2)^2)))
	interlength=(deltax-nodelength-(2*paralength1)-(2*paralength2))/6
	}
'''
def eucdist3d(point1,point2):
	"""
	
	euclidean distance between point1 and point2 - [x,y,z]
	
	"""
	
	return(((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2 + (point2[2]-point1[2])**2)**0.5)

def get_mstruct_distribution(arbor,labels,label='internode'):
	dists = []
	for branch in arbor.keys():
		for section in range(len((arbor[branch]))):
			if labels[branch][section] == 'internode':
				a = arbor[branch][section][0]
				b = arbor[branch][section][2]
				dists.append(((b[0]-a[0])**2+(b[1]-a[1])**2+(b[2]-a[2])**2)**0.5)
	return(dists)


def find_CCF_in_morph(cell):
	CCF_end = [48991.04393648394, 1277.7531299998796, 88190.47619047619]
	myelin = []
	myelin.append(np.argmin([eucdist3d(CCF_end,cell[branch][-1]) for branch in cell.keys()]))
	myelin.append(np.argmin([eucdist3d(cell[myelin[0]][0],cell[branch][-1]) for branch in cell.keys()]))
	return(myelin)

def plot_by_branch(morphology,myelin):
	from mpl_toolkits.mplot3d import Axes3D
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for branch in morphology.keys():
		xs = [item[0] for item in morphology[branch]]
		ys = [item[1] for item in morphology[branch]]
		zs = [item[2] for item in morphology[branch]]
		if branch in myelin:
			ax.plot(xs,ys,zs,color='r')
		else:
			ax.plot(xs,ys,zs,color='b')
	
	plt.show()

def make_IC(tree,myelin):
	newtree = dict()
	newmyelin = []
	for branch in tree.keys():
		if branch in myelin:
			newbranch = len(list(newtree.keys()))
			newtree[newbranch] = tree[branch]
			newmyelin.append(newbranch)
	
	return(newtree,newmyelin)

def spread_negative_internode_dist(diams,lengths):
	if lengths[3]<0:
		inodel = lengths[3]/2.0 -0.5
		lengths[2] += inodel
		lengths[3] = 1.0
		lengths[4] += inodel
	
	return(diams,lengths)

if __name__ == "__main__":
	#########################################################
	#          update swcs with microstructures			    #
	#########################################################
	fdiams = [10.75,10.25,9.0,6.0,4.0,3.0,2.0,1.5] #from firmin et al., 2014
	
	for fdiam in fdiams:
		write_IC = False
		swctool = swcToolkit()
		py_writer = Root2Py()
	#	mstruct = Microstructures()
	#	mstruct = IterativeMicrostructures()
		swcs = [os.getcwd()+'/morphs_varyingdiameter/'+fname for fname in os.listdir(os.getcwd()+'/morphs_varyingdiameter') if '.swc' in fname]
		geo = GetMicrostructureGeometry()
		diams,lengths = geo.calculate_morph_vars(fdiam)
		diams,lengths = spread_negative_internode_dist(diams,lengths)
		print(diams,lengths,fdiam)
		for swc in swcs:
			print(swc)
			morph = swctool.load(swc)
	#		morph = swctool.scale_morphology(morph,[0.1,0.1,0.1])
	#		myelin = find_CCF_in_morph(morph)
			myelin = list(morph.keys())
	#		if write_IC:
	#			morph,myelin = make_IC(morph,myelin)
			
			mstruct = LinearAddMicrostructures(morph,myelinlist = myelin,myelindimensions=[['node','paranode1','paranode2','internode','paranode2','paranode1'],lengths])
			arbor_,labels_,lengths_,centers_ = mstruct.check_return_results()
#			py_writer.arbor_to_nrn(arbor_,labels_,target=swc.strip('.swc')+str(fdiam)+'.py',suppress_points=True)
#			write_labels_to_csv(labels_,target=swc.strip('.swc')+'_section_labels'+str(fdiam)+'.csv')
#			write_lengths_to_csv(lengths_,target=swc.strip('.swc')+'_section_lengths'+str(fdiam)+'.csv')
#			write_section_centers(arbor_,centers=centers_,target=swc.strip('.swc')+'_section_centers'+str(fdiam)+'.csv')

