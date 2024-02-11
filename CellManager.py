from neuron import h
import numpy as np
import random

class CellManager():
	"""
	This class is used to assign biophysics and discretize NEURON morphologies after they are already created in NEURON python namespace.
	
	Example usage:
	
	```
	cm = CellManager()
	for section in sectionList:
		secref = cm.assign_biophysics_to_section(sec,type='node')
	
	cm.fixnseg()
	```
	
	This class is dependent upon having the fixnseg.hoc file in root and having nrnivmodl'd an AXNODE.mod and PARAK75.mod in /x86_64 in root.
	
	These items can be found in these places, respectively:
	
	https://senselab.med.yale.edu/ModelDB/ShowModel?model=114685&file=/JohnsonMcIntyre2008/GPe_model/#tabs-2
	
	https://www.neuron.yale.edu/neuron/static/docs/d_lambda/d_lambda.html
	
	
	cad    - (CP) - calcium pump
	kca	   - (KCa) - calcium dependent potassium channel
	SlowCa - (Cas) - slow calcium channel
	IKM	   - (Km) - non-inactivating voltage dependent slow potassium channel
	nap	   - (Nap) - persistent sodium channel
	kslow  - (Kslow) - slow potassium channel
	kfast  - (Kfast) - fast potassium channel
	nat	   - (Nat) - transient sodium channel Nav1.6
	na12   - (Na) - transient sodium channel Nav1.2
	h	   - (HCN) - hyperpolarization-activated cation channel
	ih	   - h
	Kv1	   - potassium
	kv	   - potassium
	nacurrent- sodium channel
	"""
	
	def __init__(self,sectionlist_fname,types_fname,lengths_fname,CCF_outer=10.25):
		self.all = []
		self.soma = []
		self.hillock = []
		self.iseg = []
		self.basal = []
		self.apical = []
		self.tuft = []
		self.node = []
		self.paranode1 = []
		self.paranode2 = []
		self.internode = []
		self.interbouton = []
		self.bouton = []
		self.outerdiam = CCF_outer
		self.set_global_params()
		self.sort_section_types(self.get_sectionlist(sectionlist_fname),types_fname,lengths_fname)
		self.apply_biophysics()
		self.set_other_mod_globals() 
#		self.fixnseg()
#		self.print_passive_properties_by_section_type()
	
	def get_sectionlist(self,sectionlist_fname):
		namespace = {}
		with open(sectionlist_fname,'r') as f:
			exec(f.read(),namespace)
		
		return(namespace['sectionList'])
	
	def set_global_params(self):
		self.e_pas = -83.056442
		self.Rm_axosomatic = 23823.061083
		self.axosomatic_cm = 2.298892
		self.spinefactor = 0.860211
		self.soma_gbar_nat = 284.546493
		self.soma_gbar_kfast = 50.802287
		self.soma_gbar_kslow = 361.584735
		self.soma_gbar_nap = 0.873246
		self.soma_gbar_km = 7.123963
		self.basal_gbar_ih = 15.709707
		self.tuft_gbar_ih = 17.694744
		self.tuft_gbar_nat = 6.558244
		self.decay_kfast = 58.520995
		self.decay_kslow = 42.208044
		self.hillock_gbar_nat = 8810.657100
		self.iseg_gbar_nat = 13490.395442
		self.iseg_vshift2_nat = -9.802976
		self.Ra_apical = 454.05939784
		self.apical_Ra = self.Ra_apical
		self.tuft_gbar_sca = 3.67649485
		self.tuft_vshift_sca = 7.4783781
		self.tuft_gbar_kca = 9.75672674
#		self.outerdiam = 4.0
		diams,lengths,self.nl = self.calculate_morph_vars(self.outerdiam)
		self.nodeD = diams[0]
		self.para1D = diams[1]
		self.para2D = diams[1]
		self.interD = diams[1]
#		self.para2D = diams[2]
#		self.interD = diams[3]
	
	def set_other_mod_globals(self):
		h.vtraub_axnode = -80
#		h.vshift_axnode75 = 15
#		h.vtraub_axnode75 = -80
#		h.vshift_parak75 = 15
#		h.gbar_km = self.soma_gbar_km
#		h.vshift_sca = self.tuft_vshift_sca
#		h.nai0_na_ion = 4
#		h.nao0_na_ion = 151
	
	def apply_biophysics(self):
		self.sectionreferences = []
		for s,sec in enumerate(self.soma):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='soma',size=self.sizes['soma'][s]))
		for s,sec in enumerate(self.basal):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='basal',size=self.sizes['basal'][s]))
		for s,sec in enumerate(self.apical):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='apical',size=self.sizes['apical'][s]))
		for s,sec in enumerate(self.tuft):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='tuft',size=self.sizes['tuft'][s]))
		for s,sec in enumerate(self.node):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='node',size=self.sizes['node'][s]))
		for s,sec in enumerate(self.paranode1):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='paranode1',size=self.sizes['paranode1'][s]))
		for s,sec in enumerate(self.paranode2):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='paranode2',size=self.sizes['paranode2'][s]))
		for s,sec in enumerate(self.internode):
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='internode',size=self.sizes['internode'][s]))
		for s,sec in enumerate(self.bouton):
#			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='node'))
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='bouton',size=self.sizes['bouton'][s]))
		for s,sec in enumerate(self.interbouton):
#			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='node'))
			self.sectionreferences.append(self.assign_biophysics_to_section(sec,type='interbouton',size=self.sizes['interbouton'][s]))
	
	def sort_section_types(self,sectionlist,types_fname,lengths_fname):
		section_types = self.load_section_types(types_fname)
		section_lengths = self.load_section_lengths(lengths_fname)
		self.sizes = dict(zip(['soma','basal','apical','tuft','node','paranode1','paranode2','internode','bouton','interbouton'],[[],[],[],[],[],[],[],[],[],[]]))
		dels = []
		for s,sec in enumerate(section_types):
#			if sec == 'bouton' or sec == 'interbouton':
##				del(sectionlist[s])
#				dels.append(s)
#				continue
			self.all.append(sectionlist[s])
			if sec == 'soma':
				self.soma.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
			if sec == 'basal':
				self.basal.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
			if sec == 'apical':
				self.apical.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
			if sec == 'tuft':
				self.tuft.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
			if sec == 'node':
#				if self.node == []:
#					self.hillock.append(sectionlist[s])
#				if len(self.node)==1:
#					self.iseg.append(sectionlist[s])
				self.node.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
			if sec == 'paranode1':
#				if self.paranode1 == []:
#					self.hillock.append(sectionlist[s])
#				if len(self.paranode1)==1:
#					self.iseg.append(sectionlist[s])
				self.paranode1.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
			if sec == 'paranode2':
#				if self.paranode2==[]:
#					self.hillock.append(sectionlist[s])
#				if len(self.paranode2)==1:
#					self.iseg.append(sectionlist[s])
				self.paranode2.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
			if sec == 'internode':
#				if self.internode==[]:
#					self.hillock.append(sectionlist[s])
#				if len(self.internode)==1:
#					self.iseg.append(sectionlist[s])
				self.internode.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
			if sec == 'bouton':
				self.bouton.append(sectionlist[s])
#				self.interbouton.append(sectionlist[s])
				self.sizes[sec].append(section_lengths[s])
			if sec == 'interbouton':
				self.bouton.append(sectionlist[s])
#				self.interbouton.append(sectionlist[s])
				self.sizes['bouton'].append(section_lengths[s])			
#		for d in dels:
#			sectionlist[d] = None
#		self.node = self.node[2:]
#		self.paranode1 = self.paranode1[2:]
#		self.paranode2 = self.paranode2[2:]
#		self.internode = self.internode[2:]
	
	def load_section_types(self,fname):
		with open(fname,'r') as f:
			rows = f.readlines()
		
		return([item.strip().split(',')[-1] for item in rows[1:]])
	
	def load_section_lengths(self,fname):
		with open(fname,'r') as f:
			rows = f.readlines()
		
		return([float(item.strip().split(',')[-1]) for item in rows[1:]])
	
	def pretty_print_properties(self,section,sec_type):
		print('\nSection Type: '+sec_type)
		print('Section Ra  : '+str(section.Ra)+' Ohm-cm')
		print('Section Cm  : '+str(section.cm)+' uF/cm^2')
		print('Section diam: '+str(section.diam)+' um')
		print('Section Len : '+str(section.L)+' um')
		print('Section xg  : '+str(section.xg[0]))
		print('Section xc  : '+str(section.xc[0])+' uF/cm^2\n')
	
	def print_passive_properties_by_section_type(self):
		if self.soma != []:
			self.pretty_print_properties(self.soma[0],'Soma')
		if self.hillock != []:
			self.pretty_print_properties(self.hillock[0],'Hillock')
		if self.iseg != []:
			self.pretty_print_properties(self.iseg[0],'Initial Segment')
		if self.node != []:
			self.pretty_print_properties(self.node[0],'Node of Ranvier')
		if self.paranode1 != []:
			self.pretty_print_properties(self.paranode1[0],'1st Paranode')
		if self.paranode2 != []:
			self.pretty_print_properties(self.paranode2[0],'2nd Paranode')
		if self.internode != []:
			self.pretty_print_properties(self.internode[0],'Internode')
		if self.bouton != []:
			self.pretty_print_properties(self.bouton[0],'Bouton')
		if self.interbouton != []:
			self.pretty_print_properties(self.interbouton[0],'Interbouton')
		if self.basal != []:
			self.pretty_print_properties(self.basal[0],'Basal Dendrite')
		if self.tuft != []:
			self.pretty_print_properties(self.tuft[0],'Tuft Dendrite')
		if self.apical != []:
			self.pretty_print_properties(self.apical[0],'Apical Dendrite')
	
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
#			print('requested fiber diameter is out of range')
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
#		interlen = int((dep_vars[6]-1-(2*3)-(2*dep_vars[7]))/6)
#		if interlen < 0.0:
#			interlen = 1.0
		interlen = int((fiberD*65)-1-(2*3)-(2*dep_vars[7]))
		return([dep_vars[3],dep_vars[4],dep_vars[5],dep_vars[2],dep_vars[5],dep_vars[4]],[1,3,int(dep_vars[7]),interlen,int(dep_vars[7]),3],int(dep_vars[8]))
	
	def assign_biophysics_to_section(self,sec,rhoa_=0.7e6,type='node',size=1):
		"""
		This class method takes a NEURON section, with an empirically validated rhoa and assigns McIntyre, Richardson, & Grill (2002) biophysics according to section type (including types: 'node','paranode1','paranode2','internode','unmyelinated') and according to Bahl et al. for ('soma','basal','apical','tuft)
		
		"""
		if type == 'hillock':
			secref = self.assign_hillock_biophysics(sec,rhoa_,0.002,sectionlen=size)
		if type == 'iseg':
			secref = self.assign_initial_segment_biophysics(sec,rhoa_,0.002,sectionlen=size)
		if type=='node':
			secref = self.assign_nodal_biophysics(sec,rhoa_,0.002,self.outerdiam,self.nodeD,sectionlen=size)
		if type=='paranode1':
			secref = self.assign_paranode1_biophysics(sec,rhoa_,0.002,self.outerdiam,self.para1D,self.nl,sectionlen=size)
		if type=='paranode2':
			secref = self.assign_paranode2_biophysics(sec,rhoa_,0.004,self.outerdiam,self.para2D,self.nl,sectionlen=size)
		if type=='internode':
			secref = self.assign_internode_biophysics(sec,rhoa_,0.004,self.outerdiam,self.interD,self.nl,sectionlen=size)
		if type=='bouton':
			secref = self.assign_bouton_biophysics(sec,rhoa_,0.002,sectionlen=size)
#			secref = self.assign_nodal_biophysics(sec,rhoa,0.002)
		if type=='interbouton':
			secref = self.assign_interbouton_biophysics(sec,rhoa_,0.002,sectionlen=size)
#			secref = self.assign_nodal_biophysics(sec,rhoa,0.002)		
		if type=='soma':
			secref = self.assign_soma_biophysics(sec,rhoa_,0.002,sectionlen=size)
		if type=='basal':
			secref = self.assign_basal_biophysics(sec,rhoa_,0.002,sectionlen=size)
		if type=='apical':
			secref = self.assign_apical_biophysics(sec,rhoa_,0.002,sectionlen=size)
		if type=='tuft':
			secref = self.assign_tuft_biophysics(sec,rhoa_,0.002,sectionlen=size)
		
		h.pop_section()
		return(secref)
	
	def fixnseg(self):
		h.xopen("fixnseg.hoc")
		h.geom_nseg()
	
	def lambda_f(self,sec):
		return((100000)*(sec.diam/(4*np.pi*100*sec.Ra*sec.cm))**0.5)
	
	def dlambda(self,sec):
		return(int((sec.L/(0.1*self.lambda_f(sec)+0.9))/2)*2+1)
	
	def calc_ra(self,diameter):
		return(5497.787143782139*(1/(np.pi*(diameter/2)**2)))
	
	def assign_soma_biophysics(self,sec,rhoa,space_p,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.Ra = 600.0
		sec.cm = self.axosomatic_cm 
		sec.nseg = self.dlambda(sec)
		sec.insert('pas')
		sec.g_pas=0.1/15000
		sec.e_pas=self.e_pas
		sec.insert('ih')
		sec.insert('nat')
		sec.ena = 55
		sec.vshift_nat=10
		sec.insert('kfast')
		sec.gbar_kfast = self.soma_gbar_kfast		
		sec.ek = -80
		sec.insert('kslow')
		sec.gbar_kslow = self.soma_gbar_kslow
		sec.insert('nap')
		sec.insert('km')
		sec.gbar_nat = self.soma_gbar_nat
		sec.gbar_nap = self.soma_gbar_nap
		sec.insert('extracellular')
		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
		sec.xg[0]=1e10 
		sec.xg[1]=1e10 
		sec.xc[0]=0
		sec.xc[1]=0
		return(secref)
	
	def assign_hillock_biophysics(self,sec,rhoa,space_p,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.diam = (self.hillock[0].diam+30)/2.0
		sec.insert('pas')
		sec.Ra = rhoa/20000
		sec.g_pas=1./15000
		sec.e_pas=self.e_pas
		sec.insert('ih')
		sec.insert('nat')
		sec.gbar_nat = self.hillock_gbar_nat
		sec.ena = 55
		sec.vshift_nat=10
		sec.cm = self.axosomatic_cm 
		return(secref)
	
	def assign_initial_segment_biophysics(self,sec,rhoa,space_p,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.insert('pas')
		sec.L = sectionlen
		sec.Ra = rhoa/20000
		sec.g_pas=1./15000
		sec.e_pas=self.e_pas
		sec.insert('ih')
		sec.insert('nat')
		sec.gbar_nat = self.iseg_gbar_nat
		sec.ena = 55
		sec.vshift2_nat=self.iseg_vshift2_nat
		sec.cm = self.axosomatic_cm 
		return(secref)
	
	def assign_basal_biophysics(self,sec,rhoa,space_p,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.Ra = 125.0
		sec.cm = 0.9
		sec.nseg = self.dlambda(sec)
		sec.insert('pas')
		sec.g_pas=1./15000
		sec.e_pas=self.e_pas
		sec.insert('ih')
		sec.ehd_ih=-47
		sec.gbar_ih = self.basal_gbar_ih
		sec.insert('extracellular')
		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
		sec.xg[0]=1e10 
		sec.xg[1]=1e10 
		sec.xc[0]=0
		sec.xc[1]=0
		return(secref)
	
	def assign_apical_biophysics(self,sec,rhoa,space_p,sectionlen):
		sec.push()
		dist_reset = h.distance()
		distance=h.distance(self.soma[0](0.5),sec=sec)
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.Ra = 125.0
		sec.cm = 2*self.axosomatic_cm
		sec.cm = 0.9
		sec.nseg = self.dlambda(sec)
		sec.insert('pas')
		sec.g_pas=2./15000
		sec.e_pas=self.e_pas
		sec.insert('ih')
		sec.ehd_ih=-47
		sec.insert('nat')
		sec.ena = 55
		sec.vshift_nat=10
		sec.insert('kfast')
		sec.ek = -80
		sec.gbar_kfast = self.soma_gbar_kfast * np.exp(-distance/self.decay_kfast)
		sec.insert('kslow')
		sec.gbar_kslow = self.soma_gbar_kslow * np.exp(-distance/self.decay_kslow)
		sec.m_nat = (sec.gbar_nat-self.soma_gbar_nat)/distance
		sec.gbar_nat = sec.m_nat*distance + self.soma_gbar_nat
		sec.insert('extracellular')
		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
		sec.xg[0]=1e10 
		sec.xg[1]=1e10 
		sec.xc[0]=0
		sec.xc[1]=0
		return(secref)
	
	def assign_tuft_biophysics(self,sec,rhoa,space_p,sectionlen):
		sec.push()
		dist_reset = h.distance()
		distance=h.distance(self.soma[0](0.5),sec=sec)
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.Ra = 125.0
		sec.cm = 2*self.axosomatic_cm
		sec.nseg = self.dlambda(sec)
		sec.insert('pas')
		sec.g_pas=2./15000
		sec.e_pas=self.e_pas
		sec.insert('ih')
		sec.ehd_ih=-47
		sec.gbar_ih = self.tuft_gbar_ih
		sec.insert('nat')
		sec.gbar_nat = self.tuft_gbar_nat
		sec.ena = 55
		sec.vshift_nat=10
		sec.insert('kfast')
		sec.ek = -80
		sec.gbar_kfast = self.soma_gbar_kfast * np.exp(-distance/self.decay_kfast)
		sec.insert('kslow')
		sec.gbar_kslow = self.soma_gbar_kslow * np.exp(-distance/self.decay_kslow)
		sec.insert('sca')
		sec.gbar_sca = self.tuft_gbar_sca
		sec.insert('kca')
		sec.gbar_kca = self.tuft_gbar_kca
		h.ion_style("ca_ion",0,1,0,0,0)
		sec.insert('cad')
		sec.m_nat = (sec.gbar_nat-self.soma_gbar_nat)/distance
		sec.eca = 140
		sec.insert('extracellular')
		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
		sec.xg[0]=1e10 
		sec.xg[1]=1e10 
		sec.xc[0]=0
		sec.xc[1]=0
		return(secref)
	
	def assign_bouton_biophysics(self,sec,rhoa,space_p,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.nseg = 1
		sec.diam = 2
		sec.Ra = rhoa/10000
		sec.cm = 1.0
		sec.insert('na12')
		sec.gbar_na12 = 1000.0
		sec.insert('axnodeX')
		sec.gnapbar_axnodeX = 0.0024378560807969537
		sec.gnabar_axnodeX = 4.24219598883859
		sec.gkbar_axnodeX = 0.12977029013875638
		sec.gl_axnodeX = 0.007
		sec.ena_axnodeX = 50.0
		sec.ek_axnodeX = -90.0
		sec.el_axnodeX = -90.0
		sec.insert('extracellular')
		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
		sec.xg[0]=1e10 
		sec.xg[1]=1e10 
		sec.xc[0]=0
		sec.xc[1]=0
		return(secref)
	
	def assign_interbouton_biophysics(self,sec,rhoa,space_p,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.nseg = 1
		sec.diam = 2
		sec.Ra = rhoa/10000
		sec.cm = 1.0
		sec.insert('na12')
		sec.gbar_na12 = 1000.0
		sec.insert('axnodeX')
		sec.gnapbar_axnodeX = 0.00370109160245544
		sec.gnabar_axnodeX = 3.068875056920868
		sec.gkbar_axnodeX = 0.10527073936782891
		sec.gl_axnodeX = 0.007
		sec.ena_axnodeX = 50.0
		sec.ek_axnodeX = -90.0
		sec.el_axnodeX = -90.0
		sec.insert('extracellular')
		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((sec.diam/2)+space_p)**2)-((sec.diam/2)**2)))
		sec.xg[0]=1e10 
		sec.xg[1]=1e10 
		sec.xc[0]=0
		sec.xc[1]=0
		return(secref)
	
	def assign_nodal_biophysics(self,sec,rhoa,space_p,outerdiam,innerdiam,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.nseg = 1
		sec.diam = innerdiam
		sec.Ra = rhoa/10000
		sec.cm = 2.0
		sec.insert('axnode')
		sec.gnapbar_axnode = 0.005
		sec.gnabar_axnode = 3.0
		sec.gkbar_axnode = 0.08
		sec.gl_axnode = 0.007
		sec.ena_axnode = 50.0
		sec.ek_axnode = -90.0
		sec.el_axnode = -90.0
		sec.insert('extracellular')
		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
		sec.xg[0]=1e10 
		sec.xg[1]=1e10 
		sec.xc[0]=0
		sec.xc[1]=0
		return(secref)
	
	def assign_paranode1_biophysics(self,sec,rhoa,space_p,outerdiam,innerdiam,nl,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.nseg = 1
		sec.diam = outerdiam
#		sec.Ra = rhoa*(1/(innerdiam/outerdiam)**2)/10000
		sec.Ra = rhoa/10000
		sec.cm = 2*innerdiam/outerdiam
		sec.insert('pas')
		sec.g_pas = 0.001*innerdiam/outerdiam
		sec.e_pas = h.v_init
		sec.insert('extracellular')
		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
		sec.xg[0]=0.001/(nl*2)
		sec.xg[1]=0.001/(nl*2)
		sec.xc[0]=0.1/(nl*2)
		sec.xc[1]=0.1/(nl*2)
		return(secref)
	
	def assign_paranode2_biophysics(self,sec,rhoa,space_p,outerdiam,innerdiam,nl,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.nseg = 1
		sec.diam = outerdiam
#		sec.Ra = rhoa*(1/(innerdiam/outerdiam)**2)/10000
		sec.Ra = rhoa/10000
		sec.cm = 2.0*innerdiam/outerdiam
		sec.insert('pas')
		sec.g_pas = 0.0001*innerdiam/outerdiam
		sec.e_pas = h.v_init
		sec.insert('extracellular')
		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
		sec.xg[0]=0.001/(nl*2)
		sec.xg[1]=0.001/(nl*2)
		sec.xc[0]=0.1/(nl*2)
		sec.xc[1]=0.1/(nl*2)
		return(secref)
	
	def assign_internode_biophysics(self,sec,rhoa,space_p,outerdiam,innerdiam,nl,sectionlen):
		sec.push()
		secref = h.SectionRef()
		sec.L = sectionlen
		sec.nseg = 3
		sec.diam = outerdiam
#		sec.Ra = rhoa*(1/(innerdiam/outerdiam)**2)/10000
		sec.Ra = rhoa/10000
		sec.cm = 2.0*innerdiam/outerdiam
		sec.insert('pas')
		sec.g_pas = 0.0001*innerdiam/outerdiam
		sec.e_pas = h.v_init
		sec.insert('extracellular')
		sec.xraxial[0]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
		sec.xraxial[1]=(rhoa*.01)/(np.pi*((((innerdiam/2)+space_p)**2)-((innerdiam/2)**2)))
		sec.xg[0]=0.001/(nl*2)
		sec.xg[1]=0.001/(nl*2)
		sec.xc[0]=0.1/(nl*2)
		sec.xc[1]=0.1/(nl*2)
		return(secref)
	
	def modify_biophysics(self,bphys):
		for section in self.interbouton:
			section.gkbar_axnodeX = bphys['k_ib']
			section.gnabar_axnodeX = bphys['na_ib']
			section.gnapbar_axnodeX = 0.0
			section.gl_axnodeX = bphys['gl_ib']
			section.gbar_na12 = bphys['gbar_ib']
		
		for section in self.bouton:
			section.gkbar_axnodeX = bphys['k_b']
			section.gnabar_axnodeX = bphys['na_b']
			section.gnapbar_axnodeX = 0.0
			section.gl_axnodeX = bphys['gl_b']
			section.gbar_na12 = bphys['gbar_b']
		
		for section in self.node:
			section.gkbar_axnode = bphys['k_nb']
			section.gnabar_axnode = bphys['na_nb']
			section.gnapbar_axnode = bphys['gnap_nb']
			section.gl_axnode = bphys['gl_nb']
		
		h.amA_axnodeX = bphys['amA']
		h.amB_axnodeX = bphys['amB']
		h.amC_axnodeX = bphys['amC']
		h.bmA_axnodeX = bphys['bmA']
		h.bmB_axnodeX = bphys['bmB']
		h.bmC_axnodeX = bphys['bmC']
		h.ahA_axnodeX = bphys['ahA']
		h.ahB_axnodeX = bphys['ahB']
		h.ahC_axnodeX = bphys['ahC']
		h.asA_axnodeX = bphys['asA']
		h.asB_axnodeX = bphys['asB']
		h.asC_axnodeX = bphys['asC']
		
		h.amA_axnode = bphys['amA_n']
		h.amB_axnode = bphys['amB_n']
		h.amC_axnode = bphys['amC_n']
		h.bmA_axnode = bphys['bmA_n']
		h.bmB_axnode = bphys['bmB_n']
		h.bmC_axnode = bphys['bmC_n']
		h.ahA_axnode = bphys['ahA_n']
		h.ahB_axnode = bphys['ahB_n']
		h.ahC_axnode = bphys['ahC_n']
		h.asA_axnode = bphys['asA_n']
		h.asB_axnode = bphys['asB_n']
		h.asC_axnode = bphys['asC_n']
		return()
