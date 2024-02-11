from neuron import h
import numpy as np
import networkx as nx

class ConductionVelocity():
	def __init__(self):
		pass
	
	def build_network(self,sections):
		g = nx.Graph()
		for sec in sections:
			child = h.SectionRef(sec=sec)
			if child.has_parent():
				parent = child.parent
				if parent in sections:
					g.add_edge(sections.index(parent),sections.index(sec),weight=parent.L/2.0+sec.L/2.0)
		return(g)
	
	def sort_data_by_cellall(self,data,cell):
		newdata = []
		for sec in cell.all:
			if sec in cell.node:
				newdata.append(data[cell.node.index(sec)])
			if sec in cell.paranode1:
				newdata.append(data[len(cell.node)+cell.paranode1.index(sec)])
			if sec in cell.paranode2:
				newdata.append(data[len(cell.node)+len(cell.paranode1)+cell.paranode2.index(sec)])
			if sec in cell.internode:
				newdata.append(data[len(cell.node)+len(cell.paranode1)+len(cell.paranode2)+cell.internode.index(sec)])
			if sec in cell.bouton:
				newdata.append(data[len(cell.node)+len(cell.paranode1)+len(cell.paranode2)+len(cell.internode)+cell.bouton.index(sec)])
			if sec in cell.interbouton:
				newdata.append(data[len(cell.node)+len(cell.paranode1)+len(cell.paranode2)+len(cell.internode)+len(cell.bouton)+cell.interbouton.index(sec)])
		return(newdata)
	
	def get_sptimes(self,cell,data,time_step):
		newdata = self.sort_data_by_cellall(data,cell)
		sptimes = []
		for sec in cell.all:
			sptimes.append(self.calc_time_to_first_spike(newdata[cell.all.index(sec)]))
			if sptimes[-1] is not None:
				sptimes[-1]*=time_step
		return(sptimes)
	
	def get_longest_path_in_region(self,cell,graph,region='CCF'):
		paths = []
		if region=='CCF':
			for sec in cell.node:
				paths.append(nx.dijkstra_path(graph,cell.all.index(sec),cell.all.index(cell.node[0])))
			
			maxpath = np.argmax([len(path) for path in paths])
			return(paths[maxpath],self.get_path_length(graph,paths[maxpath]))
		
		if region=='unmyelinated':
			for sec in cell.bouton:
				paths.append(nx.dijkstra_path(graph,cell.all.index(sec),cell.all.index(cell.bouton[0])))
			
			maxpath = np.argmax([len(path) for path in paths])
			return(paths[maxpath],self.get_path_length(graph,paths[maxpath]))
	
	
	def calculate_CVs(self,cell,data,time_step):
		sptimes = self.get_sptimes(cell,data,time_step)
		g = self.build_network(cell.all)
		myelinated = []
		CCFpath,myelinated_len = self.get_longest_path_in_region(cell,g,region='CCF')
		myelinated.append(myelinated_len)
		try:
			myelinated.append(np.abs(sptimes[CCFpath[0]]-sptimes[CCFpath[-1]]))
			if myelinated[-1] == 0:
				myelinated[-1] = None
		except:
			myelinated.append(None)
		try:
			if None in myelinated:
				myelinatedCV = None
			else:
				myelinatedCV = (myelinated[0]*1e-6)/(myelinated[1]*1e-3)
		except:
			myelinatedCV = None
		
		print('Cell CV')
#		print('unmyelinated CV: '+str(unmyelinatedCV)+' m/s')
		print('myelinated CV: '+str(myelinatedCV)+' m/s')
#		return(myelinatedCV,unmyelinatedCV)
		return(myelinatedCV)
	
	def earliest_spike(self,ndat,dt):
		mints = len(ndat[0])
		for d in ndat:
			mx = np.max(d)
			if mx<-10.0:
				continue
			
			dmax = list(d).index(mx)
			if dmax < mints:
				mints=dmax
		
		if mints == len(ndat[0]):
			mints = 0
		
		return(mints*dt)
		
	def check_prestim_slope(self,ndat):
		sum_diff = 0
		for n in ndat:
			sum_diff+=n[int(3.875/h.dt)]-n[int(1.0/h.dt)]
		return(np.abs(2*sum_diff/float(len(ndat))))

	def check_single_spike(self,ndat):
		multiples = []
		for n in ndat:
			multiples.append(self.count_independent_crossings(n))
		
		return(multiples)
	
	def count_independent_crossings(self,ndat):
		positive = [list(ndat).index(i) for i in ndat if i > -10]
		spcount = 0
		for pos in positive:
			if pos-1 or pos+1 not in positive:
				spcount+=1
		
		return(spcount/2)
	
	def return_to_rest(self,ndat):
		last_value = np.mean([dat[-1] for dat in ndat])
		if last_value > -60:
			return(False)
		else:
			return(True)
	
	def calculate_cost(self,myelinatedCV,target_myelinatedCV,dt,earlyspike_threshold,data):
		try:
			ccfcost = np.abs(target_myelinatedCV-myelinatedCV)
		except:
			ccfcost = 100
		
		
		earlyspike_penalty = 0
		first_spike = self.earliest_spike(data,dt)
		if first_spike < earlyspike_threshold:
			earlyspike_penalty+=200
		
		multispike_penalty = 0
		multiples = self.check_single_spike(data)
		for m in multiples:
			if m>1:
				multispike_penalty+=200
				break
		
		cost = earlyspike_penalty+ccfcost
		
		if not self.return_to_rest(data):
			cost = 200
		
		if cost > 200:
			cost = 200
		return(cost)
	
	def get_path_length(self,graph,path):
		testlen = 0
		for p in range(len(path[:-1])):
			testlen+=graph[path[p]][path[p+1]]['weight']
		return(testlen)
	
	def calc_time_to_first_spike(self,sec_data):
		for i in range(len(sec_data)):
			if sec_data[i] > 0.0:
				return(i)
		
		return(None)
