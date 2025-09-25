'''Yaml Generator for OpenMSCG by Ace.

Description
-----------

The ``cgyaml`` command read the user-defined mapping for residues and generate the yaml file used by cgmap. It also generate the CG coordinate file based on the input AA coordinates. 

Mapping of nonprotein moleculeS/residues is based on atom indices (starting from 0) in the molecule/residue.
Example: 2-bead methanol
CH3
[0, 1, 2, 3]
OH
[4, 5]

Proteins are treated as one "molecule". Mapping of proteins is based on absolute protein residue indices (starting from 0).
Example: heterodimer; chain A resid: 1-10; chain B resid: 2-7
# chain A start from 0, do not include this in mapping file
P0
[0, 3, 2, 1]
P1
[4, 9]
P2
[7, 6]
P4
[5, 8]
# chain A end, do not include this in mapping file
# chain B start from 10 because chain A has 10 residues, do not include this in mapping file
P5
[10, 13]
P6
[12, 11]
P7
[14]

Support random order of atom/residue indices, partial mapping (not all the atoms has to be defined in the mapping)

Protein beads are mapped by all the atoms in the given residues

Examples are provided in openmscg/examples/CGYaml/


Usage
-----

Syntax of running ``cgyaml`` command ::

    General arguments:
      -h, --help            show this help message and exit
      -v, --verbose       being noisy

    Required arguments:
       --aa                   AA coordinate file, like .pdb or .gro. Need to contain residue information
       --mapfile              YAML file indexing all mapping files. See examples. 


    Optional arguments:
      --dir                   Working directory
      --yaml                 File name of the output yaml mapping file
      --cg                    File name of the output CG coordinate file
      --prefix              Prefix of the bead names of protein beads
      --loop                 Offset of the protein's repeating pattern. Homomultimer as an example, this is # of amino acids in a chain. Set to -1 to turn off (default: -1)
      --smart               Whether try to map the undefined protein residues with the existing definition. Has to be turned off when --loop is not -1. (default: False)

'''


import yaml, MDAnalysis as mda, numpy as np
import warnings
import argparse
import re

# Global verbose print function placeholder
def vprint(*args, **kwargs):
    pass

def load_mapping_index(index_file):
    with open(index_file, 'r') as f:
        index_data = yaml.safe_load(f)
    return index_data

def main():
	global vprint

	parser = argparse.ArgumentParser(description="A mapping tool to generate coarse-grained structure and mapping file for OpenMSCG")
	parser.add_argument('--dir', type=str, metavar='PATH', help="Working directory", default=".")
	parser.add_argument('--aa', type=str, metavar='file', help="Input all-atom coordinate file", required=True)
	parser.add_argument('--mapfile', type=str, metavar='file', help="YAML file indexing all mapping files", required=True)
	parser.add_argument('--yaml', type=str, metavar='file', help="Output yaml mapping file", default="mapping.yaml")
	parser.add_argument('--cg', type=str, metavar='file', help="Output CG coordinate file", default="CG.pdb")
	parser.add_argument('--prefix', type=str, metavar='string',help="Prefix of the bead names of protein beads", default="")
	parser.add_argument('--loop', type=int, metavar='number', help="Offset of the protein's repeating pattern. Homomultimer as an example, this is # of amino acids in a chain. Set to -1 to turn off", default=-1)
	parser.add_argument('--smart', action="store_true", help="Whether try to map the undefined protein residues with the existing definition. Has to be turned off when --loop is not -1.", default=False)
	parser.add_argument('-v', '--verbose', action='store_true', help="Print verbose output")
	args = parser.parse_args()

	# Define verbose print
	if args.verbose:
		def vprint(*args_, **kwargs_):
			print(*args_, **kwargs_)
	else:
		def vprint(*args_, **kwargs_):
			pass

	# suppress some MDAnalysis warnings when writing PDB files
	warnings.filterwarnings('ignore')

	WrkDir = args.dir+"/"
	CrdFile = args.aa

	# Read mapping file interactively
	vprint("\nYAML Generator for openMSCG by Ace")
	u = mda.Universe(WrkDir+CrdFile)
	vprint("\nRead "+CrdFile+"\n\nPlease specify the mapping file for each type of residues")
	vprint("\nProtein mapping is defined by absolute residue indices (start from 0)\n\nMapping of other residues is defined by atom indices in that residues (start from 0)")

	# Protein mapping is based on absolute residue index (not resid) (start from 0)
	# Mapping of other molecule is based on atomic index of that molecule (start from 0)
	MappingFile = {} # { resname: mapping_file , "protein": protein_mapping_file }
	# Load mapping files
	MappingIndex = load_mapping_index(WrkDir + args.mapfile)
	MappingFile = {}
	for key, val in MappingIndex.items():
		MappingFile[key] = val

	# if len(u.select_atoms("protein")) > 0: 
	# 	tmp = input("\nInput the mapping file for protein. Enter an empty line to skip.\nProtein: ")
	# 	if tmp != "": MappingFile["protein"] = tmp
	# for resname in set(u.select_atoms("not protein").residues.resnames):
	# 	tmp = input("\nInput the mapping file for "+resname+". Enter an empty line to skip.\n"+resname+": ")
	# 	if tmp != "": MappingFile[resname] = tmp

	ProteinCGPrefix = args.prefix # prefix of the bead names of protein
	ProteinLoopOffset = args.loop # -1 is inactivating; the offset of the protein's repeating pattern, for homodimer, this is # of AA in a chain
	ProteinSmartMap = args.smart # experimental, has to be false when ProteinLoopOffset != -1, whether try to map the undefined protein residues with the existing definition
	
	YamlName = args.yaml # Output yaml
	CG_coord_File = args.cg # Output mapped CG coordinate, use PDB if you want to define segments

	# { resname: (segid, seg) , "protein": protein_seg }
	# Res2Seg = {
	# 	"protein": (0, "PROT"),
	# 	"DPPC": (1, "MEMB"), 	"DLIP": (1, "MEMB"), 	"CHL1": (1, "MEMB"),	
	# }

	# Globals
	Residue2Bead = {} # { resname: { (index,index): bead, (index,index): bead } }
	MappingOutput = {"site-types":{}, "system":[]}
	P_flag = True if "protein" in MappingFile.keys() else False # whether provided mappings include protein

	# return the index of x in arr
	def Find(x, arr):
		for i in range(len(arr)):
			if x == arr[i]:
				return i

	# Read mappings
	vprint("\nReading user-defined mapping from files")
	for key,value in MappingFile.items():
		Residue2Bead[key] = {}
		for line in open(WrkDir+value, "r").readlines():
			try:
				# used sorted tupple of indices as key, map index to bead for each residue
				tmp2 = sorted(list(eval(line)))
				Residue2Bead[key][tuple(tmp2)] = tmp 
				# Bead2Index might have repeating beads, doesn't matter after sorted normalization
			except:
				if line != "\n": # disregard empty lines
					tmp = line[:-1] # CG bead name
	if P_flag: 
		for residtup in Residue2Bead["protein"].keys():
			Residue2Bead["protein"][residtup] = ProteinCGPrefix+Residue2Bead["protein"][residtup]

	# Find model residues from coordinate file
	# Model residues are for finding mass, assembling beads
	vprint("\nFinding model residues in the given coordinate file")
	ModelRes = {} # resname: model_residue
	for resn in Residue2Bead.keys():
		if resn == "protein": continue
		ModelRes[resn] = u.select_atoms("resname "+resn).residues[0]

	# Setup CG beads for each model residue
	vprint("\nSetup CG beads for each model residue")
	CGModel = {} # CG model for each model residues, { resname: [[bead, anchor], [bead, anchor]] }
	for resn in Residue2Bead.keys(): # for each residue
		if resn == "protein": 
			protein_res = u.select_atoms("protein").residues
			prot_mapped = [False] * len(protein_res)
			CGModel["protein"] = []
			for i, pres in enumerate(protein_res):
				if prot_mapped[i] == False: # not mapped yet 
					if ProteinLoopOffset != -1: # activate protein loop modeller
						panchor = i//ProteinLoopOffset*ProteinLoopOffset # find the anchor according to protein loop offset
					else: panchor = 0 # use original residue index 
					P_found = False
					for residtup, bd in Residue2Bead["protein"].items():
						if i-panchor in residtup: 
							P_found = True
							CGModel["protein"].append([bd, int(pres.atoms[0].index)]) # residtup is ascending, so this atom is the starting anchor for the bead
							# CGModel["protein"].append([bd, int(pres.atoms[Find("CA", pres.atoms.names)].index)]) # residtup is ascending, so this atom is the starting anchor for the bead
							for resid in residtup: prot_mapped[panchor+resid] = True
							break
					if ProteinSmartMap and not P_found: 
						for residtup, bd in sorted(Residue2Bead["protein"].items()):
							P_segmatch = True # assume the residues can match with a defined CG bead
							for resid in residtup:
								if i+resid-residtup[0] >= len(protein_res) or protein_res.residues[i+resid-residtup[0]].resname != protein_res.residues[resid].resname:
									P_segmatch = False 
									break
							if P_segmatch: # really match
								CGModel["protein"].append([bd, int(pres.atoms[Find("CA", pres.atoms.names)].index)]) # residtup is ascending, so this atom is the starting anchor for the bead
								for resid in residtup: prot_mapped[i+resid-residtup[0]] = True
								break

			CGModel["protein_offset"] = len(protein_res.atoms)
			continue
		# record whether the atom of the residue has been mapped
		mapped = [False] * len(ModelRes[resn].atoms) 
		CGModel[resn] = []
		# try to map every atom, will pass if not found in defined beads
		for i, atom in enumerate(ModelRes[resn].atoms): 
			if mapped[i] == False: # not mapped yet 
				for idxtup, bd in Residue2Bead[resn].items():
					if i in idxtup: 
						CGModel[resn].append([bd, i]) # idxtup is ascending, so this atom is the starting anchor for the bead
						for idx in idxtup: mapped[idx] = True
						break
		CGModel[resn+"_offset"] = len(ModelRes[resn].atoms)
	vprint("\n-----Built CG Model for each residue-----\n")
	vprint(yaml.dump(CGModel, sort_keys=False, default_flow_style=None))
	vprint("-------------------------------------------------\n")
	# input("---------------------------------------------\n")

	# Write system info with CG residues
	vprint("\nWriting system info")
	prior_resn = None # record previous residue in coord file
	P_write = False # whether protein beads have been written
	for res in u.residues:
		if not P_write and P_flag and res in protein_res:
			MappingOutput["system"].append({})
			MappingOutput["system"][-1]["anchor"] = 0 # for protein, CA index always starts from 0
			MappingOutput["system"][-1]["repeat"] = 1
			MappingOutput["system"][-1]["offset"] = CGModel["protein_offset"]
			# copy CG beads (list) from above
			MappingOutput["system"][-1]["sites"] = CGModel["protein"][:] 
			P_write = True # protein has been written
		if res.resname in CGModel.keys(): # the residue has defined mapping
			if res.resname != prior_resn: # a new residue
				MappingOutput["system"].append({})
				MappingOutput["system"][-1]["anchor"] = int(res.atoms[0].index)
				MappingOutput["system"][-1]["repeat"] = 1
				MappingOutput["system"][-1]["offset"] = CGModel[res.resname+"_offset"]
				# copy CG beads (list) from CG model of the residue
				MappingOutput["system"][-1]["sites"] = CGModel[res.resname][:] 
				prior_resn = res.resname
			else:
				MappingOutput["system"][-1]["repeat"] += 1
	vprint("\n-----System info-----\n")
	vprint(yaml.dump(MappingOutput["system"], sort_keys=False, default_flow_style=None))
	vprint("-------------------------------------------------\n")
	# input("-----------Press ENTER to continue-----------\n")

	# Write site-types
	vprint("\nWriting site-types")
	for grp in MappingOutput["system"]:
		for site in grp["sites"]:
			if MappingOutput["site-types"].get(site[0], -1) == -1: # a new CG bead found, site[0] = name of CG site
				MappingOutput["site-types"][site[0]] = {}
				# find the residue that has the bead
				flag = False # whether the bead is found
				P_bd = False # whether the bead is a protein bead
				for resn in Residue2Bead.keys(): 
					for idxtup, bd in Residue2Bead[resn].items():
						if site[0] == bd: 
							flag = True
							if resn == "protein": P_bd = True
							else: P_bd = False
							break
					if flag: break
				if P_bd:
					tmp2 = [] # tmp list for atom indices in this bead
					for i in idxtup:
						tmp2 = tmp2 + [int(x.index) for x in u.residues[i].atoms]
					# tmp2 = [int(u.residues[i].atoms[Find("CA", u.residues[i].atoms.names)].index) for i in idxtup]
					tmp = [i-tmp2[0] for i in tmp2]
				else:
					tmp = [i-idxtup[0] for i in idxtup]
				MappingOutput["site-types"][site[0]]["index"] = tmp[:]
				tmp = [float(u.atoms[i].mass) for i in tmp2] if P_bd else [float(ModelRes[resn].atoms[i].mass) for i in idxtup]
				MappingOutput["site-types"][site[0]]["x-weight"] = tmp[:]
				MappingOutput["site-types"][site[0]]["f-weight"] = [1.0 for i in range(len(tmp))]
	vprint("\n-----site-types-----\n")
	vprint(yaml.dump(MappingOutput["site-types"], sort_keys=False, default_flow_style=None))
	vprint("-------------------------------------------------\n")
	# input("-----------Press ENTER to continue-----------\n")

	# Write final mapping yaml file
	vprint("\nWriting final yaml file")
	with open(WrkDir+YamlName, "w") as f:
		yaml.dump(MappingOutput, f, sort_keys=False, default_flow_style=None)

	# Write mapped CG coord file
	vprint("\nWriting mapped CG coordinate file")

	# get list of atom selection for each type
	def BeadAAIdx(mapping, bead, offset):
		return [ offset+x for x in mapping["site-types"][bead]["index"] ]

	def compress_index_ranges(s):
		# extract index
		nums = sorted(set(int(i) for i in re.findall(r'index (\d+)', s)))
		
		# combine index
		ranges = []
		start = prev = nums[0]
		for n in nums[1:]:
			if n == prev + 1:
				prev = n
			else:
				if start == prev:
					ranges.append(f"index {start}")
				else:
					ranges.append(f"index {start} to {prev}")
				start = prev = n

		if start == prev:
			ranges.append(f"index {start}")
		else:
			ranges.append(f"index {start} to {prev}")
		
		return ' or '.join(ranges)

	CG_coord = []
	CG_top = {"atoms":[], "residues":[], "atom_resindices":[], "residue_segindices": []}
	resid = 0

	Mapping = yaml.safe_load(open(WrkDir+YamlName, "r"))
	Site2Sel = [] # [sel1, sel2, ]
	for grp in Mapping["system"]:
		anchor = grp["anchor"]
		offset = grp["offset"]
		for repeat in range(grp["repeat"]):
			for site in grp["sites"]:
				Site2Sel.append( compress_index_ranges(" or ".join(["index "+str(x) for x in BeadAAIdx(Mapping, site[0], anchor+offset*repeat+site[1])])) )
				tmp_sel = u.select_atoms(Site2Sel[-1])
				CG_coord.append(tmp_sel.center_of_mass())
				CG_top["atoms"].append(site[0])
				CG_top["atom_resindices"].append(resid)
			
			# every repeat is a new residue
			resid += 1
			CG_top["residue_segindices"].append(0)
			if (tmp_sel & u.select_atoms("protein")).n_atoms == tmp_sel.n_atoms:
				CG_top["residues"].append("PRO")
			else:
				CG_top["residues"].append(tmp_sel.resnames[0])

	Segname = ["SYS"]
	# for item in Res2Seg.values():
	# 	if not item[1] in Segname: Segname.append(item[1])
	u2 = mda.Universe.empty(
		n_atoms=len(CG_coord), n_residues=len(CG_top["residues"]), n_segments=len(Segname),
		atom_resindex=CG_top["atom_resindices"][:],
		residue_segindex=CG_top["residue_segindices"][:],
		trajectory=True
	)
	u2.add_TopologyAttr("name", CG_top["atoms"][:])
	u2.add_TopologyAttr("type", CG_top["atoms"][:])
	u2.add_TopologyAttr("resname", CG_top["residues"][:])
	u2.add_TopologyAttr("resid", list(range(1, len(u2.residues)+1)))
	u2.add_TopologyAttr("segid", Segname[:])
	u2.atoms.positions = np.copy(np.array(CG_coord))
	u2.dimensions = np.copy(u.dimensions)
	u2.atoms.write(WrkDir+CG_coord_File, reindex=True)

	vprint("\nDone! Thanks for using! Have a great day!")

if __name__ == "__main__":
    main()