'''Internal State Regression method (ISR) for UCG modeling.

Description
-----------

The ``cgisr`` command determines a variationally optimal and local definition of UCG state probaiblities according to the UCG-ISR method (doi.org/10.1063/5.0244427). The ISR method requires a reference state assignment and then minimizes the cross-entropy for the local UCG state definition. Requires PyTorch and PyTorch Geometric for use. The output of cgisr is a file `isr.dat` which contains the optimal UCG state definition for use with the RLECUSTOM class. The bead is designated as `VAL` and should be changed by the user after generation.

Usage
-----

Syntax of running ``cgisr`` command ::

    General arguments:
      -h, --help            show this help message and exit
      -v L, --verbose L     screen verbose level (default: 0)

    Required arguments:
      --traj_class          LAMMPS trajecyory where types dictate state assignment. Note that ISR only handles a binary set of states for one CG bead type. Consequently, ISR should be conducted independently for each UCG bead. 
       --phi                filter for distance in message passing
       --psi                scaling factor for messages
       --alpha              filter which determines binary state assignment
       --beta               scaling factor for state assignment
       --chi                learning rate for ISR
       --epochs             number of epochs
       --batch_size         batch_size


    Optional arguments:
      --chi                 Chi-value (default: 1.0)
      --table               prefix of table names (default: )
      --maxiter             maximum iterations (default: 20)
      --restart file        restart file (default: restart)
      --models file         initial model parameters (default: model.txt)
      --optimizer name,args Define optimizer (default: [])

'''


from mscg import *
import numpy as np

try:
    import torch
    from torch.optim import Adam
except:
    screen.fatal("PyTorch must be installed to use cgisr.")
try:
    import torch_geometric
    from torch_geometric.nn import MessagePassing
    from torch_geometric.loader import DataLoader
    from torch_geometric.data import Data
except:
    screen.fatal("PyTorch Geometric must be installed to use cgisr.")

import copy

BASE=torch.Tensor([1e-6]).detach()

class state_model(torch.nn.Module):
    def __init__(self, phi, psi, alpha, beta, natoms, cutoff):
        super(state_model, self).__init__()
        self.phi = phi
        self.psi = psi
        self.alpha = alpha
        self.beta = beta
        self.natoms = natoms
        self.cutoff = cutoff
        self.block = InteractionBlock(cutoff=cutoff)
        self.batch_size = 1
        
    def forward(self, data):

        x = torch.ones((self.natoms*self.batch_size,1))
        rho = self.block(x, data.edge_index, data.edge_weight, self.phi, self.psi)
        p1 = 0.5 * (1.0 - torch.tanh((rho - self.alpha) / (self.beta * self.alpha)))
        p1= torch.reshape(p1, (-1,))

        return p1, 1.0 - p1

#this is a simple 1-layer GNN

class InteractionBlock(torch.nn.Module):
    def __init__(self, cutoff):
        super().__init__()

        self.conv = CFConv(cutoff)

    def forward(self, x, edge_index, edge_weight, phi, alpha):
        x = self.conv(x, edge_index, edge_weight, phi, alpha)

        return x

#run one round of message passing here

class CFConv(MessagePassing):
    def __init__(self, cutoff):
        super().__init__(aggr='add')
        self.cutoff = cutoff

    def forward(self, x, edge_index, edge_weight, phi, psi):
        
        W =  0.5 * (1.0 - torch.tanh((edge_weight - phi) / (psi * phi)))

        n = self.propagate(edge_index, x=x, W=W)

        return n

    def message(self, x_j, W):

        return  W.view(-1, 1)*x_j

#utilize cross-entropy loss for logisitic regression

def loss_function(y, p1, p0, batch_size, mask, natoms):

    loss = 0.0
    loss = -(1.0 /batch_size) *(y * torch.log(p0+BASE) +  (1.0 - y) * torch.log(p1+BASE))
    loss *= mask

    return loss.sum()

def main(*args, **kwargs):

    models.empty()

    # parse argument

    desc = 'Run logistic regression to determine ideal set of local density dependant states from predetermined state classification ' 

    parser = CLIParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@', add_help=False)

    group = parser.add_argument_group('General arguments')
    group.add_argument("-h", "--help", action="help", help="show this help message and exit")
    group.add_argument("-v", "--verbose", metavar='L', type=int, default=0, help="screen verbose level")

    group = parser.add_argument_group('Required arguments')
    group.add_argument("--top",  metavar='file', action=TopAction, help="topology file", required=True)

    group = parser.add_argument_group('Optional arguments')
    group.add_argument("--names",  metavar='', type=str, help="comma separated atom names (needed when using LAMMPS data file for topology)")
    group.add_argument("--traj_class", metavar='file[,args]', action=TrajReaderAction, default=[], help="trajectory to obtain classes for ISR")
    group.add_argument("--traj", metavar='file[,args]', action=TrajReaderAction, default=[], help="trajectory for analysis")
    group.add_argument("--cut", metavar='', type=float, default=24.0, help="cut-off for pair interactions")
    group.add_argument("--exclude", metavar='', type=str, default="111", help="exclude 1-2, 1-3, 1-4 bonding neighbors in the pair-list")
    group.add_argument("--save",  metavar='', type=str, default="model", help="file name for model output")
    
    group.add_argument("--phi",  metavar='', type=float, default=1.0, help="cutoff for distance in message passing")
    group.add_argument("--psi",  metavar='', type=float, default=0.1, help="scaling factor for messages")
    group.add_argument("--alpha",  metavar='', type=float, default=1.0, help="cutoff which determines binary state assignment")
    group.add_argument("--beta",  metavar='', type=float, default=0.1, help="scaling factor for state assignment")
    group.add_argument("--chi",  metavar='', type=float, default=0.1, help="learning rate for ISR")
    group.add_argument("--epochs",  metavar='', type=int, default=1, help="number of epochs")
    group.add_argument("--batch_size",  metavar='', type=int, default=32, help="batch_size")

    group.add_argument("--pair",  metavar='[key=value]', action=ModelArgAction, help=ModelArgAction.help('pair'), default=[])
    group.add_argument("--bond",  metavar='[key=value]', action=ModelArgAction, help=ModelArgAction.help('bond'), default=[])
    group.add_argument("--angle", metavar='[key=value]', action=ModelArgAction, help=ModelArgAction.help('angle'), default=[])
    group.add_argument("--dihedral", metavar='[key=value]', action=ModelArgAction, help=ModelArgAction.help('dihedral'), default=[])

    group.add_argument("--ucg", metavar='[key=value]', action=UCGArgAction, help=UCGArgAction.help(), default=None)
    group.add_argument("--ucg-wf", metavar='[key=value]', action=WFArgAction, help=WFArgAction.help(), default=[])

    if len(args)>0 or len(kwargs)>0:
        args = parser.parse_inline_args(*args, **kwargs)
    else:
        args = parser.parse_args()

    screen.info("Check topology ... ")

    if args.names is not None:
        args.top.reset_names(args.names.split(','))

    # prepare lists

    screen.info("Build pair and bonding list-based algorithm ...")
    plist = PairList(cut = args.cut, binsize = args.cut * 0.5)
    plist.init(args.top.types_atom, args.top.linking_map(*([bit=='1' for bit in args.exclude[:3]])))
    blist = BondList(
        args.top.types_bond, args.top.bond_atoms,
        args.top.types_angle, args.top.angle_atoms,
        args.top.types_dihedral, args.top.dihedral_atoms)
    UCG.init(plist, blist)

    # build up tables

    [pair.setup(args.top, plist) for pair in args.pair]
    [bond.setup(args.top, blist) for bond in args.bond]
    [angle.setup(args.top, blist) for angle in args.angle]
    [dihedral.setup(args.top, blist) for dihedral in args.dihedral]

    for model in models.items:
        screen.info(" ".join([str(i) for i in
            ["Model:", model.style, model.name, "T-" + str(model.tid)]]))
    nframes = 0
    for reader in TrajBatch(args.traj, natoms = args.top.n_atom, cut = plist.cut):
        nframes+=1

    natoms = args.top.n_atom
    max1 = int(args.batch_size/2)
    data = []

    for reader in TrajBatch(args.traj, natoms = args.top.n_atom, cut = plist.cut):
        if reader.nread == 1:
            plist.setup_bins(reader.traj.box)

        TIMER.click('io')
        TIMER.click('pair', plist.build(reader.traj.x))
        TIMER.click('bond', blist.build(reader.traj.box, reader.traj.x))
        for page in plist.pages(0, index=True):

            edge_index = copy.deepcopy(np.concatenate([page.index, page.index[[1,0]]], axis=1))
            edge_weight = copy.deepcopy(np.concatenate([page.r, page.r]))
            classes = reader.traj.t

        t=torch.linspace(0, natoms-1, natoms)cd 
        idx = torch.randperm(t.shape[0])
        mask = torch.zeros(natoms)
        t = t[idx].view(t.size())
        n0 = 0
        n1 = 0
        for i in t:
            ii = int(i)
            if classes[ii] == 0 and n0 < max1:
                n0 += 1
                mask[ii] += 1
            elif classes[ii] == 1 and n1 < max1:
                n1 += 1
                mask[ii] += 1
            elif n0 >= max1 and n1 >= max1:
                break
        
        d = Data(edge_index=torch.from_numpy(edge_index), edge_weight = torch.from_numpy(edge_weight), y=torch.from_numpy(copy.deepcopy(classes)), mask=mask, num_nodes=edge_index.max() + 1)
        data.append(d)

    data_train_loader = DataLoader(data, batch_size=1, shuffle=False,drop_last=True)
    alpha = torch.Tensor([args.alpha])
    phi = torch.Tensor([args.phi])
    beta = torch.Tensor([args.beta])
    psi = torch.Tensor([args.psi])
    chi = args.chi
    phi.requires_grad=True
    alpha.requires_grad=True
    beta.requires_grad=True
    psi.requires_grad=True
    natoms = args.top.n_atom
    optimizer = Adam([alpha, phi, psi, beta], lr=chi)
    smodel = state_model(phi, psi, alpha, beta, natoms, args.cut)

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        loss = 0.0
        cross_entropy = 0.0

        for df in data_train_loader:
            p1, p0 = smodel(df)
            loss = loss_function(df.y,p1, p0, args.batch_size, df.mask, natoms)
            cross_entropy += loss.item() / nframes

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("LOSS: %s" %cross_entropy)

    if args.save == 'return':
        return phi, alpha, psi, beta
    else:
        isr_out = open('isr.dat', 'w')
        isr_out.write("--ucg-wf RLECUSTOM,I=VAL,J=VAL,high=Near,low=Far,rth=%s,wth=%s,psi=%s,beta=%s" %(str(phi.item())[:4], str(alpha.item())[:4], str(psi.item())[:4], str(beta.item())[:4]))
        isr_out.close()
    
if __name__ == '__main__':
    main()
