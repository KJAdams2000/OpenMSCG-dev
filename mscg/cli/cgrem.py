'''Iterative relative-entroy method (REM) for CG modeling.

Description
-----------

The ``cgrem`` command the main engine that drives the REM approach. It applies the iterative approch to minimize the differences of ensemble averages of between derivatives from the reference model and trial models. Therefore, this command calls the ``cgderiv`` command and an MD engine (currently only LAMMPS is supported) in loops.

Usage
-----

Syntax of running ``cgrem`` command ::

    General arguments:
      -h, --help            show this help message and exit
      -v L, --verbose L     screen verbose level (default: 0)

    Required arguments:
      --ref file            checkpoint file for reference model (default: None)
      --cgderiv-arg file    file for cgderiv arguments (default: None)
      --md file             file containing MD command (default: None)

    Optional arguments:
      --chi                 Chi-value (default: 1.0)
      --maxchange           Maximum change of parameters allowed in one iteration (default: 1.0)
      --table               prefix of table names (default: )
      --maxiter             maximum iterations (default: 20)
      --restart file        restart file (default: restart)
      --models file         initial model parameters (default: model.txt)
      --optimizer name,args Define optimizer (default: [])

'''

from mscg import *
from mscg.cli.cgderiv import CGDeriv
from mscg.table import Table
# from mscg.model import models, ModelSymmetryAction

import pickle, os, sys, shutil
from copy import deepcopy

class OptimizerAdam:

    def __init__(self, **kwargs):

        self.lr = float(kwargs.get('lr', 0.01))
        self.beta1 = float(kwargs.get('beta1', 0.9))
        self.beta2 = float(kwargs.get('beta2', 0.999))
        self.eps = float(kwargs.get('eps', 1e-8))
        self.weight_decay = float(kwargs.get('weight_decay', 0.0))
        self.decouple = int(kwargs.get('decouple', 1))               # 1=AdamW (decoupled), 0=coupled (L2 in grad)
        self.amsgrad = int(kwargs.get('amsgrad', 0))                 # 1=use AMSGrad
        self.lr_decay = float(kwargs.get('lr_decay', 0.0))           # simple inverse decay: lr_t = lr/(1+lr_decay*(t-1))
        self.min_lr = float(kwargs.get('min_lr', 0.0))               # lower bound for decayed lr

        self.time = 0
        self.m = {}
        self.v = {}
        self.vhat_max = {}    
        self.t = float(kwargs.get('t', 298.15))
        self.kbt = float(kwargs.get('kbt', 0.001985875 * self.t))
        self.beta = 1.0 / self.kbt
        self.begknots=int(kwargs.get('begknots', 0))
        self.endknots=int(kwargs.get('endknots', 0))      
        self.maxchange = float(kwargs.get('maxchange', 1.0))
        

    def run(self, params, dudl_ref, dudl_mean, dudl_var, **kwargs):
        if dudl_mean is None:
            return params

        self.time += 1

        lr_t = self.lr / (1.0 + self.lr_decay * (self.time - 1))
        if lr_t < self.min_lr:
            lr_t = self.min_lr

        for name, dudl_aa in dudl_ref.items():
            dudl_cg = dudl_mean[name].copy()

            if name not in self.m:
                self.m[name] = np.zeros_like(dudl_cg)
                self.v[name] = np.zeros_like(dudl_cg)
                if self.amsgrad and name not in self.vhat_max:
                    self.vhat_max[name] = np.zeros_like(dudl_cg)

            grad = (dudl_aa - dudl_cg) * self.beta
            if self.weight_decay > 0.0 and not self.decouple:
                grad = grad + self.weight_decay * params[name]

            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grad
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad * grad)

            m_hat = self.m[name] / (1.0 - self.beta1**self.time)
            if self.amsgrad:
                self.vhat_max[name] = np.maximum(self.vhat_max[name], self.v[name])
                v_used = self.vhat_max[name]
            else:
                v_used = self.v[name]
            v_hat = v_used / (1.0 - self.beta2**self.time)

            step = -lr_t * m_hat / (np.sqrt(v_hat) + self.eps)

            if self.weight_decay > 0.0 and self.decouple:
                params[name] *= (1.0 - lr_t * self.weight_decay)


            step[:self.begknots] = 0
            step[(len(step)-self.endknots):] = 0

            # Cap the maximum change
            for i in range(len(step)):
                if abs(step[i]) > self.maxchange:
                    step[i] = self.maxchange if step[i]>0 else -self.maxchange

            # Symmetry action: add dudl to symmetric models
            # Note that this is cross-added. E.g., if A links to B, then B also links to A.
            params[name] += step
            for linked_model in models[name].linksto:
                params[linked_model.name] += step
            
        return params


class OptimizerNR:
    
    def __init__(self, **kwargs):
        
        self.chi = float(kwargs.get('chi', 0.05))
        self.t = float(kwargs.get('t', 298.15))
        self.kbt = float(kwargs.get('kbt', 0.001985875 * self.t))
        self.beta = 1.0 / self.kbt
        self.begknots=int(kwargs.get('begknots', 0))
        self.endknots=int(kwargs.get('endknots', 0))      
        self.maxchange = float(kwargs.get('maxchange', 1.0))
        self.time = 0
        
    def run(self, params, dudl_ref, dudl_mean, dudl_var, **kwargs):

        if dudl_mean is None:
            return params
        
        self.time += 1
        
        for name, dudl_aa in dudl_ref.items():
            dudl_cg = dudl_mean[name].copy()
            var = dudl_var[name].copy()
            for i in range(len(var)):
                if var[i] < 1e-6:
                    var[i] = 1e+6
            
            step = self.chi * (dudl_cg - dudl_aa) / (self.beta * var)
            step[:self.begknots] = 0
            step[(len(step)-self.endknots):] = 0
            for i in range(len(step)):
                if abs(step[i]) > self.maxchange:
                    step[i] = self.maxchange if step[i]>0 else -self.maxchange

            # Symmetry action: add dudl to symmetric models
            # Note that this is cross-added. E.g., if A links to B, then B also links to A.
            params[name] += step
            for linked_model in models[name].linksto:
                params[linked_model.name] += step
            
        return params


class OptimizerAction(argparse.Action):
    
    help = "Name and parameters for the optimizer"
    
    def __call__(self, parser, namespace, values, option_string=None):
        
        if type(values) != str:
            raise ValueError("incorrect format of value for option --optimizer")
        
        segs = values.split(",")
        kwargs = {}
        
        for i in range(1, len(segs)):
            w = [seg.strip() for seg in segs[i].split("=")]
            
            if len(w)==1:
                raise ValueError("incorrect format of value for option --optimizer: " + segs[i])
            else:
                kwargs[w[0]] = w[1]
        
        if segs[0].upper() in ["BUILTIN", "NR"]:
            model_class = OptimizerNR
        elif segs[0].upper() == "ADAM":
            model_class = OptimizerAdam
        else:
            sys.path.append(os.getcwd())
            model_module = importlib.import_module(segs[0])
            model_class = getattr(model_module, "Optimizer")
            
        setattr(namespace, self.dest, model_class(**kwargs))


def main(*args, **kwargs):
    
    # parse argument
    
    desc = 'Iterative REM. ' + doc_root + 'commands/cgrem.html'
    
    parser = CLIParser(description=desc, formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars='@', add_help=False)
    
    group = parser.add_argument_group('General arguments')
    group.add_argument("-h", "--help", action="help", help="show this help message and exit")
    group.add_argument("-v", "--verbose", metavar='L', type=int, default=0, help="screen verbose level")
    
    group = parser.add_argument_group('Required arguments')
    group.add_argument("--ref",  metavar='file', help="checkpoint file for reference model", required=True)
    group.add_argument("--cgderiv-arg", metavar='file', help="file for cgderiv arguments", required=True)
    group.add_argument("--md",  metavar='file', type=str, help="file containing MD command", required=True)
        
    group = parser.add_argument_group('Optional arguments')
    group.add_argument("--table", metavar='', type=str, default='', help="prefix of table names")
    group.add_argument("--maxiter", metavar='', type=int, default=20, help="maximum iterations")
    group.add_argument("--cutoff", metavar='', type=float, default=25, help="distance cutoff")
    
    group.add_argument("--restart", metavar='file', default="restart", help="restart file")
    group.add_argument("--models", metavar='file', default="model.txt", help="initial model parameters")
    group.add_argument("--optimizer", metavar='name,[key=value]', action=OptimizerAction, help=OptimizerAction.help, default=[])
    group.add_argument("--savepath", metavar='dir', default=".", help="directory to save output files")
    group.add_argument("--symmetry", metavar='', type=str, default='', help="symmetry for the model, pairs seperated by ':' will be treated as a group with same values, groups seperated by comma, e.g. 'T0-T1:T0-T2:T0-T3,T4-T4:T5-T5'")

    # parse args
    
    if len(args)>0 or len(kwargs)>0:
        args = parser.parse_inline_args(*args, **kwargs)
    else:
        args = parser.parse_args()
    
    screen.verbose = args.verbose
    screen.info("OpenMSCG CLI Command: " + __name__)
    
    # build cgderiv object
    
    screen.info("Build up calculator for derivatives ...")
    deriv = CGDeriv('@' + args.cgderiv_arg)
    deriv.args.verbose = args.verbose
    deriv.args.save = 'return'
    screen.verbose = args.verbose

    for m in models.items:
        print(m.name)
    print(len(models.items))
    for group in args.symmetry.split(','):
        group = group.split(':')
        for m in group:
            if m not in models:
                raise Exception('Undefined model name in symmetry setting: [%s]' % (m)) 
        for m in group:
            for n in group:
                if m!=n:
                    models[m].linksto.append(models[n])
    
    # read reference model
    
    screen.info("Read reference model ...")
    ref = pickle.load(open(args.ref, 'rb'))
    
    for name, m in ref['models'].items():
        screen.info(" -> Model: " + name)
        
        if models[name] is None:
            screen.fatal("Reference model [%s] is not in targets." % (name))
            
        if m['nparam'] != models[name].nparam:
            screen.fatal("Incorrect number of parameters for reference model %s. (%d != %d)" % (name, models[name].nparam, m['nparam']))
    
    dudl_ref = ref['dudl_mean']
    
    # read tables
    
    targets = {}
    
    with open(args.models, "r") as f:
        rows = [_.strip().split() for _ in f.read().strip().split("\n")]

        for row in rows:
            model_name = row[0]
            model_params = [float(_) for _ in row[5:]]
            m = models[model_name]

            if m is None:
                screen.fatal("Model %s is not defined." % (model_name))

            if m.nparam != len(model_params):
                screen.fatal("Incorrect number of parameters for model %s. (%d != %d)" % (model_name, m.nparam, len(model_params)))
            
            if ('U' not in row[4]) and ('L' not in row[4]) and ('H' not in row[4]) and ('F' not in row[4]) and ('P' not in row[4]) and ('L2' not in row[4]) and ('E' not in row[4]) and ('G' not in row[4]):
                screen.fatal("Incorrect padding option model %s. (%s)" % (model_name, row[4]))
            
            targets[model_name] = {
                'min': float(row[1]),
                'max': float(row[2]),
                'inc': float(row[3]),
                'pad': row[4],
                'init_params': model_params
            }
    
    # read md command
    
    with open(args.md, "r") as f:
        md_cmd = f.readline().strip()
        screen.info("MD command: " + md_cmd)

    os.makedirs(args.savepath, exist_ok=True)

    # restart or init

    if os.path.isfile(os.path.join(args.savepath, args.restart + ".p")):
        screen.info("Restart iterations from checkpoint ...")
        iters     = pickle.load(open(os.path.join(args.savepath, args.restart + ".p"), 'rb'))['iterations']
        params    = iters[-1]['params']
        dudl_mean = iters[-1]['dudl_mean']
        dudl_var  = iters[-1]['dudl_var']
        args.optimizer.time = len(iters) - 1
        # Note that time += 1 is done at first in run()
    else:
        screen.info("Initialize models for iterations ...")
        iters    = []
        params   = {}
        dudl_mean, dudl_var = None, None
        
        for m in models.items:
            if m.name not in targets:
                screen.fatal("Parameters are not initialized for model %s" % (model_name))
        
            params[m.name] = targets[m.name]['init_params']
            
    # iterations
    
    for _ in range(args.maxiter):
        it = len(iters)
        screen.info("* Iteration %d" % (it))

        iter_dir = os.path.join(args.savepath, "iter_%d" % (it))
        os.makedirs(iter_dir, exist_ok=True)
        
        # generate tables

        Checkpoint(os.path.join(args.savepath, args.restart + ".bak"), __file__).update({'dudl_ref': dudl_ref, 'iterations': iters}).dump()

        # Optimizer option
        params = args.optimizer.run(params.copy(), dudl_ref, dudl_mean, dudl_var)
                        
        for m in models.items:
            m.params = np.array(params[m.name])
            screen.info("Generate table [%s] to [%s]" % (m.name, args.table + m.name + '.table'))

            tbl = Table(m, force=False, prefix=os.path.join(iter_dir, args.table))
            if 'Bon' in m.name:
                tbl.compute(targets[m.name]['inc'], args.cutoff, targets[m.name]['inc'])
                tbl.padding_low(targets[m.name]['min'])
                tbl.padding_high(targets[m.name]['max'])
                tbl.u = tbl.u - tbl.u.min()
                tbl.dump_lammps()
            elif 'Ang' in m.name:
                tbl.compute(0.0, 180.0, targets[m.name]['inc'])
                tbl.padding_low(targets[m.name]['min'])
                tbl.padding_high(targets[m.name]['max'])
                tbl.u = tbl.u - tbl.u.min()
                tbl.dump_lammps()
            else:
                tbl.compute(targets[m.name]['min'], targets[m.name]['max'], targets[m.name]['inc'])
                pad = targets[m.name]['pad'].split(':')
                if "L2" in pad: tbl.padding_low2(m.min) #+args.optimizer.begknots*targets[m.name]['inc'])
                elif "LOG" in pad: tbl.padding_low_log(model_min=models[m.name].min)
                elif "L" in pad: tbl.padding_low(models[m.name].min)
                if "H" in pad: tbl.padding_high(targets[m.name]['max'])
                if "F" in pad: tbl.padding_fill_wells()
                if "P" in pad: tbl.padding_low_power(model_min=models[m.name].min)
                if "E" in pad: tbl.padding_low_exp(model_min=models[m.name].min)
                if "G" in pad: tbl.padding_gauss(model_min=m.min)

                tbl.dump_lammps()

        # run MD
        for m in models.items:
            table = m.name + '.table'
            src_file = os.path.join(iter_dir, table)
            dst_file = table
            shutil.copy(src_file, dst_file)

        if os.system(md_cmd) != 0:
            screen.fatal("MD command terminated with failures.")
        
        # calculate derivative
        
        dudl_mean, dudl_var = deriv.process()

        shutil.move('dump.lammpstrj', iter_dir)
        shutil.move('log.lammps', iter_dir)

        with open(os.path.join(iter_dir, "params.npz"), "wb") as f:
            np.savez(f, **params)
        with open(os.path.join(iter_dir, "dudl_mean.npz"), "wb") as f:
            np.savez(f, **dudl_mean)
        with open(os.path.join(iter_dir, "dudl_var.npz"), "wb") as f:
            np.savez(f, **dudl_var)

        # update checkpoint 
        
        iters.append({
            'params': params,
            'dudl_mean': dudl_mean,
            'dudl_var': dudl_var,
        })

        Checkpoint(os.path.join(args.savepath, args.restart), __file__).update({ 'dudl_ref': dudl_ref, 'iterations': iters}).dump()

    

if __name__ == '__main__':
    main()
