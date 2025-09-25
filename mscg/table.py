#

import numpy as np

class Table:
    
    def __init__(self, model, force=True, prefix=''):
        self.model = model
        self.force = force
        self.prefix = prefix
    
    def compute(self, xmin, xmax, xinc):
        self.x = np.arange(xmin, xmax+xinc, xinc)
        self.v = self.model.compute_table(self.x, self.force)
        self.n = self.v.shape[0]
        
        if(self.force):
            self.f = self.v
            u = np.zeros(self.n)
            u[:-1] = 0.5 * (self.f[:-1] + self.f[1:])
            u = np.cumsum(u[::-1])[::-1]
            self.u = u * xinc
            
        else:
            self.u = self.v
            f = np.zeros(self.n+1)
            f[1:-1] = -np.diff(self.u, 1) / xinc
            f[0] = 2.0 * f[1] - f[2]
            f[-1] = 2.0 * f[-2] - 2.0 * f[-3]
            self.f = 0.5 * (f[:-1] + f[1:])
    
    
    def padding_low(self, model_min):
        i = 0
        
        while self.x[i] < model_min - 1.0E-6:
            i += 1
        
        while i<self.n-1 and (self.f[i]<0 or self.f[i] - self.f[i+1] <= 0):
            i += 1

        if i<self.n-1:
            df = self.f[i] - self.f[i+1]
            i -= 1
            
            while i>=0:
                self.f[i] = self.f[i+1] + df
                self.u[i] = (self.f[i+1] + 0.5 * df) * (self.x[i+1] - self.x[i]) + self.u[i+1]
                i-=1

        self.u -= self.u[-1]

    def padding_low2(self, model_min):
        i = 0
        
        while self.x[i] < model_min - 1.0E-6:
            i += 1
        
        i += 1
        df = abs(self.f[i+1] - self.f[i+2])
                  
        while i>=0:
            self.f[i] = self.f[i+1] + df
            self.u[i] = (self.f[i+1] + 0.5 * df) * (self.x[i+1] - self.x[i]) + self.u[i+1]
            i-=1
            
        self.u -= self.u[-1]

    def padding_low_power(self, model_min):
        i = 0

        while self.x[i] < model_min - 1.0E-6:
            i += 1
        while i<self.n-1 and (self.f[i]<0 or self.f[i] - self.f[i+1] <= 0):
            i += 1
        i += 1
        print('Padding low power from', self.x[i])

        k = 20
        # while self.f[k] > 0:
        #     k += 1
        # print('k:', k, 'x[k]:', self.x[k])

        power = - self.x[i]*self.f[i] / (self.u[i] - self.u[k])
        A = (self.u[i] - self.u[k]) / self.x[i]**power
        for j in range(i):
            self.u[j] = A * self.x[j]**(power) + self.u[k]
            self.f[j] = -power * A * self.x[j]**(power-1)

    def padding_low_log(self, model_min):
        i = 0

        while self.x[i] < model_min - 1.0E-6:
            i += 1
        while i<self.n-1 and (self.f[i]<0 or self.f[i] - self.f[i+1] <= 0):
            i += 1
        i += 3
        print('Padding low power from', self.x[i])

        B = self.u[i] + self.f[i] * self.x[i] * np.log(self.x[i])
        A = self.f[i] * self.x[i]
        for j in range(i):
            self.u[j] = -A * np.log(self.x[j]) + B
            self.f[j] = A / self.x[j]
        

    def padding_low_exp(self, model_min):
        i = 0
        while self.x[i] < model_min - 1.0E-6:
            i += 1
        while i<self.n-1 and (self.f[i]<0 or self.f[i] - self.f[i+1] <= 0):
            i += 1
        i += 1
        print('Padding low exp from', self.x[i])
        
        k = 0
        while self.f[k] > 0:
            k += 1

        B = self.f[i] / (self.u[i] - self.u[k])
        A = (self.u[i] - self.u[k]) * np.exp(B * self.x[i])
        for j in range(i):
            self.u[j] = A * np.exp(- B * self.x[j]) +self.u[k]
            self.f[j] = A * B * np.exp(- B * self.x[j])

    def padding_gauss(self, model_min, u0 = 100):
        i = 0

        while self.x[i] < model_min - 1.0E-6:
            i += 1
        while i<self.n-1 and (self.f[i]<0 or self.f[i] - self.f[i+1] <= 0):
            i += 1
        i += 1
        print('Padding low power from', self.x[i])

        # B = np.arccos(2 * self.u[i] /u0 -1 ) / self.x[i]
        # for j in range(i):
        #     self.u[j] = u0 / 2 * (1+np.cos(B * self.x[j]))
        #     self.f[j] = B * u0 /2 * np.sin(B * self.x[j])

        # B = np.log(u0 / self.u[i]) / (self.x[i]**4)
        # for j in range(i):
        #     self.u[j] = u0 * np.exp(-B * self.x[j]**4)
        #     self.f[j] = -4 * B * self.x[j]**3 * u0 * np.exp(-B * self.x[j]**4)

        A = self.x[0] * (u0 - self.u[i]) 
        B = self.u[i] - A / self.x[i]
        for j in range(i):
            self.u[j] = A / self.x[j] + B
            self.f[j] =  A / self.x[j]**2

        # oldP, P = 0, 1
        # while abs(P - oldP) > 1.0E-2:
        #     oldP = P
        #     P = self.f[i] *self.x[i]**(P+1) / self.x[0]**(P) / (u0 - self.u[i])
        # A = self.x[0]**P * (u0 - self.u[i])
        # B = self.u[i] - A / self.x[i]**P
        # print('P:', P, 'A:', A, 'B:', B)
        # for j in range(i):
        #     self.u[j] = A / self.x[j]**P + B
        #     self.f[j] = P * A / self.x[j]**(P+1)

    def padding_high(self, model_max):
        i = self.n-1
        
        while self.x[i] > model_max + 1.0E-6:
            i -= 1
        
        while i>0 and (self.f[i]>0 or self.f[i] - self.f[i-1] >= 0):
            i -= 1
        
        if i>0:
            df = self.f[i] - self.f[i-1]
            i += 1
            
            while i<self.n:
                self.f[i] = self.f[i-1] + df
                self.u[i] = - (self.f[i-1] + 0.5 * df) * (self.x[i] - self.x[i-1]) + self.u[i-1]
                i+=1
        
    
    def dump_lammps(self, xmin=None, xmax=None, xinc=None):
        if xmin is not None:
            self.compute(xmin, xmax, xinc)
        
        txt = "# Table %s: id, r, potential, force\n\n" % (self.model.name)
        txt += self.model.name.split("_")[1] + "\n"

        if 'Pair' in self.model.name:
            txt += "N %d R %f %f\n\n" % (self.n, self.x[0], self.x[-1])
        else:
            txt += "N %d\n\n" % (self.n)
        
        for i in range(self.n):
            txt += "%d %f %f %f\n" % (i+1, self.x[i], self.u[i], self.f[i])
        
        with open(self.prefix + self.model.name + ".table", "w") as f:
            f.write(txt)
    
    @staticmethod
    def load_lammps(filename, xinc=0.001):
        with open(filename, 'r') as file:
            rows = file.read().strip().split("\n")
        
        n = int(rows[3].split()[1])
        w = rows[5].split()
        x1, e1, f1 = float(w[1]), float(w[2]), float(w[3])
        xmin = x = x1
        e, f = [], []
        
        for row in rows[6:6+n]:
            w = row.strip().split()
            
            x0, e0, f0 = x1, e1, f1
            x1, e1, f1 = float(w[1]), float(w[2]), float(w[3])
            dx, de, df = x1 - x0, e1 - e0, f1 - f0
            
            while x<=x1:
                e.append(e0 + (x-x0) / dx * de)
                f.append(f0 + (x-x0) / dx * df)
                x += xinc
        
        return {'min':xmin, 'inc':xinc, 'efac':np.array(e, dtype=np.float32), 'ffac':np.array(f, dtype=np.float32)}
        
        
        
        
        
        
    
    
