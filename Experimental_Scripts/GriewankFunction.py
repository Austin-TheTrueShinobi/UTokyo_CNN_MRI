import numpy as np
import spotpy

class spotpy_setup(object):
    def __init__(self):
        self.dim=2
        self.parameternames=['x','y']
        self.params=[]
        for parname in self.parameternames:
            spotpy.parameter.Uniform(parname,-10,10,1.5,3.0)

    def parameters(self):           
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        n = len(vector)
        fr = 4000
        s = 0
        p = 1
        for j in range(n): 
            s = s+vector[j]**2
        for j in range(n): 
            p = p*np.cos(vector[j]/np.sqrt(j+1))
        simulation = [s/fr-p+1]
        return simulation

    def evaluation(self):
        observations=[0]
        return observations

    def objectivefunction(self,simulation,evaluation):
        objectivefunction= -spotpy.objectivefunctions.rmse(evaluation,simulation)
        return objectivefunction












    results=[]
spotpy_setup=spotpy_setup()
rep=5000

sampler=spotpy.algorithms.mc(spotpy_setup,    dbname='GriewankMC',    dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.lhs(spotpy_setup,   dbname='GriewankLHS',   dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.mle(spotpy_setup,   dbname='GriewankMLE',   dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.mcmc(spotpy_setup,  dbname='GriewankMCMC',  dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.sceua(spotpy_setup, dbname='GriewankSCEUA', dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.sa(spotpy_setup,    dbname='GriewankSA',    dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.demcz(spotpy_setup, dbname='GriewankDEMCz', dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())

sampler=spotpy.algorithms.rope(spotpy_setup,  dbname='GriewankROPE',  dbformat='csv')
sampler.sample(rep)
results.append(sampler.getdata())




algorithms=['MC','LHS','MLE','MCMC','SCEUA','SA','DEMCz','ROPE']
spotpy.analyser.plot_heatmap_griewank(results,algorithms)