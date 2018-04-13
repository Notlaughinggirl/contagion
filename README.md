# contagion
The package *contact_dynamics.py* defines a class named Contagion with the same named core function.
This method is a Monte-Carlo simulator for disease spreading on static, complex networks with multiple outbreak locations.
It calculates infection arrival times for SIR processes.
We assume a standard contact dynamics, i.e. an infected individual transmits the pathogen with a uniform infection rate to a susceptible
neighbor and recovers with a (uniform) recovery rate. The continuous time problem is mapped to a percolation problem as descibed in 
```
Kenah E., Robins J.M., "Second look at the spread of epidemics on networks", Phys. Rev. E (2007)
```
This approach leads to an efficient implementation because every realization, i.e. every Monte-Carlo step, returns the results for all outbreak locations at once.

For more information and hands-on examples try out the jupyter notebook. 
