import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import shutil
from scipy.optimize import curve_fit
from scipy.signal import convolve
from scipy.stats import linregress

import matplotlib as mpl
font = {'weight':'normal',
        'size':12}
mpl.rc('font',**font)

diam = 1.0/np.sqrt(10) 

IDs = np.arange(9)
##IDs =np.arange(3)
dt = 0.005; wr_int = 1000
argument_vector = []
chosen_IDs = np.random.choice(IDs,size=1)#for plotting
pi = 3.1415926535
phi_jam = 0.64

d = {} #Dictionary with the info pertaining to each ID (each reduced length)
'''
For Valentin's l values
'''
l = [0.95,1.0,1.02,1.04,1.06,1.1,1.2,1.3,1.4] #reduced lengths fro valentin
b_vect = [0.00022268, 0.00024674, 0.00025671, 0.00026687, 0.00027724,0.00029856, 0.00035531, 0.00041699, 0.00048361] #corresponding to each reduced length
phi_0_vect = [0.0,0.0,0.05,0.095,0.15,phi_jam,phi_jam,phi_jam,phi_jam]
'''
For my test on the gaseous simulations
'''
l = [1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09]
b_vect = [0.000251699585663564, #1.01
              0.00025670840988566997, #1.02
              0.00026176658212966866,#1.03
              0.0002668741023955601,#1.03
              0.0002720309706833441,#1.04
              0.0002772371869930208,#1.06
              0.0002824927513245902,#1.07
              0.00028779766367805214,#1.08
              0.0002931519240534069]#1.09
phi_0_vect = [0.023,     #1.01
         0.05,#0.045,     #1.02
         0.07,      #1.03
         0.094,     #1.04
         0.12,      #1.05
         0.15,      #1.06
         0.18,      #1.07
         0.2245,    #1.08
         0.27]   #1.09
phi_0_measured = []
phi_0_measured_std = []

for ID in IDs:
    d[ID] = [l[ID],phi_0_vect[ID]]

def cos(x,amplitude):
    L = x[-1]
    out = amplitude*np.cos(pi/2*x/L)
    return out    
    


first_AB_time = 20000
second_AB_time = 40000
try:
    shutil.rmtree('out')
    shutil.rmtree('out/figures')
    
except: pass
os.mkdir('out')
os.mkdir('out/figures')


for ID in IDs:
    print(ID)
    out_dir = 'out.'+str(ID)
    #Density at the floor
    density_floor_file = os.path.join(out_dir,'density_floor.dat')
    try:
        with open(density_floor_file,'rb') as f:
            density_floor = pickle.load(f)
            f.close()

        print('density at the floor ok')
    except:
        continue
    
    #Jam Wavefront
    jam_front_file = os.path.join(out_dir,'jam_wavefront.dat')
    try:
        with open(jam_front_file,'rb') as f:
            jam_wavefront = pickle.load(f)
            jam_wavefront = np.array(jam_wavefront)/2
            #the /2 correction is there because the main.py file does not account
            #for the dx=0.5 and take the index as the position
            ###Piece of code to undo the /2 in parts where it was applied
            #already in the main.py file
            f.close()

        print('jam front ok')
    except:
        continue
   



    #### Check if TOTAL unjamming occured
    unjamming = 0 #did not occur
    if len(np.where(jam_wavefront<diam)[0])>0: #set threshold at size 1.0=cell diam.
        unjamming = 1
    
    #Total Population
    total_pop_file = os.path.join(out_dir,'total_population.dat')
    try:
        with open(total_pop_file,'rb') as f:
            total_pop = pickle.load(f)
            total_pop = np.array(total_pop)
            f.close()
        print('total pop ok')
    except: continue

        
    #Arguments
    args_file = os.path.join(out_dir,'arguments.dat')
    try:
        with open(args_file,'rb') as f:
            arguments = pickle.load(f)
            f.close()
            b = arguments
            argument_vector.append((ID,arguments,unjamming))
        print('arguments ok')
    except: continue
    density_file = os.path.join(out_dir,'density.dat')
    try:
        with open(density_file,'rb') as f:
            density = pickle.load(f)
            f.close()
            print('density ok')
    except: continue  




#############################################################



#### Build figures for selected simulations
##if ID in chosen_IDs:
    if ID in IDs:
        

        
        #Convert density to packing fraction
        dy = 0.5;
        Ly=100
        y_space = np.arange(len(density))*dy
        pack_frac = density*(diam/2)**2*pi


        L_red, phi_0 = d[ID]
        xdata = y_space
        popt,pcov = curve_fit(cos,xdata,pack_frac,p0=[phi_0])
        
        std = pcov[0,0]**0.5
        
        print(popt,std)

                              
        f,ax =plt.subplots()
        ax.plot(y_space,pack_frac,label='Individual-Based Model')
        if phi_0<phi_jam:
            ax.scatter(0,phi_0,s=5,c='k',label='Theory')
            ax.errorbar(0,popt,yerr=std,capsize=2,markersize=2,
                        marker='o',elinewidth=1,
                        linewidth=0,c='r',label='Fit')
            ax.plot(y_space,cos(y_space,popt),'r:')
        ax.set_xlabel('Position')
        ax.set_ylabel('Packing fraction')
        ax.set_title('Density Profile')
        ax.set_ylim(-0.05,1.1*phi_jam)
        ax.legend(loc='best',frameon=False)
        plt.tight_layout()
##        plt.show()
        plt.savefig('out/figures/'+str(ID)+'.density.png')
        plt.close()
        
        f,ax = plt.subplots()
        t_vect = np.arange(0,len(total_pop))*dt*wr_int
        ax.plot(t_vect,total_pop,'k-',
                label='Total')
##        ax.plot(t_vect,np.array(total_pop)*np.array(resistant_fraction),'g-',
##                label = 'Resistant')
##        ax.plot(t_vect,np.array(total_pop)*(1-np.array(resistant_fraction)),
##                'b--',label='Susceptible')
        ax.set_ylabel('No. of cells')
        ax.set_ylim(0,1.3*max(total_pop))
        ax.legend(loc='upper left',frameon=False)
        ax2 = ax.twinx()
        ax2.plot(t_vect,jam_wavefront,'r-',
                 label = 'Jam front')
        ax2.set_ylabel('Position of jammed front')
        ax2.legend(loc='upper right',frameon=False)
        ax2.set_ylim(0,Ly)
        plt.tight_layout()
##        plt.show()
##        plt.pause(0.2)
        plt.savefig('out/figures/'+str(ID)+'.png')
        plt.close()

        pack_frac_floor = np.array(density_floor)*(diam**2/4*3.141592)
        ### Total Average
##        domain = min(10000,len(pack_frac_floor))
        domain = len(pack_frac_floor)
##        domain = 900
        
        avg = np.mean(pack_frac_floor[-domain:])
        std = np.std(pack_frac_floor[-domain:])

        #### Append measured phi_0
##        phi_0_measured.append(popt[0]) #inferred phi_0 at the floor from cosine fit
        phi_0_measured.append(avg) #average of measured phi_0 over last time instances

        
        phi_0_measured_std.append(std)
        ### ·Rolling average
##        window_size = 100
##        domain = 10000
##        avg = convolve(pack_frac_floor[-domain:],np.ones(window_size)/window_size,
##                       mode='same',method='fft')
##        std = np.zeros(len(avg))
        
        f,ax = plt.subplots()
        ax.plot(t_vect[-len(pack_frac_floor):],pack_frac_floor,
                label='Simulation',zorder=0)
        ### if using Total Average
        ax.plot(t_vect[-domain:],[avg]*domain,'r:',
                label='Simulation Average',zorder=1)
        ax.fill_between(t_vect[-domain:],
                        [avg+std]*domain,
                        [avg-std]*domain,
                        alpha=0.2,facecolor='r',edgecolor=None,zorder=1,
                        label='Simulation avg. '+r'$\pm \sigma$')
        ### if using Rolling Average
##        ax.plot(t_vect[-domain:],avg,'r:',
##                label='Simulation Average',zorder=1)
##        ax.fill_between(t_vect[-domain:],
##                        avg+std,
##                        avg-std,
##                        alpha=0.2,facecolor='r',edgecolor=None,zorder=1,
##                        label='Simulation avg. '+r'$\pm \sigma$')

        ax.plot(t_vect[-len(density_floor):],[phi_0]*len(pack_frac_floor),'k-',
                label = 'Theory')
##        ax.plot(t_vect[-len(density_floor):],[phi_0*np.sqrt(2)]*len(pack_frac_floor),
##                'g-',
##                label = 'Alternative Theory')
        ax.set_ylim(bottom=-0.01)
        ax.set_ylabel(r'$\Phi_0$')
        ax.set_title('Packing fraction at the floor of the cavity, '+r'$\Phi_0$')
        ax.set_xlabel('Time')
        plt.legend(loc='best',frameon=False)
        plt.tight_layout()
        plt.savefig('out/figures/'+str(ID)+'packing_fraction_floor.png')
        plt.close()


###############################################################################
###If using the non-valentin parameters, remove l=1.01 and l=1.09
phi_0_vect = phi_0_vect#[:-2]
phi_0_measured = phi_0_measured#[:-2]
phi_0_measured_std = phi_0_measured_std#[:-2]
###############################################################################

###COMSOL
comsol_l = [1.02,1.04,1.06]
comsol_phi_0_theory = [0.05, 0.095, 0.15]
comsol_phi_0_measured = np.array([0.0595,0.1196,0.192])
comsol_phi_0_measured *= (0.5)**2*pi
comsol_std = [0,0,0]

f,ax = plt.subplots()
Delta_phi_0_min = 0.04 #see notebook on ipad "Debugging phi_0 mismatch...". It's the minimum change in Phi_0 from an addition/substraction of a single cell.
'''
Import and plot the results from Study5 which shows the effects of finite N
'''

with open('phi_0_measured.big_cells.dat','rb') as f:
    phi_0_measured_big_cells = pickle.load(f)
    f.close()
with open('phi_0_measured_std.big_cells.dat','rb') as f:
    phi_0_measured_big_cells_std = pickle.load(f)
    f.close()
ax.errorbar(phi_0_vect,np.array(phi_0_measured_big_cells),yerr = phi_0_measured_big_cells_std,
            capsize=5,linewidth=0,
            elinewidth=3,markersize=0,
            c='tab:red',
            zorder=10)
ax.scatter(phi_0_vect,np.array(phi_0_measured_big_cells),
           marker='^',
           s=60,c='tab:red',label=r'IBM - large cells',)


#### Plot the correct phi_0
ax.errorbar(phi_0_vect,np.array(phi_0_measured),yerr = phi_0_measured_std,
            capsize=5,linewidth=0,
            elinewidth=3,markersize=0,
            c='tab:blue',
            zorder=10)
ax.scatter(phi_0_vect,np.array(phi_0_measured),
           marker='^',
           s=60,c='tab:blue',label='IBM - small cells',)
ax.scatter(comsol_phi_0_theory,comsol_phi_0_measured,
            marker='o',s=60,
            c='tab:orange',label='COMSOL',zorder=2)
ideal = np.arange(0.0,0.4,(phi_0_vect[-1]-phi_0_vect[0])/100)
ax.plot(ideal,ideal,'k--',label='1D-RD',linewidth=3,
        zorder=0)
ax.set_xlabel(r'Theoretical $\Phi_0$')
ax.set_ylabel(r'Measured $\Phi_0$ in Simulation')
plt.legend(loc='best',frameon=False)
plt.tight_layout()
plt.savefig('out/figures/predicted_vs_theoretical_phi0.png')
plt.show()
plt.close()

   
    








