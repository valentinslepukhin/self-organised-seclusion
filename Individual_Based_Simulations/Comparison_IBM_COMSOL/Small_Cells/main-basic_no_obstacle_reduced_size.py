
import numpy as np
import time
import datetime
from numba import njit,jit,prange,get_num_threads
import os
import pickle
import shutil
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import cm
import itertools
import multiprocessing as mp

pi = 3.141592653589793

def read_input(inputfile):
    with open(inputfile,'r') as f:
        #Steps
        line = next(f)
        var,name = line.split('#')
        steps = int(var)
        #Writing Interval
        line = next(f)
        var,name = line.split('#')
        wr_int = int(var)
        #Initial number of mother cells
        line = next(f)
        var,name = line.split('#')
        N_in = int(var)
        #Box length: x-direction (perpendicular to cavity length)
        line = next(f)
        var,name = line.split('#')
        Lx = float(var)
        #Box length: y-dir (along cavity length)
        line = next(f)
        var,name = line.split('#')
        Ly = float(var)
        #dx: size of the side of each bin
        line = next(f)
        var,name = line.split('#')
        dx = float(var)
        #Timestep
        line = next(f)
        var,name = line.split('#')
        dt = float(var)
    f.close()
    return steps,wr_int,N_in,Lx,Ly,dx,dt


def initialize(positions,flag,lineage,birth_rate,N_in,Lx,Ly,
               b):
    for n in range(N_in):
        positions[n,0] = np.random.random()*Lx #initial x-pos
        positions[n,1] = np.arcsin(np.random.random())*2*Ly/pi #initial y-pos. Give them the cosine shape.

        flag[n] = 1 #particle exists
        birth_rate[n] = b
        lineage[n] = 0 #set them all as non-fluorescent wild-types

        
    for n in range(N_in,N_max):
        positions[n,0]=0.0
        positions[n,1]=0.0
        flag[n]=0

    f,ax = plt.subplots()
    ax.scatter(positions[:,0],positions[:,1])
    plt.show()
    plt.close()

    return positions,flag,lineage,birth_rate

#   
@njit(cache=False,parallel=False)
def calculate_bins(positions,flag,dx,bins,N_in): #O(N)
    for n in range(N_in): #prange(N_in): #do not do for particles whose flag is 0
        if flag[n]==1:
             bin_x = int(positions[n,0]/dx)
             bin_y = int(positions[n,1]/dx)
             bins[n,0] = bin_x
             bins[n,1] = bin_y
             
##             if bin_x>=20 or bin_y>=40:
##                 print(bin_x,bin_y,n,positions[n])
    return bins


@njit(cache=False,parallel=False)
def calculate_density(positions,flag,dx,X,Y,bins,N_in,density,density_temp,
                      realN_in,R):
    #R should be in lattice positions
    density = np.zeros((X,Y)) 
    density_temp = np.zeros((X,Y))
    for n in range(N_in):
        if flag[n]==1:
            ii,jj = bins[n]
            density[ii,jj] += 1.0 #make it floats since you then average over squares
            realN_in +=1
    density=density/(dx*dx)

                        
    return density,realN_in

##@profile
@njit(cache=False,parallel=False)
def gradient(density,dx,delta_density_x,delta_density_y,X,Y,outside_value): #O(squares*X*Y)
    '''
    Will output two arrays. One for the x-difference and one for the y-difference.
    The gradient of the density is done using center-differences.
    '''
    
    for ii in prange(X):
        for jj in prange(Y):
            #Apply BCs at the 3 walls
            if jj==0: #bottom of the cave
                if ii==0: #left wall
                    delta_density_x[ii,jj] = (density[ii+1,jj]-density[ii,jj])/(dx)
                    delta_density_y[ii,jj] = (density[ii,jj+1]-density[ii,jj])/(dx)
                if ii==X-1: #right wall
                    delta_density_x[ii,jj] = (density[ii,jj]-density[ii-1,jj])/(dx)
                    delta_density_y[ii,jj] = (density[ii,jj+1]-density[ii,jj])/(dx)
                else:
                    delta_density_y[ii,jj] = (density[ii,jj+1]-density[ii,jj])/(dx)
                    delta_density_x[ii,jj] = (density[ii+1,jj]-density[ii,jj])/(dx)
            elif jj==Y-1: #handle outflow of particles at the mouth
                if ii==0: #left wall
                    delta_density_x[ii,jj] = (density[ii+1,jj]-density[ii,jj])/(dx)
                    delta_density_y[ii,jj] = (outside_value-density[ii,jj])/(dx)
                if ii==X-1: #right wall
                    delta_density_x[ii,jj] = (density[ii,jj]-density[ii-1,jj])/(dx)
                    delta_density_y[ii,jj] = (outside_value-density[ii,jj])/(dx)
                else:
                    delta_density_y[ii,jj] = (outside_value-density[ii,jj])/(dx)
                    delta_density_x[ii,jj] = (density[ii+1,jj]-density[ii,jj])/(dx)
            else:
                if ii==0: #left wall, but not bottom
                     delta_density_x[ii,jj] = (density[ii+1,jj]-density[ii,jj])/(dx)
                     delta_density_y[ii,jj] = (density[ii,jj+1]-density[ii,jj])/(dx)
                elif ii==X-1: #right wall, but not bottom
                     delta_density_x[ii,jj] = (density[ii,jj]-density[ii-1,jj])/(dx)
                     delta_density_y[ii,jj] = (density[ii,jj+1]-density[ii,jj])/(dx)
                #In the bulk
                else:
                    delta_density_x[ii,jj] = (density[ii+1,jj]-density[ii-1,jj])/(2*dx)
                    delta_density_y[ii,jj] = (density[ii,jj+1]-density[ii,jj-1])/(2*dx)         

    return delta_density_x,delta_density_y


######## Diffusivities
##@profile
@njit(cache=False)
def collective_diffusivity_value(rho):
    '''
    We take the collective diffusivity function of hard spheres, normalized to 1 at 0-density.
    '''
    if rho>=rho_jam:
        return float(1000.0*(rho-rho_jam)+ 0.6864147263571838) #it's D_col upon jamming.
    else:
        phi=rho*(diam/2)**2*pi #change from number density to packing fraction.
        #take as hard spheres
        mu = 1 * (1-phi)**5.8
        A = (1+phi+phi**2-phi**3)/(1-phi)**3 #equation of state dependent on phi. Carlahan-Starling equation.
        dA = (1+2*phi-3*phi**2)/(1-phi)**3+3/(1-phi)**4*(1+phi+phi**2-phi**3)
        dP = 6/pi*A+6*phi/pi*dA
        out = 1*mu*dP
        return  float(out/1.90985932) #normalize by D_col(phi=0)=1.909859...


##@profile
@njit(cache=False)
def SingleCell_diffusivity_value(rho):
    '''
    Stupid example where D_singcell = 1 if rho<rho_jam, 0.0001 otherwise
    '''    
    if rho>=rho_jam:
        return   0.0015*D_0 #*0.0 makes it deterministic when in jammed mode.
    else:
        out = D_0*(1-rho*0.95/rho_jam) #make it such that it's still not exactly zero at rho_jam
        
        return   out 




##@profile
@njit(cache=False,parallel=False)
def collective_diffusivity_array(D_collective,density,X,Y,N_in): #O(X*Y)
    
    for ii in prange(X):
        for jj in prange(Y):
            rho = density[ii,jj] #density in cells/area
            D_collective[ii,jj] = collective_diffusivity_value( 
                rho)
    return D_collective

# ##@profile
@njit(cache=False,parallel=False)
def SingleCell_diffusivity_list(flag,D_singlecell,bins,density,N_in):
    '''
    For each cell:
        - Recall its bin.
        - Calculate its single-cell diffusivity, evaluated
            at the density of the corresponding bin
    '''
    for n in range(N_in): #prange(N_in): #no need to go beyond existing cells
        if flag[n]==1:
            ii,jj = bins[n]
            rho = density[ii,jj] #density in cells/area
            D_singlecell[n] = SingleCell_diffusivity_value(
                rho)
            
    return D_singlecell


############## UPDATE POSITIONS
##@profile
@njit(cache=False,parallel=False)
def update_positions(positions,flag,bins,D_singlecell,
                     density,delta_density_x,delta_density_y,D_collective,X,Y,N_in,
                     problem,
                     #wall,wall_length,
                     ):

    D_self_array = np.zeros((X,Y),dtype=np.float64)
    for ii in prange(X):
        for jj in prange(Y):
            D_self_array[ii,jj] = SingleCell_diffusivity_value(
                density[ii,jj])
            
    
    delta_D_self_x = np.zeros((X,Y),dtype=np.float64)
    delta_D_self_y = np.zeros((X,Y),dtype=np.float64)
    #Calculate grad(D_self) 
    delta_D_self_x,delta_D_self_y = gradient(
        D_self_array,dx,delta_D_self_x,delta_D_self_y,X,Y,D_0)

    ### WALL POSITION AND LENGTH
    for n in range(N_in): #prange(N_in):
        
        if flag[n]==1:#if particle in domain
            ii,jj = bins[n]

            #### COLLECTIVE FLOW
            ####Diffusivites
            diff_collective = D_collective[ii,jj]
            diff_single = D_singlecell[n]

            

            x_change = 0.0; y_change = 0.0
            #BCs. 0 density flux at the walls
            #BCs also 0 flux at the bottle-butt wall
            if dx<positions[n,0]<Lx-dx:
                x_change = -delta_density_x[ii,jj]/density[ii,jj]*(diff_collective-diff_single)*dt
                #Change due to grad(D_self)
                x_change += delta_D_self_x[ii,jj]*dt
                
            if dx<positions[n,1]:
                y_change = -delta_density_y[ii,jj]/density[ii,jj]*(diff_collective-diff_single)*dt
                #Change due to grad(D_self)
                y_change += delta_D_self_y[ii,jj]*dt

            if x_change>dx or y_change>dx:
                print('FLOW Change too big',n,jj,x_change,y_change)

            positions[n,0] += x_change
            positions[n,1] += y_change

            #### RANDOM WALK
            
            ran_x = np.random.normal()
            ran_y = np.random.normal()
            x_change = 0.0; y_change = 0.0
            
            x_change = np.sqrt(2*diff_single*dt)*ran_x
            y_change = np.sqrt(2*diff_single*dt)*ran_y

            positions[n,0]+=x_change
            positions[n,1]+=y_change

            
            ########### OJO ##############
            ####### Beware children being born across the wall from the mother. To be handled in birth function.
            

            if x_change>dx or y_change>dx:
                print('RW Change too big',n,x_change,y_change)
            
    return positions,problem

############# GIVE BIRTH
##@profile
@njit(cache=False)
def give_birth(positions,flag,birth_rate,lineage,N_in,bins,density,
##               wall,wall_length,
               ):
    N_in_temp = N_in
    for n in range(N_in_temp):
        if flag[n]==1: #only if in the box, can it give a child
            #only if in non_jammed situation, can it give birth:
            ii,jj = bins[n]

            rand = np.random.random()
            if rand<birth_rate[n]*dt:
               

               N_in+=1 #a baby is born :)
               lineage[N_in-1] = lineage[n] #with the mother's lineage
               birth_rate[N_in-1] = birth_rate[n] #and the mother's birth rate


               #daughter's position
               #Do not use Gaussian noise in x and y. Use fixed distance,
               #and random angle. Else there is the chance that a newborn
               #might go accross a barrier.
               r1 = np.random.random()
               
               positions[N_in-1,0] = positions[n,0]+diam*np.cos(r1/(2*pi))
               positions[N_in-1,1] = positions[n,1]+diam*np.sin(r1/(2*pi))

               
               flag[N_in-1] = 1
                      
                
                   
    return positions,flag,birth_rate,lineage,N_in
               
            
            

#### Enforce wall-rebound st all particles inside the box
#### and flag the particles that have exited
##@profile
@njit(cache=False)
def elastic_walls_and_checkIfInside(positions,flag,N_in):
    for n in range(N_in): #prange(N_in):
        if flag[n]==1:
            x,y = positions[n]
            #### Regular Version
            if x<0:
                positions[n,0] = -x
            elif x>Lx:
                positions[n,0] = Lx-(x-Lx)
            if y<0:
                positions[n,1] = -y
                
            if y>Ly:
                flag[n] = 0
                ##                print(n)

    return positions,flag

#### Integration
##@profile
############ VERY IMPORTANT
############ DO NOT NUMBIFY THE INTEGRATE FUNCTION. IT CAUSES MEMORY LEAKS
def integrate(positions,bins,density,density_temp,delta_density_x,delta_density_y,
              D_collective,D_singlecell,
              flag,N_in,step,realN_in,R,
              birth_rate,lineage,X,Y,
              problem,
              ):

    bins = calculate_bins(positions,flag,dx,bins,N_in)

    density,realN_in = calculate_density(positions,flag,dx,X,Y,bins,N_in,density,density_temp,realN_in,R=0)


    
    #Filter in 3 sweeps around the wall (left all, right all, left&right beyond wall).
    ### This is done so the profile is not smoothed across the wall.
    density = gaussian_filter(density,5)

    delta_density_x,delta_density_y = gradient(density,dx,delta_density_x,delta_density_y,X,Y,
                                               density_outside)
    
    #Find position of jamming wavefront.
    jam_wavefront = 0.0
    epsilon = 0.005
    density_1D = np.mean(density,axis=0)
    zeros = np.where(abs(density_1D-rho_jam)<epsilon)[0]
    if len(zeros)>0:
        jam_wavefront = zeros[-1]
    

    delta_D_self_x = np.zeros((X,Y),dtype=np.float64)
    delta_D_self_y = np.zeros((X,Y),dtype=np.float64)
    
    D_collective =collective_diffusivity_array(D_collective,density,X,Y,N_in)
    D_singlecell =SingleCell_diffusivity_list(flag,D_singlecell,bins,density,N_in)


    ##### UPDATE POSITIONS
    positions,problem = update_positions(positions,flag,bins,D_singlecell,
                     density,delta_density_x,delta_density_y,D_collective,X,Y,N_in,
                     problem,
##                     wall,wall_length,
                    )
    density_at_00 = np.mean(density[:,0])


    ##### GIVE BIRTH

    positions,flag,birth_rate,lineage,N_in = give_birth(positions,flag,
                                                        birth_rate,lineage,N_in,
                                                        bins,density,
##                                                        wall,wall_length,
                                                        )


    ###Apply BCs at the single-cell level: wall rebounds
    positions,flag = elastic_walls_and_checkIfInside(positions,flag,N_in)


    return positions,flag,lineage,N_in,density_at_00,problem,realN_in,jam_wavefront,density_1D


################################################################################
################################################################################
################################################################################
################################################################################
########################                 MAIN           ########################
################################################################################
################################################################################
################################################################################
################################################################################


################################################################################
########################    CONSTANTS ##########################################
################################################################################
pi= 3.141592653589793
N_max = 300000 #60000 for diam=1.0 #maximum number of allowed particles
    
diam= 1.0/np.sqrt(10) #st cell area (i.e. cell size) is changed by a factor 10
cell_area = (diam/2)**2*pi #the area fraction
                           #occupied by a single cell in a bin
density_outside = 0.0 #density outside the mouth of the cave
sigma = 2.0*diam #std_dev of where mother-daughter distance at birth

out = 0.0
D_0 = 1.0 #single-cell
D =1.0 #collective
squares = 9 #no. of squares over which it'll be averaged.


############### CHOICE OF BIRTH RATE:
'''
FOr a cavity of length Ly=100 and D_0 = D(phi=0) = 1, establishment occurs for
b_est = 0.0002467400073616

and jamming occurs around L/L_est = 1.1 which corresponds to
b_jam = 0.0002985554089075361

we thus set b above b_jam, and then by lowering the b of the Antibiotic-susceptible strain,
we will go towards unjamming.
'''
##b = 0.00031 #is jammed before AB, but will unjam after AB even for small AB effects.


phi_jam = 0.64
rho_jam = 0.64/((diam/2)**2*pi) #corresponds to
print('rho_jam',rho_jam)


################################################################################
########################    SYSTEM VARIABLES   #################################
################################################################################

positions = np.zeros((N_max,2)) #input obviously wrong initial positions
bins = np.zeros((N_max,2),dtype=np.int32) #the bin that each particle is in
flag = np.zeros(N_max,dtype=np.int32)
lineage = np.zeros(N_max,dtype=np.int32)
birth_rate = np.zeros(N_max)

inputfile = 'input.txt'
steps,wr_int,N_in,Lx,Ly,dx,dt = read_input(inputfile)
density = np.zeros((round(Lx/dx),round(Ly/dx)),dtype=np.float64)
density_temp = np.zeros((round(Lx/dx),round(Ly/dx)),dtype=np.float64)
delta_density_x = np.zeros(np.shape(density))
delta_density_y = np.zeros(np.shape(density))
D_self_array = np.zeros(np.shape(density))
delta_D_self_x = np.zeros(np.shape(density))
delta_D_self_y = np.zeros(np.shape(density))
D_collective = np.zeros(np.shape(density))
D_singlecell = np.zeros(N_max) #each cell has its own diffusivity
(X,Y) = np.shape(density)
print(np.shape(density))
step0=0
total_pop = [sum(flag)]
density_at_00_vect = []
density_array = []
R = 0.0 #dummy argument from old code, useless now.


def simulation(arguments):

    arguments,ID = arguments
    old_arguments,b = arguments
    positions,flag,lineage,birth_rate,N_in,step0,density,total_pop,wr_int =old_arguments
##    q1,q2 = qs

    
    problem = False
    realN_in = 0 #actual number of particles inside box. Consider that N_in also counts the ones
            #that exited
    
    out_dir = 'out.'+str(ID)
    
    continuation=True
    if os.path.exists(out_dir) and continuation==False: #if cont.=True don't remove existing files!
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    elif continuation ==False:
        os.mkdir(out_dir)

##    if os.path.exists(out_dir):
##        shutil.rmtree(out_dir)
##    os.mkdir(out_dir)


    ####Output arguments to out directory
    args = (b)
    args_file = os.path.join(out_dir,'arguments.dat')
    with open(args_file,'wb') as f:
        pickle.dump(args,f)
        f.close()

    positions,flag,lineage,birth_rate = initialize(positions,flag,lineage,birth_rate,N_in,Lx,Ly,
                                                   b)
    
    total_pop=[]
    resistant_fraction = []
    jam_wavefront_vect = []
    
    if continuation==True:
        ####### Also. Binarize the lineage and the birth rate
        last_dir = out_dir
        pos_out = os.path.join(last_dir,'positions.dat')
        with open(pos_out,'rb') as f:
            positions_temp = pickle.load(f)
            f.close()
        lin_out = os.path.join(last_dir,'lineage.dat')
        with open(lin_out,'rb') as f:
            lineage_temp = pickle.load(f)
            f.close()
        #Load total population vector
        file_total_population = os.path.join(out_dir,'total_population.dat')
        with open(file_total_population,'rb') as f:
            total_pop = pickle.load(f)
            f.close()
##                #Update resistant fraction
##                file_res_frac = os.path.join(out_dir,'resistant_fraction.dat')
##                with open(file_res_frac,'wb') as f:
##                    pickle.dump(resistant_fraction,f)
##                    f.close()
        #Load jam_wavefront vector
        file_jam_wavefront = os.path.join(out_dir,'jam_wavefront.dat')
        with open(file_jam_wavefront,'rb') as f:
            jam_wavefront_vect = pickle.load(f)
            f.close()
   
        positions = np.zeros((N_max,2)) #input obviously wrong initial positions
        flag = np.zeros(N_max,dtype=np.int32)
        lineage = np.zeros(N_max,dtype=np.int32)
        birth_rate = np.zeros(N_max)
        ii=0
        for n in range(N_max): 
            if 0<positions_temp[n,1]<Ly and 0<positions_temp[n,0]<Lx: #means the particle is initiated and not out
                flag[ii] = 1
                positions[ii,0] = positions_temp[n,0]
                positions[ii,1] = positions_temp[n,1]
                birth_rate[ii] = b
                lineage[ii] = lineage_temp[n]

                ii+=1
                
##        plt.figure()
##        plt.scatter(positions[:,1],positions[:,0])
##        plt.show()
##        plt.close()

        
    N_in = sum(flag)
    frac_resistant = sum(lineage)
    step0=0 #set to the corresponding value
    step0 = 0
    
        

    threads = get_num_threads()
    print('Number of threads available',threads)
    print('N_in', N_in,'dx',dx,'cell diameter',diam)
    print('Number of cells when completely jammed',Lx*Ly*rho_jam)
    print('Number of cells when pack. frac.==1',Lx*Ly)
    print('Prob(>1 births in dt at b and N_max',1-(1-dt*b)**(Lx*Ly)-(1-dt*b)**((Lx*Ly)-1)*b*dt*(Lx*Ly))
##    print('Particles in bin when jammed',dx**2/(cell_area)*0.64)
##    print('Particles in bin when phi>1',dx**2/(cell_area))
    print('Max. allowed Diffusion',dx/dt)
    print('Maximum expected number of births per unit time',Lx*Ly*b*dt)
    print('Maximum expected number of outwashed cells p.u.t.',100*dt/Ly/dx,'*D')
    t1 = time.time()
    
    binarized_lineages = False
    antibiotic_added = False
    t_max = 250000
    counter = 0
    for step in range(step0,step0+steps+1):
        positions,flag,lineage,N_in,density_at_00,problem,realN_in,jam_wavefront,density_1D = integrate(
                      positions,bins,density,density_temp,delta_density_x,delta_density_y,
                      D_collective,D_singlecell,
                      flag,N_in,step,realN_in,R,
                      birth_rate,lineage,X,Y,
                      problem,
                      )
        
        
        density_array.append(density)
        if problem==True:
            with open('out/total_population','wb') as f:
                pickle.dump(total_pop,f)
                f.close()
            os.exit()
        if N_in>N_max or step*dt>t_max:
            break

##        #### ADD ANTIBIOTIC #####
##        t_AB= 20000 #intervals at which AB is added. Make it big enough that it reaches steady state.
##        wall=Lx/2
##        AB_counter = 0
##        if counter== 4000000:
####        if counter==0:
##            #The (step*dt%t_AB)<dt condition should be met only once, for step*dt in the interval (t_AB-dt,t_AB+dt)
##            AB_counter += 1
##            for n in range(N_in):
##                if lineage[n]==0:
##                    if AB_counter==0: #first AB intro
##                        birth_rate[n] = b*(1-q1) #decrease birth rate of Blue cells by 20% each time.
##                    elif AB_counter==2: #second AB intro.
##                        birth_rate[n] = b*(1-q2)
##                elif lineage[n]==1:
##                    birth_rate[n] *= 1.0*1 #effect on Green cells is none by definition
##            counter = 0
##        else:
##            counter +=1
                
            
        if step%wr_int==0:
            ### Every wr_int, append to output arrays (but do not write to output)
            
            N_inside = sum(flag)
##            fraction_resistant = len(np.where((lineage==1) & (flag==1))[0])
            total_pop.append(N_inside)
##            resistant_fraction.append(fraction_resistant/N_inside)
            jam_wavefront_vect.append(jam_wavefront)
            density_at_00_vect.append(density_at_00)
            print(round(step*dt,4),N_in,N_inside,N_in-N_inside)

            ### Every 100*wr_int, write to output.
            if step%(20*wr_int)==0:
                
                #WRITE TO OUTPUT:
                #Update total population
                file_total_population = os.path.join(out_dir,'total_population.dat')
                with open(file_total_population,'wb') as f:
                    pickle.dump(total_pop,f)
                    f.close()
##                #Update resistant fraction
##                file_res_frac = os.path.join(out_dir,'resistant_fraction.dat')
##                with open(file_res_frac,'wb') as f:
##                    pickle.dump(resistant_fraction,f)
##                    f.close()
                #Update jam_wavefront
                file_jam_wavefront = os.path.join(out_dir,'jam_wavefront.dat')
                with open(file_jam_wavefront,'wb') as f:
                    pickle.dump(jam_wavefront_vect,f)
                    f.close()
                #Update density_at_00
                file_density_floor = os.path.join(out_dir,'density_floor.dat')
                with open(file_density_floor,'wb') as f:
                    pickle.dump(density_at_00_vect,f)
                    f.close()

                '''
                f,ax = plt.subplots()
                t_vect = np.arange(0,len(total_pop))*dt*wr_int
                ax.plot(t_vect,total_pop,'k-')
                ax.plot(t_vect,np.array(total_pop)*np.array(resistant_fraction),'g-')
                ax.plot(t_vect,np.array(total_pop)*(1-np.array(resistant_fraction)),'b-')
                ax2 = ax.twinx()
                ax2.plot(t_vect,jam_wavefront_vect,'r-')
                plt.show()
                plt.pause(0.2)
                plt.close()
                '''

            
                ####Update state of the cavity
                #Output positions
                file_positions = os.path.join(out_dir,'positions.dat')
    ##            np.savetxt(file_positions,positions)
                with open(file_positions,'wb') as f:
                    pickle.dump(positions,f)
                    f.close()
                #Output flag
                file_flag = os.path.join(out_dir,'flag.dat')
    ##            np.savetxt(file_flag,flag)
                with open(file_flag,'wb') as f:
                    pickle.dump(flag,f)
                    f.close()
                #Output lineage
                file_lineage = os.path.join(out_dir,'lineage.dat')
    ##            np.savetxt(file_lineage,lineage)
                with open(file_lineage,'wb') as f:
                    pickle.dump(lineage,f)
                    f.close()
                #1D-flattened density profile
                file_density = os.path.join(out_dir,'density.dat')
    ##            np.savetxt(file_lineage,lineage)
                with open(file_density,'wb') as f:
                    pickle.dump(density_1D,f)
                    f.close()  

            
            
                

    t2 = time.time()
    print('N_in', N_in)
    print('Time Taken: ',t2-t1,' seconds')
    
    return

if __name__=='__main__':
    print(datetime.datetime.now())

    #### Parallel-ready code ####
    b = 0.00031 #is jammed before AB, but will unjam after AB even for small AB effects.

    #b_vect as specified to valentin
##    b_vect = [0.00022268, 0.00024674, 0.00025671, 0.00026687, 0.00027724,0.00029856, 0.00035531, 0.00041699, 0.00048361]
    #For l = [1.01,1.02,1.03,1.04,1.05,1.06,1.07,1.08,1.09]
    b_vect = [0.000251699585663564, #1.01
              0.00025670840988566997, #1.02
              0.00026176658212966866,#1.03
              0.0002668741023955601,#1.03
              0.0002720309706833441,#1.04
              0.0002772371869930208,#1.06
              0.0002824927513245902,#1.07
              0.00028779766367805214,#1.08
              0.0002931519240534069]#1.09



    old_arguments = [(positions,flag,lineage,birth_rate,N_in,step0,density,total_pop,wr_int)]
    
    arguments = itertools.product(old_arguments,b_vect)
    arguments = list(arguments)
    ID_vect = np.arange(len(arguments))
    arguments_new = zip(arguments,ID_vect)
    arguments = list(arguments_new)
    print(len(arguments))
    ##### TEST: USE ONLY 4 FIRST ARGUMENTS OF LIST TO MAKE SURE THE CODE WORKS AND TEST HOW
    ##### LONG IT TAKES TO RUN.        
    
    
    ctx = mp.get_context('spawn')
    with ctx.Pool() as p:
        results = p.map(simulation,arguments)
    p.close()
    
    simulation(arguments)

    
    














