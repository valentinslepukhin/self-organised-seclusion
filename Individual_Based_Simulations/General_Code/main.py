
import numpy as np
import time
from numba import njit,jit,prange,get_num_threads
import os
import pickle
import shutil
##import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import cm
import itertools
import multiprocessing as mp
import matplotlib.pyplot as plt

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
    '''
    Initialize the cavity with some initial population.
    '''
    for n in range(N_in):
        positions[n,0] = np.random.random()*Lx #initial x-pos
        rany = np.random.random()
        positions[n,1] = np.arcsin(rany)*2*Ly/pi #y-pos sampled from a cosine-shaped profile.

        flag[n] = 1 #particle exists
        birth_rate[n] = b
        lineage[n] = 0 #sets them all as non-fluorescent wild-types
        
        
    for n in range(N_in,N_max):
        positions[n,0]=0.0
        positions[n,1]=0.0
        flag[n]=0

    return positions,flag,lineage,birth_rate

 
@njit(cache=False,parallel=False)
def calculate_bins(positions,flag,dx,bins,N_in): #O(N)
    for n in range(N_in): #prange(N_in): #do not do for particles whose flag is 0
        if flag[n]==1:
             bin_x = int(positions[n,0]/dx)
             bin_y = int(positions[n,1]/dx)
             bins[n,0] = bin_x
             bins[n,1] = bin_y
             

    return bins

def calculate_density_nonaveraged(positions,flag,dx,X,Y,bins,N_in,density,density_temp,
                      realN_in,R): #O(N)
    #integer division by mesh size returns the corresponding bin
    '''
    OJO: maybe this function is O(2*N). O(N) from the division by dx
    and O(N) from the assignment to density.
    Maybe if looping in positions and assigining directly to density
    it will turn O(N), saving half the computation time.
    '''
    realN_in = 0 #actual number of particles inside box. Consider that N_in also counts the ones
                #that exited
    density = np.zeros((X,Y))
    for n in range(N_in): #prange(N_in):
        if flag[n]==1:
            ii,jj = bins[n]
            density[ii,jj] += 1

            realN_in +=1

    density = density/(dx**2)
           
   
    return density,realN_in

@njit(cache=False)
def calculate_density_square(positions,flag,dx,X,Y,bins,N_in,density,density_temp,
                      realN_in): #O(N)
    #integer division by mesh size returns the corresponding bin
    '''
    OJO: maybe this function is O(2*N). O(N) from the division by dx
    and O(N) from the assignment to density.
    Maybe if looping in positions and assigining directly to density
    it will turn O(N), saving half the computation time.
    '''
    

    density = np.zeros((X,Y)) 
    
    for n in range(N_in):
        if flag[n]==1:
            ii,jj = bins[n]
            realN_in +=1

    #### So far so good. This is the non-averaged density. Now do the averaging over 9 squares.
    for ii in range(X):
        for jj in range(Y):
            if ii==0:
                if jj==0: #left and bottom
                    density_temp[ii,jj] = 4*density[ii,jj]+2*density[ii+1,jj]+2*density[ii,jj+1]+density[ii+1,jj+1]
                elif jj==Y-1: #left and mouth
                    density_temp[ii,jj] = 2*density[ii,jj]+2*density[ii,jj-1]+density[ii+1,jj]+density[ii+1,jj-1]+3*density_outside
                else: #left wall but not bottom
                    density_temp[ii,jj] = 2*density[ii,jj-1]+2*density[ii,jj]+2*density[ii,jj+1]+density[ii+1,jj-1]+density[ii+1,jj]+density[ii+1,jj+1]
            elif ii==X-1:
                if jj==0: #right and bottom
                    density_temp[ii,jj] = 4*density[ii,jj]+2*density[ii-1,jj]+2*density[ii,jj+1]+density[ii-1,jj+1]
                elif jj==Y-1: #right and mouth
                    density_temp[ii,jj] = 2*density[ii,jj]+2*density[ii,jj-1]+density[ii-1,jj]+density[ii-1,jj-1]+3*density_outside
                else: #right wall but not bottom
                    density_temp[ii,jj] = 2*density[ii,jj-1]+2*density[ii,jj]+2*density[ii,jj+1]+density[ii-1,jj-1]+density[ii-1,jj]+density[ii-1,jj+1]
            else: #not left, not right
                if jj==0: #bottom
                    density_temp[ii,jj] = 2*density[ii,jj]+2*density[ii+1,jj]+2*density[ii-1,jj]+density[ii+1,jj+1]+density[ii-1,jj+1]+density[ii,jj+1]
                elif jj==Y-1: #left and mouth
                    density_temp[ii,jj] = density[ii-1,jj-1]+density[ii,jj-1]+density[ii+1,jj-1]+density[ii-1,jj]+density[ii,jj]+density[ii+1,jj]+3*density_outside
                else: #left wall but not bottom
                    density_temp[ii,jj] = density[ii-1,jj-1]+density[ii-1,jj]+density[ii-1,jj+1] +density[ii,jj-1]+density[ii,jj]+density[ii,jj+1] +density[ii+1,jj-1]+density[ii+1,jj]+density[ii+1,jj+1]
    density = density_temp/float(squares) #this seems to be the expensive step

    #so far the density should be in #cells. now we want it in no. of cells per unit area
    density = density/(dx*dx) #local density in cells per unit area
    
    return density,realN_in

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


def calculate_density_profile(positions):
    '''
    Version of the function that wraps everything and takes only the positions array.
    '''

    
    density = np.zeros((round(Lx/dx),round(Ly/dx)))
    for position in positions:
        x,y = position

        bin_x = round(x//dx)
        bin_y = round(y//dx)

        density[bin_x,bin_y] += 1

    density = density/dx**2
    pos_wall = round(Lx/2/dx)
    pos_wall_length=round(5/dx)
    s = 5
    density[:pos_wall,:] = gaussian_filter(density[:pos_wall,:],s)
    density[pos_wall:,:] = gaussian_filter(density[pos_wall:,:],s)
    density[:,pos_wall_length:] = gaussian_filter(density[:,pos_wall_length:],s)


    return density

@njit(cache=False,parallel=False)
def gradient(density,dx,delta_density_x,delta_density_y,X,Y,outside_value): #O(squares*X*Y)
    '''
    Will output two arrays. One for the x-difference and one for the y-difference.
    The gradient of the density is done using center-differences.
    '''
    
    for ii in range(X):
        for jj in range(Y):
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

physical_factor = 376 #to make parameters physical
@njit(cache=False)
def collective_diffusivity_value(rho):
    '''
    We take the collective diffusivity function of hard spheres, normalized to 1 at 0-density.
    '''
    if rho>=rho_jam:
        return float(physical_factor*1000.0*(rho-rho_jam)+ 0.6864147263571838) #it's D_col upon jamming.
    else:
        phi=rho*(diam/2)**2*pi #change from number density to packing fraction.
        #take as hard spheres
        mu = 1 * (1-phi)**5.8
        A = (1+phi+phi**2-phi**3)/(1-phi)**3
        dA = (1+2*phi-3*phi**2)/(1-phi)**3+3/(1-phi)**4*(1+phi+phi**2-phi**3)
        dP = 6/pi*A+6*phi/pi*dA
        out = 1*mu*dP
        return  float(physical_factor*out/1.90985932) #normalize by D_col(phi=0)=1.909859...


@njit(cache=False)
def SingleCell_diffusivity_value(rho):
    '''
    Stupid example where D_singcell = 1 if rho<rho_jam, 0.0001 otherwise
    '''
    
    if rho>=rho_jam:
        return   physical_factor*0.0015*D_0 #*0.0 makes it deterministic when in jammed mode.
    else:
        out = D_0*(1-rho*0.95/rho_jam) #make it such that it's still not exactly zero at rho_jam
        
        return   physical_factor*out 




@njit(cache=False,parallel=False)
def collective_diffusivity_array(D_collective,density,X,Y,N_in): #O(X*Y)
    
    for ii in range(X):
        for jj in range(Y):
            rho = density[ii,jj] #density in cells/area
            D_collective[ii,jj] = collective_diffusivity_value( 
                rho)
    return D_collective

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
@njit(cache=False,parallel=False)
def update_positions(positions,flag,bins,D_singlecell,
                     density,delta_density_x,delta_density_y,D_collective,X,Y,N_in,
                     problem,
                     
                     ):

    D_self_array = np.zeros((X,Y),dtype=np.float64)
    for ii in range(X):
        for jj in range(Y):
            D_self_array[ii,jj] = SingleCell_diffusivity_value(
                density[ii,jj])
            
    
    delta_D_self_x = np.zeros((X,Y),dtype=np.float64)
    delta_D_self_y = np.zeros((X,Y),dtype=np.float64)
    #Calculate grad(D_self) 
    delta_D_self_x,delta_D_self_y = gradient(
        D_self_array,dx,delta_D_self_y,delta_D_self_x,X,Y,D_0)

    ### WALL POSITION AND LENGTH
    for n in range(N_in): #prange(N_in):
        
        if flag[n]==1:#if particle in domain
            ii,jj = bins[n]

            #### COLLECTIVE FLOW
            ####Diffusivites
            diff_collective = D_collective[ii,jj]
            diff_single = D_singlecell[n]

            wall_check = 0 #default. Cell far enough from wall that no care is needed
            if positions[n,1]<wall_length: #length of the invisible wall
                if wall-dx<positions[n,0]<wall:
                    wall_check = -1 #cell slightly to the left of the wall
                elif wall<positions[n,0]<wall+dx:
                    wall_check= 1 #cell slightly to the right of the wall
            #The previous code is slightly erroneous very close to the tip
            #of the invisible wall, but let's assume the error is negligible.


            x_change = 0.0; y_change = 0.0
            
            #BCs. 0 density flux at the walls
            #BCs also 0 flux at the bottle-butt wall
            if dx<positions[n,0]<Lx-dx:
                if positions[n,1]>wall_length: #beyond the invisible wall
                    x_change = -delta_density_x[ii,jj]/density[ii,jj]*(diff_collective-diff_single)*dt
                    #Change due to grad(D_self)
                    x_change += delta_D_self_x[ii,jj]*dt
                elif positions[n,1]<=wall_length: #in the region of the invisible wall
                    if wall-dx<positions[n,0]<wall+dx:
                        pass
                    else:
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

            if wall_check == -1: #cell was left of the wall before
                if positions[n,0]>wall: #but is now on the right
                    positions[n,0] -= 2*(positions[n,0]-wall) #bounce back
            elif wall_check == +1: #cell was right of the wall
                if positions[n,0]<wall: #but is now right
                    positions[n,0] += 2*(wall-positions[n,0]) #bounce back
                
            ########### OJO ##############
            ####### Beware children being born across the wall from the mother. To be handled in birth function.
            

            if x_change>dx or y_change>dx:
                print('RW Change too big',n,x_change,y_change)
            
    return positions,problem

############# GIVE BIRTH
@njit(cache=False)
def give_birth(positions,flag,birth_rate,lineage,N_in,bins,density):
    N_in_temp = N_in
    for n in range(N_in_temp):
        if flag[n]==1: #only if in the box, can it give a child
            #only if in non_jammed situation, can it give birth:
            ii,jj = bins[n]

            rand = np.random.random()
            if rand<birth_rate[n]*dt:

                ### Check if mother cell is close to invisible wall
               wall_check = 0
               if positions[n,1]<wall_length:
                    if positions[n,0] < wall:
                        wall_check = -1
                    elif wall < positions[n,0]:
                        wall_check = 1
               

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

               #check baby's collision with wall
               if wall_check==-1: #mother was to the left
                   if positions[N_in-1,0]>wall: #and child to the right
                       #make it bounce back
                       positions[N_in-1,0] -= 2*(positions[N_in-1,0]-wall)
               if wall_check==1: #if mother to the right
                   if positions[N_in-1,0]<wall: #and child to the left
                       #make it bounce back
                       positions[N_in-1,0] += 2*(wall-positions[N_in-1,0])

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
            if x<0:
                positions[n,0] = -x
            elif x>Lx:
                positions[n,0] = Lx-(x-Lx)
            if y<0:
                positions[n,1] = -y
                
            if y>Ly:
                flag[n] = 0

    return positions,flag

#### Integration
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


    #### Apply gaussian filter to the density to smooth it out.
    
    pos_wall_length = round(wall_length/dx);
    pos_wall = round(wall/dx)

    #Filter in 3 sweeps around the wall (left all, right all, left&right beyond wall).
    ### This is done so the profile is not smoothed across the wall.
    density[:pos_wall,:] = gaussian_filter(density[:pos_wall,:],5)
    density[pos_wall:,:] = gaussian_filter(density[pos_wall:,:],5)
    density[:,pos_wall_length:] = gaussian_filter(density[:,pos_wall_length:],5)
    
    #Find position of jamming wavefront.
    jam_wavefront = 0.0
    epsilon = 0.005
    density_1D = np.mean(density,axis=0)
    zeros = np.where(abs(density_1D-rho_jam)<epsilon)[0]
    if len(zeros)>0:
        jam_wavefront = zeros[-1]
    

    delta_density_x,delta_density_y = gradient(density,dx,delta_density_x,delta_density_y,X,Y,
                                               density_outside)

    D_collective =collective_diffusivity_array(D_collective,density,X,Y,N_in)
    D_singlecell =SingleCell_diffusivity_list(flag,D_singlecell,bins,density,N_in)


    ##### UPDATE POSITIONS
    positions,problem = update_positions(positions,flag,bins,D_singlecell,
                     density,delta_density_x,delta_density_y,D_collective,X,Y,N_in,
                     problem,
                     )
    density_at_00 = density[0,0]


    ##### GIVE BIRTH

    positions,flag,birth_rate,lineage,N_in = give_birth(positions,flag,
                                                        birth_rate,lineage,N_in,
                                                        bins,density,
                                                        )


    ###Apply BCs at the single-cell level: wall rebounds
    positions,flag = elastic_walls_and_checkIfInside(positions,flag,N_in)


    return positions,flag,lineage,N_in,density_at_00,problem,realN_in,jam_wavefront


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
N_max = 120000 #maximum number of allowed particles
    
###### Cell diameter is defined in the __main__ piece of the code at the bottom.
 #the area fraction
                           #occupied by a single cell in a bin
density_outside = 0.0 #density outside the mouth of the cave


out = 0.0
D_0 = 1.0 #single-cell
D =1.0 #collective
squares = 9 #no. of squares over which it'll be averaged.


############### CHOICE OF BIRTH RATE:
b = 0.33 #h^-1 physical parameter from Yuya et al PNAS 2021.



phi_jam = 0.64


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

###########################################                 WALL            ####################################
wall_length = 5 #length of pin
wall = Lx/2 #positions

def simulation(arguments):
    contour_list = [] #list with all inter-strain boundaries 

    arguments,ID = arguments
    
    positions,flag,lineage,birth_rate,N_in,step0,density,total_pop,wr_int =arguments

    
    problem = False
    realN_in = 0 #actual number of particles inside box. Consider that N_in also counts the ones
            #that exited
    
    out_dir = 'out.'+str(ID)
    
    continuation=False
    if os.path.exists(out_dir) and continuation==False: #if cont.=True don't remove existing files!
        shutil.rmtree(out_dir)
        os.mkdir(out_dir)
    elif continuation ==False:
        os.mkdir(out_dir)

    

    ###Define observables of interest
    total_pop_left=[]
    total_pop_right=[]
    resistant_fraction_left = []
    resistant_fraction_right = []
    jam_wavefront_vect = []


    positions,flag,lineage,birth_rate = initialize(positions,flag,lineage,birth_rate,N_in,Lx,Ly,
                                                   b)

    if continuation==False:
        file_positions = os.path.join(out_dir,'initial.positions.'+str(ID)+'.dat')
        with open(file_positions,'wb') as f:
            pickle.dump(positions,f)
            f.close()
        #Output flag
        file_flag = os.path.join(out_dir,'initial.flag.'+str(ID)+'.dat')
        with open(file_flag,'wb') as f:
            pickle.dump(flag,f)
            f.close()
        #Output lineage
        file_lineage = os.path.join(out_dir,'initial.lineage.'+str(ID)+'.dat')
        with open(file_lineage,'wb') as f:
            pickle.dump(lineage,f)
            f.close()
        
    
    if continuation==True:
        '''
        To be used if you do not want to start a simulation from scratch. It will instead take
        the configuration of some old simulation that you've already run. Very useful to skip the
        transient growth until jamming.
        '''
        ####### Also. Binarize the lineage and the birth rate
        last_dir = 'out.'+str(ID)
        pos_out = os.path.join(last_dir,'positions.'+str(ID)+'.dat')
        with open(pos_out,'rb') as f:
            positions_temp = pickle.load(f)
            f.close()
        lin_out = os.path.join(last_dir,'lineage.'+str(ID)+'.dat')
        with open(lin_out,'rb') as f:
            lineage_temp = pickle.load(f)
            f.close()
   
        positions = np.zeros((N_max,2)) #input obviously wrong initial positions
        flag = np.zeros(N_max,dtype=np.int32)
        lineage = np.zeros(N_max,dtype=np.int32)
        birth_rate = np.zeros(N_max)
        ii=0
        old_Lx=40
        for n in range(N_max): 
            if 0<positions_temp[n,1]<Ly and 0<positions_temp[n,0]<old_Lx: #means the particle is initiated and not out
                ''' Mark cell as being IN the cavity '''
                flag[ii] = 1
                ''' Set cell position '''
                positions[ii,0] = positions_temp[n,0] 
                positions[ii,1] = positions_temp[n,1]
                ''' Set cell lineage '''
                lin = lineage_temp[n]

                ii+=1
        N_in = ii


        ''' Print initial Configuration to Output '''
        last_dir = 'out.'+str(ID)
        pos_out = os.path.join(last_dir,'positions.'+str(ID)+'.dat')
        with open(pos_out,'wb') as f:
            pickle.dump(positions,f)
            f.close()
        lin_out = os.path.join(last_dir,'lineage.'+str(ID)+'.dat')
        with open(lin_out,'wb') as f:
            pickle.dump(lineage,f)
            f.close()

        
            
        
        
    frac_resistant = sum(lineage)
    step0=0 #set to the corresponding value
    
    

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
    t_max = 120000
    counter = 0
    binarized_population = False
    
    for step in range(step0,step0+steps+1):
        positions,flag,lineage,N_in,density_at_00,problem,realN_in,jam_wavefront = integrate(
                      positions,bins,density,density_temp,delta_density_x,delta_density_y,
                      D_collective,D_singlecell,
                      flag,N_in,step,realN_in,R,
                      birth_rate,lineage,X,Y,
                      problem
                      )
        density_at_00_vect.append(density_at_00)
        density_array.append(density)
        if problem==True:
            with open('out/total_population','wb') as f:
                pickle.dump(total_pop,f)
                f.close()
            os.exit()
        if N_in>N_max or step*dt>t_max:
            break
                
        
        if step%wr_int==0:
            ### Every wr_int, append to output arrays (but do not write to output)
            
            if (step%(wr_int)==0 and step*dt>0):        
                ####Update state of the cavity
                if step%(1000000)==0:
                #Output positions
                    file_positions = os.path.join(out_dir,'positions.'+str(ID)+'.'+str(step)+'.dat')
                    with open(file_positions,'wb') as f:
                        pickle.dump(positions,f)
                        f.close()
                    #Output flag
                    file_flag = os.path.join(out_dir,'flag.'+str(ID)+'.'+str(step)+'.dat')
                    with open(file_flag,'wb') as f:
                        pickle.dump(flag,f)
                        f.close()
                    #Output lineage
                    file_lineage = os.path.join(out_dir,'lineage.'+str(ID)+'.'+str(step)+'.dat')
                    with open(file_lineage,'wb') as f:
                        pickle.dump(lineage,f)
                        f.close()         
                

    t2 = time.time()
    print('N_in', N_in)
    print('Time Taken: ',t2-t1,' seconds')
    
    return

if __name__=='__main__':


    #### Parallel-ready code ####
    

    how_many_sims = 20

    ### Run only the given task in SLURM
    task_id = int(os.getenv('SLURM_PROCID',0))
    
    # diam =  0.3+1.2*task_id/how_many_sims #from 0.3 to 1.5
    diam = 1.0 
    cell_area = (diam/2)**2*pi
    rho_jam = 0.64/((diam/2)**2*pi) 

    arguments = [(positions,flag,lineage,birth_rate,N_in,step0,density,total_pop,wr_int)]

    
    arguments = itertools.product(arguments,[task_id])
    arguments = list(arguments)[0]
    
    print('Arguments',arguments[0])
    print('Starting task:',task_id)
    
    results = simulation(arguments)
    
    
    














