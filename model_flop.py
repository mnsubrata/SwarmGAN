from fvcore.nn import FlopCountAnalysis
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms as T
device='cuda' if torch.cuda.is_available() else 'cpu'
trf=T.Compose([T.ToTensor(),T.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
ref_img=trf(Image.open('pp_lat_256.png'))
ref_img=(2*((ref_img-ref_img.min())/(ref_img.max()-ref_img.min()))-1).to(device) #normalize to [-1 and 1]
mu_n_var=torch.load('VAE_mean_mu_var_CELEBAHQ_LAT_256.pth')
mu,logvar=mu_n_var['mu_mean'],mu_n_var['logvar_mean']
kl_div=-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())

class FWA_OBJ(nn.Module):
    def __init__(self, *args, **kwargs):
        super(FWA_OBJ,self).__init__(*args, **kwargs)
        
    def calculate_objectives(self,fp):
        shp=fp.shape
        if len(shp)>3:
            target=ref_img.expand(shp[0],*list(ref_img.size()))
        else:
            target=ref_img
        rec_loss=torch.nn.functional.mse_loss(fp,target)
        err=rec_loss+kl_div
        return err.item()

    def FWA(self,model_G,*args):
        
        n_fireworks = 3         # how many fireworks in the swarm, aka how many solutions
        fitcount=0   #fitness count.
        FES=200
        best=np.finfo(float).max
        ######fireworks=np.random.standard_normal((n_fireworks,nz))
        fireworks=np.random.uniform(-1,1,(n_fireworks,512)) #GENERATING NOISE AS FIREWORKS TO BE OPTIMIZE
        nz=fireworks.shape[1]
        best_candidate=np.zeros((1,nz))
        best_fitness=np.zeros((FES,1))
        Xmin=np.min(fireworks) #lower bound of the variable
        Xmax=np.max(fireworks) #upper bound of the variable
        a=0.04
        b=0.8
        max_m=20 #######20
        a_hat=Xmax-Xmin
        m_min=np.round(a*max_m)  #minimum no of sparks
        m_max=np.round(b*max_m)  #maximum no of sparks
        
        #####======generation of initial fireworks=====#####
        objective_values=np.zeros((n_fireworks))
        img=[]
        for i in range(n_fireworks):
            with torch.no_grad():
                ltnt=torch.FloatTensor(np.expand_dims(fireworks[i,:],0)).to(device=device)
                # img=model_G(ltnt, label, truncation_psi=truncation_psi, noise_mode=noise_mode).squeeze(0).clamp(min=-1, max=1)
                img=model_G(ltnt,*args).squeeze(0).clamp(min=-1, max=1)
            objective_values[i]=self.calculate_objectives(img)

        ######### best_candidate and best fitness assignment
            if objective_values[i]< best:#as it is minimization function desired solution will always should minimum than previous best
                best=objective_values[i]
                best_candidate=fireworks[i,:]
            best_fitness[fitcount]=objective_values[i]
            if fitcount > 0:
                if best_fitness[fitcount] >  best_fitness[fitcount-1]:
                    best_fitness[fitcount] =  best_fitness[fitcount-1] #as it is minimization function desired solution will always should minimum than previous best
            fitcount+=1


        ## sorting the fireworks asecnding order of the objective values
        arr1_idx=np.argsort(objective_values)
        fireworks=fireworks[arr1_idx]
        objective_values=objective_values[arr1_idx]

        while fitcount<FES:

            ####========explosion spark generation=======####

            # objective_values=minmax_scale(objective_values,feature_range=(-1,1)) #fireworks paper eq 8
            objective_values=2*((objective_values-objective_values.min())/(objective_values.max()-objective_values.min()))-1 #normalize to [-1 and 1] priya fireworks paper eq 8
            diff_fit=np.diff(objective_values) #fireworks paper eq 9
            diff_fit=np.append(np.array([0]),diff_fit) # 0 is append to 0th element of diff_fit
            score=np.cumsum(diff_fit)+diff_fit  #fireworks paper eq 10
            transfer_fun=1/(1+np.exp(score))  #fireworks paper eq 11
            num_explosion_spark=np.round(max_m*(transfer_fun/sum(transfer_fun))) #fireworks paper eq 6 number of explosion sparks
            num_explosion_spark=np.minimum(np.maximum(num_explosion_spark,m_min),m_max).astype(int) # select minimum and max between array and global parameter
            #fireworks paper eq 7 amplitude of explosion sparks
            ampli_expl_spark=np.array([a_hat*transfer_fun[n_fireworks-(i[0]+1)]/sum(transfer_fun) for i in enumerate(transfer_fun)])
            sum_no_explosion_spark=np.sum(num_explosion_spark)

            #=========generating location of explosion spark========#
            explosion_X=np.zeros((sum_no_explosion_spark,nz))
            obj_val_explosion_spark=np.zeros((sum_no_explosion_spark))
            index=0
            for i in range(n_fireworks):
                for j in range(num_explosion_spark[i]):
                    if fitcount<FES:
                        explosion_X[index,:]=fireworks[i,:]
                        z=int(np.random.randint(0,nz-1))#no of dimension to be oscillated(it will oscillate in some selected dimensions within D)
                        pos=np.random.uniform(0,nz-1,z).astype(int)  #position to be oscillated
                    
                        offset=np.random.uniform(-1,1,(nz))*ampli_expl_spark[i]
                    
                        explosion_X[index,pos]=explosion_X[index,pos]+ offset[pos]
                        #-------boundary condition checking------#
                        temp=(explosion_X[index,:] < Xmin) | (explosion_X[index,:]>Xmax)
                        temp_sol=Xmin+np.remainder(abs(explosion_X[index,:]),a_hat)
                        explosion_X[index,temp]=temp_sol[temp]


                        with torch.no_grad():
                            ltnt=torch.FloatTensor(np.expand_dims(explosion_X[index,:],0)).to(device=device)
                            img=model_G(ltnt,*args).squeeze(0).clamp(min=-1, max=1)
                        obj_val_explosion_spark[index]=self.calculate_objectives(img)


            #===========best_candidate and best fitness assignment =============#

                        if obj_val_explosion_spark[index]< best:
                            best=obj_val_explosion_spark[index]
                            best_candidate=explosion_X[index,:]

                        best_fitness[fitcount]=obj_val_explosion_spark[i]
                        if best_fitness[fitcount] >  best_fitness[fitcount-1]:
                            best_fitness[fitcount]=best_fitness[fitcount-1]

                        fitcount+=1
                        index+=1

            #=======generating location of gaussian spark=======#
            gaussian_X=np.zeros((n_fireworks,nz))
            obj_val_gaussian_spark=np.zeros((n_fireworks))
            index=0
            for i in range(n_fireworks):
                if fitcount<FES:
                    select=int(np.random.randint(0,n_fireworks))
                    gaussian_X[index,:]=fireworks[select,:]
                    z=int(np.random.randint(0,nz-1))#no of dimension to be oscillated(it will oscillate in some selected dimensions within D)
                    pos=np.random.uniform(0,nz-1,z).astype(int)  #position to be oscillated
                    offset=Xmin+(Xmax-Xmin)*np.random.normal(1,1,nz)
                    gaussian_X[index,pos]= offset[pos]
                    
                    with torch.no_grad():
                        ltnt=torch.FloatTensor(np.expand_dims(gaussian_X[index,:],0)).to(device=device)
                        img=model_G(ltnt, *args).squeeze(0).clamp(min=-1, max=1)
                        obj_val_explosion_spark[index]=self.calculate_objectives(img)

            #===========best_candidate and best fitness assignment =============#
                    if obj_val_gaussian_spark[index]< best:
                            best=obj_val_gaussian_spark[index]
                            best_candidate=explosion_X[index,:]

                    best_fitness[fitcount]=obj_val_gaussian_spark[i]
                    if best_fitness[fitcount] >  best_fitness[fitcount-1]:
                        best_fitness[fitcount]=best_fitness[fitcount-1]

                    fitcount+=1
                    index+=1

            all_fireworks=np.vstack((fireworks,explosion_X,gaussian_X))
            all_objective_values=np.hstack((objective_values,obj_val_explosion_spark,obj_val_gaussian_spark))
            ####========selection of location from firework and sparks=======####
            sorted_objectives_idx=np.argsort(all_objective_values)
            all_fireworks=all_fireworks[sorted_objectives_idx][:n_fireworks,:]
            all_objective_values=all_objective_values[sorted_objectives_idx][:n_fireworks]
        return best_candidate


if __name__=='__main__':
    import dnnlib
    import legacy
    import time
    network_pkl='stylegan2-celebahq-256x256.pkl'
    seeds=0
    truncation_psi=1
    noise_mode='const'
    outdir='out'
    class_idx=None
    projected_w=None

    print(f'Loading networks from {network_pkl}')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    label = torch.zeros([1, G.c_dim], device=device)
    params=[label,truncation_psi,noise_mode]
    F_obj=FWA_OBJ()
    start_time = time.time()
    latent=F_obj.FWA(G,*params)
    end_time=time.time()
    print("TIME TAKEN:",(end_time-start_time))
    print("FLOPs:",FlopCountAnalysis(F_obj))
    pp=10