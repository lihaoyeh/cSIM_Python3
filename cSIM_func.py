import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft, ifft, fft2, ifft2, fftshift, ifftshift

from IPython import display
import time
import pickle
from dftregistration import dftregistration
from zernfun import zernfun, cart2pol


def image_upsampling(Ic_image, upsamp_factor = 1, bg = 0):
    F = lambda x: ifftshift(fft2(fftshift(x)))
    iF = lambda x: ifftshift(ifft2(fftshift(x)))
    
    N_defocus, Nimg, Ncrop, Mcrop = Ic_image.shape

    N = Ncrop*upsamp_factor
    M = Mcrop*upsamp_factor

    Ic_image_up = np.zeros((N_defocus,Nimg,N,M))
    
    for i in range(0,Nimg):
        for j in range(0, N_defocus):
            Ic_image_up[j,i] = abs(iF(np.pad(F(np.maximum(0,Ic_image[j,i]-bg)),\
                                      (((N-Ncrop)//2,),((M-Mcrop)//2,)),mode='constant')))
        
    return Ic_image_up


def display_image_movie(image_stack, frame_num, size, pause_time=0.0001):
    f1,ax = plt.subplots(1,1,figsize=size)
    max_val = np.max(image_stack)

    for i in range(0,frame_num):
        if i != 1:
            ax.cla()
        ax.imshow(image_stack[i],cmap='gray',vmin=0,vmax=max_val)
        display.display(f1)
        display.clear_output(wait=True)
        time.sleep(pause_time)
        
        
        
def image_registration(img_stack,usfac, img_up):
    N_defocus,Nimg,_,_ = img_stack.shape
    xshift = np.zeros((N_defocus,Nimg))
    yshift = np.zeros((N_defocus,Nimg))

    for i in range(0, Nimg):
        for j in range(0, N_defocus):
            if i == 0:
                yshift[j,i] == 0
                xshift[j,i] == 0
            else:
                output = dftregistration(fft2(img_stack[j,0]),fft2(img_stack[j,i]),usfac)
                yshift[j,i] = output[0] * img_up
                xshift[j,i] = output[1] * img_up
            
    return xshift, yshift


def rotate(obj, theta):
    Ncrop, Mcrop = obj.shape
    obj = np.pad(obj, ((Ncrop//2,),(Mcrop//2,)), mode='constant')
    N,M = obj.shape

    x = np.r_[0:M]-M//2
    y = np.r_[0:N]-N//2

    fx = ifftshift(x/M)
    fy = ifftshift(y/N)
    
    
    
    if abs(theta) <= np.pi/2:
    
        alpha = -np.tan(theta/2)
        beta = np.sin(theta)
        gamma = -np.tan(theta/2)

        obj_rot = ifft(fft(obj,axis=0)*np.exp(-1j*2*np.pi*alpha*fy.reshape(N,1).dot(x.reshape(1,M))),axis=0)
        obj_rot = ifft(fft(obj_rot,axis=1)*np.exp(-1j*2*np.pi*beta*y.reshape(N,1).dot(fx.reshape(1,M))),axis=1)
        obj_rot = ifft(fft(obj_rot,axis=0)*np.exp(-1j*2*np.pi*gamma*fy.reshape(N,1).dot(x.reshape(1,M))),axis=0)
    elif theta >0: 

        alpha = -np.tan((np.pi/2)/2)
        beta = np.sin(np.pi/2)
        gamma = -np.tan((np.pi/2)/2)

        obj_rot = ifft(fft(obj,axis=0)*np.exp(-1j*2*np.pi*alpha*fy.reshape(N,1).dot(x.reshape(1,M))),axis=0)
        obj_rot = ifft(fft(obj_rot,axis=1)*np.exp(-1j*2*np.pi*beta*y.reshape(N,1).dot(fx.reshape(1,M))),axis=1)
        obj_rot = ifft(fft(obj_rot,axis=0)*np.exp(-1j*2*np.pi*gamma*fy.reshape(N,1).dot(x.reshape(1,M))),axis=0)

        alpha = -np.tan((theta-np.pi/2)/2)
        beta = np.sin(theta-np.pi/2)
        gamma = -np.tan((theta-np.pi/2)/2)

        obj_rot = ifft(fft(obj_rot,axis=0)*np.exp(-1j*2*np.pi*alpha*fy.reshape(N,1).dot(x.reshape(1,M))),axis=0)
        obj_rot = ifft(fft(obj_rot,axis=1)*np.exp(-1j*2*np.pi*beta*y.reshape(N,1).dot(fx.reshape(1,M))),axis=1)
        obj_rot = ifft(fft(obj_rot,axis=0)*np.exp(-1j*2*np.pi*gamma*fy.reshape(N,1).dot(x.reshape(1,M))),axis=0)
    else: 

        alpha = -np.tan(-(np.pi/2)/2)
        beta = np.sin(-np.pi/2)
        gamma = -np.tan(-(np.pi/2)/2)

        obj_rot = ifft(fft(obj,axis=0)*np.exp(-1j*2*np.pi*alpha*fy.reshape(N,1).dot(x.reshape(1,M))),axis=0)
        obj_rot = ifft(fft(obj_rot,axis=1)*np.exp(-1j*2*np.pi*beta*y.reshape(N,1).dot(fx.reshape(1,M))),axis=1)
        obj_rot = ifft(fft(obj_rot,axis=0)*np.exp(-1j*2*np.pi*gamma*fy.reshape(N,1).dot(x.reshape(1,M))),axis=0)

        alpha = -np.tan((theta+np.pi/2)/2)
        beta = np.sin(theta+np.pi/2)
        gamma = -np.tan((theta+np.pi/2)/2)

        obj_rot = ifft(fft(obj_rot,axis=0)*np.exp(-1j*2*np.pi*alpha*fy.reshape(N,1).dot(x.reshape(1,M))),axis=0)
        obj_rot = ifft(fft(obj_rot,axis=1)*np.exp(-1j*2*np.pi*beta*y.reshape(N,1).dot(fx.reshape(1,M))),axis=1)
        obj_rot = ifft(fft(obj_rot,axis=0)*np.exp(-1j*2*np.pi*gamma*fy.reshape(N,1).dot(x.reshape(1,M))),axis=0)


    return obj_rot[N//2-Ncrop//2:N//2+Ncrop//2,M//2-Mcrop//2:M//2+Mcrop//2]


class cSIM_solver:
    
    def __init__(self, Ic_image_up, xshift, yshift, N_bound_pad, lambda_c, pscrop, z_camera, upsamp_factor, NA_obj, NAs, Gaussian_width, itr):
        
        # Basic parameter 
        self.N_defocus, self.Nimg, self.N, self.M = Ic_image_up.shape
        self.N_bound_pad = N_bound_pad
        self.Nc = self.N + 2*N_bound_pad
        self.Mc = self.M + 2*N_bound_pad
        self.ps = pscrop/upsamp_factor
        self.itr = itr
        
        # Shift variable
        self.xshift = xshift.copy()
        self.yshift = yshift.copy()        
        self.xshift_max = np.int(np.round(np.max(abs(xshift))))
        self.yshift_max = np.int(np.round(np.max(abs(yshift))))
        
        
        # Frequency grid definition to create TF
        fx_c = np.r_[-self.Mc/2:self.Mc/2]/self.ps/self.Mc
        fy_c = np.r_[-self.Nc/2:self.Nc/2]/self.ps/self.Nc

        fxx_c, fyy_c = np.meshgrid(fx_c,fy_c)

        fxx_c = ifftshift(fxx_c)
        fyy_c = ifftshift(fyy_c)
        
        Npp = self.Nc + 2*self.yshift_max
        Mpp = self.Mc + 2*self.xshift_max


        fxp = np.r_[-Mpp/2:Mpp/2]/self.ps/Mpp
        fyp = np.r_[-Npp/2:Npp/2]/self.ps/Npp

        fxxp, fyyp = np.meshgrid(fxp,fyp)

        self.fxxp = ifftshift(fxxp)
        self.fyyp = ifftshift(fyyp)

        
        # Initialization of object and pattern
        self.obj = np.ones((self.Nc, self.Mc))
        self.field_p_whole = np.ones((Npp, Mpp))
        
        for i in range(0, self.Nimg):
            field_p_shift_back = np.maximum(0,np.real(ifft2(fft2(np.pad(np.pad((Ic_image_up[0,i])**(1/2),(self.N_bound_pad,),mode='constant'),((self.yshift_max,),(self.xshift_max,)), mode='constant'))\
            * np.exp(-1j*2*np.pi*self.ps*(self.fxxp * self.xshift[0,i] + self.fyyp * self.yshift[0,i])))))
            self.field_p_whole += field_p_shift_back/self.Nimg
        
        # Compute transfer function
        Pupil_obj = np.zeros((self.Nc,self.Mc))
        frc = (fxx_c**2 + fyy_c**2)**(1/2)
        Pupil_obj[frc<NA_obj/lambda_c] = 1
        Pupil_prop_sup = Pupil_obj.copy()
        self.Pupil_obj = Pupil_obj.copy()
        
        Hz_det = np.zeros((self.N_defocus, self.Nc, self.Mc),complex)

        for i in range(0, self.N_defocus):
            Hz_det[i] = Pupil_prop_sup * np.exp(1j*2*np.pi/lambda_c*z_camera[i]*\
                                                (1-lambda_c**2 * frc**2 *Pupil_prop_sup)**(1/2))
        self.Hz_det = Hz_det.copy()
        
        # Set up Zernike polynomials
        
        n_idx = np.array([0,1,1,2,2,2,3,3,3,3,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6,6,6])
        m_idx = np.array([0,-1,1,-2,0,2,-3,-1,1,3,-4,-2,-0,2,4,-5,-3,-1,1,3,5,-6,-4,-2,0,2,4,6])

        N_poly = len(n_idx)

        self.zerpoly = np.zeros((N_poly,self.Nc,self.Mc))

        [rr, theta_theta] = cart2pol(fxx_c/NA_obj*lambda_c,fyy_c/NA_obj*lambda_c)
        idx = rr<=1

        for i in range(0,N_poly):
            z = np.zeros_like(fxx_c)
            temp = zernfun(n_idx[i],m_idx[i],rr[idx],theta_theta[idx])
            z[idx] = temp.ravel()
            self.zerpoly[i] = z/np.max(z)
        
        # Compute support function
        self.Pattern_support = np.zeros((Npp,Mpp))
        frp = (self.fxxp**2 + self.fyyp**2)**(1/2)
        self.Pattern_support[frp<NAs/lambda_c] = 1

        self.Object_support = np.zeros((self.Nc,self.Mc))
        self.Object_support[frc<(NA_obj+NAs)/lambda_c] = 1
        self.Gaussian = np.exp(-frc**2/(2*((NA_obj + NAs)*Gaussian_width/lambda_c)**2))
        self.Gaussian = (self.Gaussian/np.max(self.Gaussian)).copy()
        
        
        # iteration error
        self.err = np.zeros(self.itr+1)
    
    def iterative_algorithm(self, Ic_image_up, update_shift=1, shift_alpha=1, update_Pupil=0, Pupil_alpha=1, figsize=(10,10)):
        f1,ax = plt.subplots(2,2,figsize=figsize)

        tic_time = time.time()
        print('|  Iter  |  error  |  Elapsed time (sec)  |')

        for i in range(0,self.itr):

            # sequential update
            for j in range(0,self.Nimg):
                for m in range(0,self.N_defocus):
                
                    fieldp_shift = ifft2(fft2(self.field_p_whole) * \
                                          np.exp(1j*2*np.pi*self.ps*(self.fxxp * self.xshift[m,j] + self.fyyp * self.yshift[m,j])))
                    field_p = fieldp_shift[self.yshift_max:self.Nc+self.yshift_max, \
                                           self.xshift_max:self.Mc+self.xshift_max]
                    Ic_image_current_sqrt = (Ic_image_up[m,j])**(1/2)
                    
                    field_f = fft2(field_p * self.obj)
                    field_est = ifft2(self.Hz_det[m] * self.Pupil_obj * field_f)
                    field_est_crop_abs = np.abs(field_est[self.N_bound_pad:self.N_bound_pad+self.N,\
                                                          self.N_bound_pad:self.N_bound_pad+self.M])
                    I_sqrt_diff = Ic_image_current_sqrt - field_est_crop_abs
                    residual = fft2(field_est/(np.abs(field_est)+1e-4) *\
                                    np.pad(I_sqrt_diff, (self.N_bound_pad,), mode='constant'))
                    field_temp = ifft2(np.conj(self.Pupil_obj * self.Hz_det[m]) * residual)
                    
                    # gradient computation
                    
                    grad_obj = -np.conj(field_p) * field_temp
                    grad_fieldp = -ifft2(fft2(np.pad(np.conj(self.obj)*field_temp, \
                                                    ((self.yshift_max,),(self.xshift_max,)), mode='constant')) *\
                                        np.exp(-1j*2*np.pi*self.ps*(self.fxxp * self.xshift[m,j] + \
                                                                    self.fyyp * self.yshift[m,j])))
                    
                    if update_Pupil ==1:
                        grad_Pupil = -np.conj(self.Hz_det[m]*field_f)*residual

                    # updating equation
                    self.obj = (self.obj - grad_obj/(np.max(np.abs(field_p))**2)).copy()
                    self.field_p_whole = (self.field_p_whole - grad_fieldp/(np.max(np.abs(self.obj))**2)).copy()

                    if update_Pupil ==1:
                        self.Pupil_obj = (self.Pupil_obj - grad_Pupil/np.max(abs(field_f)) * \
                             abs(field_f) / (abs(field_f)**2 + 1e-3) * Pupil_alpha).copy()

                    # shift estimate
                    if update_shift ==1:
                        Ip_shift_fx = ifft2(fft2(self.field_p_whole) * (1j*2*np.pi*self.fxxp) * \
                                           np.exp(1j*2*np.pi*self.ps*(self.fxxp * self.xshift[m,j] + self.fyyp * self.yshift[m,j])))
                        Ip_shift_fy = ifft2(fft2(self.field_p_whole) * (1j*2*np.pi*self.fyyp) * \
                                           np.exp(1j*2*np.pi*self.ps*(self.fxxp * self.xshift[m,j] + self.fyyp * self.yshift[m,j])))
                        Ip_shift_fx = Ip_shift_fx[self.yshift_max:self.yshift_max+self.Nc,\
                                                  self.xshift_max:self.xshift_max+self.Mc]
                        Ip_shift_fy = Ip_shift_fy[self.yshift_max:self.yshift_max+self.Nc,\
                                                  self.xshift_max:self.xshift_max+self.Mc]

                        grad_xshift = -np.real(np.sum(np.conj(field_temp) * self.obj * Ip_shift_fx))
                        grad_yshift = -np.real(np.sum(np.conj(field_temp) * self.obj * Ip_shift_fy))

                        self.xshift[m,j] = (self.xshift[m,j] - grad_xshift\
                                            /self.N/self.M/(np.max(np.abs(self.obj))**2) * shift_alpha).copy()
                        self.yshift[m,j] = (self.yshift[m,j] - grad_yshift\
                                            /self.N/self.M/(np.max(np.abs(self.obj))**2) * shift_alpha).copy()

                    self.err[i+1] += np.sum(np.abs(I_sqrt_diff)**2)

            self.obj = ifft2(fft2(self.obj) * self.Gaussian)
            self.field_p_whole = ifft2(fft2(self.field_p_whole) * self.Pattern_support)
            
            if update_Pupil==1:
                Pupil_angle = np.angle(self.Pupil_obj)
                Pupil_angle = (Pupil_angle - np.sum(Pupil_angle*self.zerpoly[1])/np.sum(self.zerpoly[1]**2)\
                *self.zerpoly[1] - np.sum(Pupil_angle*self.zerpoly[2])/np.sum(self.zerpoly[2]**2)\
                *self.zerpoly[2]).copy()
                self.Pupil_obj = np.abs(self.Pupil_obj) * np.exp(1j*Pupil_angle)
                

            if np.mod(i,1) == 0:
                print('|  %d  |  %.2e  |   %.2f   |'%(i+1,self.err[i+1],time.time()-tic_time))
                if i != 0:
                    ax[0,0].cla()
                    ax[0,1].cla()
                    ax[1,0].cla()
                    ax[1,1].cla()
                ax[0,0].imshow(np.angle(self.obj),cmap='gray');
                ax[0,1].imshow(np.abs(self.field_p_whole)**2,cmap='gray')
                ax[1,0].imshow(fftshift(np.angle(self.Pupil_obj)))
                ax[1,1].plot(self.xshift[0],self.yshift[0],'w')
                ax[1,1].plot(self.xshift[1],self.yshift[1],'y')
                display.display(f1)
                display.clear_output(wait=True)
                time.sleep(0.0001)
                if i == self.itr-1:
                    print('|  %d  |  %.2e  |   %.2f   |'%(i+1,self.err[i+1],time.time()-tic_time))


