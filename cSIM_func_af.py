import numpy as np
import matplotlib.pyplot as plt

from numpy.fft import fft2, ifft2, fftshift, ifftshift

import arrayfire as af


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


def af_pad(image, NN, MM, val):
    N,M = image.shape
    Np = N + 2*NN
    Mp = M + 2*MM
    if image.dtype() == af.Dtype.f32 or image.dtype() == af.Dtype.f64:
        image_pad = af.constant(val,Np,Mp)
    else:
        image_pad = af.constant(val*(1+1j*0),Np,Mp)
    image_pad[NN:NN+N,MM:MM+M] = image
    
    return image_pad



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
        
        fxxp = ifftshift(fxxp)
        fyyp = ifftshift(fyyp)
        
        self.fxxp = af.interop.np_to_af_array(fxxp)
        self.fyyp = af.interop.np_to_af_array(fyyp)

        
        # Initialization of object and pattern
        self.obj = np.ones((self.Nc, self.Mc))
        self.obj = af.interop.np_to_af_array(self.obj)
        self.field_p_whole = np.ones((Npp, Mpp))
        
        
        for i in range(0, self.Nimg):
            field_p_shift_back = np.maximum(0,np.real(ifft2(fft2(np.pad(np.pad((Ic_image_up[0,i])**(1/2),(self.N_bound_pad,),mode='constant'),((self.yshift_max,),(self.xshift_max,)), mode='constant'))\
            * np.exp(-1j*2*np.pi*self.ps*(fxxp * self.xshift[0,i] + fyyp * self.yshift[0,i])))))
            self.field_p_whole += field_p_shift_back/self.Nimg
            
        self.field_p_whole = af.interop.np_to_af_array(self.field_p_whole)
        
        # Compute transfer function
        Pupil_obj = np.zeros((self.Nc,self.Mc))
        frc = (fxx_c**2 + fyy_c**2)**(1/2)
        Pupil_obj[frc<NA_obj/lambda_c] = 1
        Pupil_prop_sup = Pupil_obj.copy()
        self.Pupil_obj = af.interop.np_to_af_array(Pupil_obj)
        
        Hz_det = np.zeros((self.N_defocus, self.Nc, self.Mc),complex)

        for i in range(0, self.N_defocus):
            Hz_det[i] = Pupil_prop_sup * np.exp(1j*2*np.pi/lambda_c*z_camera[i]*\
                                                (1-lambda_c**2 * frc**2 *Pupil_prop_sup)**(1/2))
        self.Hz_det = af.interop.np_to_af_array(Hz_det)
        
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
            
        self.zerpoly = af.interop.np_to_af_array(self.zerpoly)
        
        # Compute support function
        self.Pattern_support = np.zeros((Npp,Mpp))
        frp = (fxxp**2 + fyyp**2)**(1/2)
        self.Pattern_support[frp<NAs/lambda_c] = 1
        self.Pattern_support = af.interop.np_to_af_array(self.Pattern_support)

        self.Object_support = np.zeros((self.Nc,self.Mc))
        self.Object_support[frc<(NA_obj+NAs)/lambda_c] = 1
        self.Gaussian = np.exp(-frc**2/(2*((NA_obj + NAs)*Gaussian_width/lambda_c)**2))
        self.Gaussian = (self.Gaussian/np.max(self.Gaussian)).copy()
        self.Gaussian = af.interop.np_to_af_array(self.Gaussian)
        
        
        # iteration error
        self.err = np.zeros(self.itr+1)
    
    def iterative_algorithm(self, Ic_image_up, update_shift=1, shift_alpha=1, update_Pupil=0, Pupil_alpha=1, figsize=(10,10)):
        f1,ax = plt.subplots(2,2,figsize=figsize)
        
        F = lambda x: af.signal.fft2(x)
        iF = lambda x: af.signal.ifft2(x)
        pad = lambda x, pad_y, pad_x: af_pad(x, pad_y, pad_x, 0)
        max = lambda x: af.algorithm.max(af.arith.real(x), dim=None)
        sum = lambda x: af.algorithm.sum(af.algorithm.sum(x, 0), 1)
        angle = lambda x: af.arith.atan2(af.arith.imag(x), af.arith.real(x))
        reshape =  lambda x, N: af.transpose(af.moddims(af.transpose(x), N[1], N[0]))
        
        tic_time = time.time()
        print('|  Iter  |  error  |  Elapsed time (sec)  |')

        for i in range(0,self.itr):

            # sequential update
            for j in range(0,self.Nimg):
                for m in range(0,self.N_defocus):
                
                    fieldp_shift = iF(F(self.field_p_whole) * \
                                          af.arith.exp(1j*2*np.pi*self.ps*(self.fxxp * self.xshift[m,j] + self.fyyp * self.yshift[m,j])))
                    field_p = fieldp_shift[self.yshift_max:self.Nc+self.yshift_max, \
                                           self.xshift_max:self.Mc+self.xshift_max]
                    Ic_image_current_sqrt = af.interop.np_to_af_array((Ic_image_up[m,j])**(1/2))
                    
                    Hz_prop = reshape(self.Hz_det[m],(self.Hz_det.shape[1],self.Hz_det.shape[2]))
                    
                    field_f = F(field_p * self.obj)
                    field_est = iF(Hz_prop * self.Pupil_obj * field_f)
                    field_est_crop_abs = af.arith.abs(field_est[self.N_bound_pad:self.N_bound_pad+self.N,\
                                                          self.N_bound_pad:self.N_bound_pad+self.M])
                    I_sqrt_diff = Ic_image_current_sqrt - field_est_crop_abs
                    residual = F(field_est/(af.arith.abs(field_est)+1e-4) *\
                                    pad(I_sqrt_diff, self.N_bound_pad, self.N_bound_pad))
                    field_temp = iF(af.arith.conjg(self.Pupil_obj * Hz_prop) * residual)
                    
                    # gradient computation
                    
                    grad_obj = -af.arith.conjg(field_p) * field_temp
                    grad_fieldp = -iF(F(pad(af.arith.conjg(self.obj)*field_temp, \
                                                    self.yshift_max, self.xshift_max)) *\
                                        af.arith.exp(-1j*2*np.pi*self.ps*(self.fxxp * self.xshift[m,j] + \
                                                                    self.fyyp * self.yshift[m,j])))
                    
                    if update_Pupil ==1:
                        grad_Pupil = -af.arith.conjg(Hz_prop*field_f)*residual

                    # updating equation
                    self.obj = (self.obj - grad_obj/(max(af.arith.abs(field_p))**2)).copy()
                    self.field_p_whole = (self.field_p_whole - grad_fieldp/(max(af.arith.abs(self.obj))**2)).copy()

                    if update_Pupil ==1:
                        self.Pupil_obj = (self.Pupil_obj - grad_Pupil/max(af.arith.abs(field_f)) * \
                             af.arith.abs(field_f) / (af.arith.pow(af.arith.abs(field_f),2) + 1e-3) * Pupil_alpha).copy()

                    # shift estimate
                    if update_shift ==1:
                        Ip_shift_fx = iF(F(self.field_p_whole) * (1j*2*np.pi*self.fxxp) * \
                                           af.arith.exp(1j*2*np.pi*self.ps*(self.fxxp * self.xshift[m,j] + self.fyyp * self.yshift[m,j])))
                        Ip_shift_fy = iF(F(self.field_p_whole) * (1j*2*np.pi*self.fyyp) * \
                                           af.arith.exp(1j*2*np.pi*self.ps*(self.fxxp * self.xshift[m,j] + self.fyyp * self.yshift[m,j])))
                        Ip_shift_fx = Ip_shift_fx[self.yshift_max:self.yshift_max+self.Nc,\
                                                  self.xshift_max:self.xshift_max+self.Mc]
                        Ip_shift_fy = Ip_shift_fy[self.yshift_max:self.yshift_max+self.Nc,\
                                                  self.xshift_max:self.xshift_max+self.Mc]

                        grad_xshift = -af.arith.real(sum(af.arith.conjg(field_temp) * self.obj * Ip_shift_fx))
                        grad_yshift = -af.arith.real(sum(af.arith.conjg(field_temp) * self.obj * Ip_shift_fy))

                        self.xshift[m,j] = (self.xshift[m,j] - np.array(grad_xshift\
                                            /self.N/self.M/(max(af.arith.abs(self.obj))**2)) * shift_alpha).copy()
                        self.yshift[m,j] = (self.yshift[m,j] - np.array(grad_yshift\
                                            /self.N/self.M/(max(af.arith.abs(self.obj))**2)) * shift_alpha).copy()

                    self.err[i+1] += np.array(sum(af.arith.abs(I_sqrt_diff)**2))

            self.obj = (iF(F(self.obj) * self.Gaussian)).copy()
            self.field_p_whole = (iF(F(self.field_p_whole) * self.Pattern_support)).copy()
            
            if update_Pupil==1:
                Pupil_angle = angle(self.Pupil_obj)
                zerpoly_1 = reshape(self.zerpoly[1],(self.zerpoly.shape[1],self.zerpoly.shape[2]))
                zerpoly_2 = reshape(self.zerpoly[2],(self.zerpoly.shape[1],self.zerpoly.shape[2]))
                
                Pupil_angle = (Pupil_angle - np.array(sum(Pupil_angle*zerpoly_1)/sum(af.arith.pow(zerpoly_1,2)))[0]\
                *zerpoly_1 - np.array(sum(Pupil_angle*zerpoly_2)/sum(af.arith.pow(zerpoly_2,2)))[0]\
                *zerpoly_2).copy()
                self.Pupil_obj = (af.arith.abs(self.Pupil_obj) * af.arith.exp(1j*Pupil_angle)).copy()
                

            if np.mod(i,1) == 0:
                print('|  %d  |  %.2e  |   %.2f   |'%(i+1,self.err[i+1],time.time()-tic_time))
                if i != 0:
                    ax[0,0].cla()
                    ax[0,1].cla()
                    ax[1,0].cla()
                    ax[1,1].cla()
                ax[0,0].imshow(angle(self.obj),cmap='gray');
                ax[0,1].imshow(af.arith.pow(af.arith.abs(self.field_p_whole),2),cmap='gray')
                ax[1,0].imshow(fftshift(np.array(angle(self.Pupil_obj))))
                ax[1,1].plot(self.xshift[0],self.yshift[0],'w')
                ax[1,1].plot(self.xshift[1],self.yshift[1],'y')
                display.display(f1)
                display.clear_output(wait=True)
                time.sleep(0.0001)
                if i == self.itr-1:
                    print('|  %d  |  %.2e  |   %.2f   |'%(i+1,self.err[i+1],time.time()-tic_time))


