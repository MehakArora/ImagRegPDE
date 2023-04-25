# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:26:55 2023

@author: mehak
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage


def rotate(image, angle, center = None, scale = 1.0):
    """ 
    image: image to be transformed
    angle: angle of rotation
    center: translation in (x,y) - center of image
    scale: sclaing factor
    """
    
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)
        print(center)

    # Perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated, M


def V(x, y):
    #Function to get deformation field
    point = np.array([x, y])
    V = R.T @ (point - center) + center 
    return V

def derivative(img, dim):
    """
    Parameters
    ----------
    img : numpy array
        input image
    dim : numpy array
        axis of the derivative - 0 for x axis, 1 for y axis

    Returns
    -------
    Id : numpy array
        the first order derivative of img (with reflection on boundary points)
    """ 
    Id = np.zeros(img.shape)
    if dim == 0:
        for n in range(img.shape[1]):
            if n != img.shape[1] - 1:
                Id[:, n] = (img[:, n+1] - img[:, n])
            else:
                #reflect the last column
                Id[:, n] = (img[:,n] - img[:,n])
    elif dim == 1:
        for n in range(img.shape[0]):
            if n != img.shape[0] - 1:
                Id[n, :] = (img[n+1, :] - img[n, :])
            else:
                #Reflect the last row
                Id[n, :] = (img[n,:] - img[n, :])
            
    return Id

def second_derivative(img, dim):
    """
    Parameters
    ----------
    img : numpy array
        input image
    dim : numpy array
        axis of the derivative - 0 for x axis, 1 for y axis

    Returns
    -------
    Id : numpy array
        the second order derivative (central difference) of img (with reflection on boundary points)

    """
    Id = np.zeros(img.shape)
    
    if dim == 0:
        for n in range(img.shape[1]):
            if n != img.shape[1] - 1 and n > 0:
                Id[:, n] = (img[:, n+1] + img[:, n-1] - 2*img[:, n])/2
            elif n == img.shape[1] - 1:
                #reflect the last column
                Id[:, n] = (img[:,n] + img[:,n-1] - 2*img[:,n])/2
            elif n == 0:
                #Reflect first column
                Id[:, n] = (img[:, n+1] + img[:, n] - 2*img[:,n])/2
    elif dim == 1:
        for n in range(img.shape[0]):
            if n != img.shape[0] - 1 and n > 0:
                Id[n, :] = (img[n+1, :]  + img[n-1, :] - 2*img[n, :])/2
            elif n == img.shape[0] - 1:
                #reflect the last column
                Id[n,:] = (img[n,:] + img[n-1,:] - 2*img[n,:])/2
            elif n == 0:
                #Reflect first column
                Id[n, :] = (img[n+1,:] + img[n,:] - 2*img[n,:])/2
            
    return Id

def descretized_laplacian(img):
    uxx = second_derivative(img, 0) 
    uyy = second_derivative(img, 1) 
    return uxx + uyy

if __name__ == '__main__':
    #read image
    img = cv2.imread('normal-ct-brain.jpg')[:,:,0]/255.
    plt.imshow(img, cmap = 'gray')
    
    #Rotate, translate
    theta = -(15/360)*np.pi*2 #Angle of rotation
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]) #Rotation Matrix
    T = np.array([0, 0]) #Translation matrix
    center = np.array([img.shape[0]/2, img.shape[1]/2])

    deformation_x = np.zeros(img.shape)
    deformation_y = np.zeros(img.shape)

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            deformation_x[x, y] = #np.sin(x) + np.cos(y) #V(x,y)[0]
            deformation_y[x, y] = #np.cos(x) + np.sin(y) #V(x,y)[1]
            
    mapx_base, mapy_base = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    mapx = deformation_y
    mapy = deformation_x 

    #Applying the deformation using linear interpolation
    deformed = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR)
    
    f, a = plt.subplots(1, 4, figsize = (10,10))
    a[0].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    a[1].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    a[2].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    a[3].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    a[0].imshow(deformation_x, cmap = 'gray')
    a[0].set_title('Ground Truth u(x,y)')
    a[1].imshow(deformation_y, cmap = 'gray')
    a[1].set_title('Ground Truth v(x,y)')
    a[2].imshow(deformed, cmap = 'gray')    
    a[2].set_title('Deformed')
    a[3].imshow(img, cmap = 'gray')    
    a[3].set_title('Original')
    #f.savefig('example.pdf', dpi = 1000, bbox_inches = 'tight')
    
    #Update PDEs
    
    img_new = deformed
    uNew = np.zeros(img.shape)
    vNew = np.zeros(img.shape)
    deformedOld = np.copy(img)
    img_target = np.copy(img_new)
    num_iterations = 8000
    lamda = 0.7
    del_t = 0.5
    rmse = []
    energy = []

    mapx_base, mapy_base = np.meshgrid(np.arange(img.shape[0]), np.arange(img.shape[1]))
    for iteration in range(num_iterations):
        uOld = uNew.copy()
        vOld = vNew.copy()
        
        uNew = np.zeros(img.shape)
        vNew = np.zeros(img.shape)
        
        mapx =  mapx_base + uOld
        mapy =  mapy_base + vOld
        
        deformed = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR)
        data_fidelity_term = lamda*(deformed - img_target)
        rmse_ = np.sqrt(np.linalg.norm(img_target - deformed)**2)/(img.shape[0]*img.shape[1])
        rmse.append(rmse_)

        
        
        Ix = derivative(deformed, 0)
        Iy = derivative(deformed, 1)
        ux = derivative(uOld, 0)
        uy = derivative(uOld, 1)
        vx = derivative(uOld, 0)
        vy = derivative(uOld, 1)
        
        d = 1- (Ix - Ix.min())/(Ix.max() - Ix.min()) - (Iy - Iy.min())/(Iy.max() - Iy.min())
        
        energy_ = 0.5*lamda*np.linalg.norm(img_target - deformed)**2 + 0.5*(1-lamda)*(np.linalg.norm([ux, uy])**2 +  np.linalg.norm([vx, vy])**2)
        energy.append(energy_)
        
        uNew = uOld + (1-lamda)*del_t*descretized_laplacian(uOld) - del_t*lamda*data_fidelity_term*Ix
        vNew = vOld + (1-lamda)*del_t*descretized_laplacian(vOld) - del_t*lamda*data_fidelity_term*Iy
        
        if iteration > 10 and energy[-2] - energy[-1] < 1e-4:
            break
        else:
            iteration += 1
            
    mapx = mapx_base + uNew
    mapy = mapy_base + vNew

    deformed = cv2.remap(img, mapx.astype(np.float32), mapy.astype(np.float32), cv2.INTER_LINEAR)

    f, a = plt.subplots(1, 3)
    a[0].imshow(img1, cmap = 'gray')
    a[0].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    a[1].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    a[2].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    a[0].set_title('Source')
    a[1].imshow(img_new1, cmap = 'gray')
    a[1].set_title('Target')
    a[2].imshow(deformed, cmap = 'gray')    
    a[2].set_title('Deformed')
    f.savefig('exp3_lamda_{}.pdf'.format(lamda), dpi = 1000, bbox_inches = 'tight')
    
    msee = np.sqrt(np.linalg.norm(img_target - deformed)**2)/(img.shape[0]*img.shape[1])

    plt.imshow(img_target - deformed, cmap = 'gray')
    plt.title('Difference in deformed and target image\nRMSE:{}'.format(np.round(msee,6)))
    plt.savefig('exp3_mse_lamda_{}.pdf'.format(lamda), dpi = 1000, bbox_inches = 'tight')

    plt.plot(energy)
    plt.title('Minimization of Energy functional with time')
    plt.ylabel('E(u,v)')
    plt.xlabel('Iterations')
    plt.savefig('exp3_energy_lamda_{}.pdf'.format(lamda), dpi = 1000, bbox_inches = 'tight')


    plt.plot(np.sqrt(rmse)) 
    plt.title('Minimization of RMSE with time')
    plt.ylabel('RMSE')
    plt.xlabel('Iterations')
    plt.savefig('exp22_rmse_lamda_{}.pdf'.format(lamda), dpi = 1000, bbox_inches = 'tight')

    #%%

    f, a = plt.subplots(1,4, figsize = (10,5))
    a[0].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    a[1].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    a[2].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    a[3].tick_params(left = False, right = False , labelleft = False ,
                    labelbottom = False, bottom = False)
    a[0].imshow(mapx, cmap = 'gray')
    a[0].set_title('Ground Truth u(x,y)')
    a[1].imshow(mapy, cmap = 'gray')
    a[1].set_title('Ground Truth v(x,y)')

    a[2].imshow(uNew, cmap = 'gray')
    a[2].set_title('u(x,y) at convergence')
    a[3].imshow(vNew, cmap = 'gray')
    a[3].set_title('v(x,y) at convergence')
    f.savefig('u_compare_good.pdf', dpi = 1000, bbox_inches = 'tight')
        