import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2

def L2Fit(image):
    rows, cols = image.shape 

    A = np.zeros([rows*cols, 3])
    b = np.zeros([rows*cols, 1])
    for i in range(rows):
        for j in range(cols):
            A[i*cols + j] = [j,i,1]   ### [x,y,1]
            b[i*cols + j] = image[i,j]
    ATA_Inv = np.linalg.inv(np.matmul(np.transpose(A), A))
    plane_coeff = np.matmul(np.matmul(ATA_Inv , np.transpose(A)), b)
    return plane_coeff, rows, cols

def L1Fit(img, iterations):
    ###### Memory Error due to large matrix size
    rows, cols = img.shape 
    A = np.zeros([rows*cols, 3])
    b = np.zeros([rows*cols, 1])
    for i in range(rows):
        for j in range(cols):
            A[i*cols + j] = [j,i,1]   ### [x,y,1]
            b[i*cols + j] = img[i,j]
    ATA_Inv = np.linalg.inv(np.matmul(np.transpose(A), A))
    plane_coeff = np.matmul(np.matmul(ATA_Inv , np.transpose(A)), b)
    
    x = np.arange(cols)
    y = np.arange(rows)
    xv, yv = np.meshgrid(x,y)

    for i in range(iterations):
        out = img - (plane_coeff[0]*xv + plane_coeff[1]*yv + plane_coeff[2])
        D = np.diag(np.abs(out.flatten()))

        ATA_Inv = np.linalg.inv(np.matmul(np.matmul(np.transpose(A), D), A))
        plane_coeff = np.matmul(np.matmul(ATA_Inv , np.transpose(A)), np.matmul(D,b))
    return plane_coeff, rows, cols

if __name__ == '__main__':
    img = loadmat('unwrapped_phase_matrix_swan_10hz_.mat')['unwrapped_phase_matrix']
    
    #### Memory Error due to large Matrix size
    plane_coeff, rows, cols = L1Fit(img, 15)
    x = np.arange(cols)
    y = np.arange(rows)
    xv, yv = np.meshgrid(x,y)
    out_1 = img - (plane_coeff[0]*xv + plane_coeff[1]*yv + plane_coeff[2])

    '''plane_coeff_2, rows, cols = L2Fit(img)
    x_2 = np.arange(cols)
    y_2 = np.arange(rows)
    xv_2, yv_2 = np.meshgrid(x_2,y_2)
    out_2 = (img - (plane_coeff_2[0]*xv_2 + plane_coeff_2[1]*yv_2 + plane_coeff_2[2])) * 0.56 * 100 / (2 * 0.125 * 10 * np.pi)
     '''


    cv2.imshow('original', cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    cv2.imshow('Demod_L2', cv2.normalize(out_2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F))
    cv2.waitKey(10000)
    cv2.destroyAllWindows()
    print(plane_coeff_2)