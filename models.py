import numpy as np


class Models:

    def __init__(self):
        pass

    @staticmethod
    def Gauss1linearbackground(x,a,mu,sigma,m,c):
        p1 = (x-mu)**2/(2*sigma**2)
        return a*np.exp(-p1)+m*x+c
    
    @staticmethod
    def Gauss1linearbackground1(x,params):
        a,mu,sigma,m,c = params
        p1 = (x-mu)**2/(2*sigma**2)
        return a*np.exp(-p1)+m*x+c
    
    @staticmethod
    def Gauss1nobackground(x,a,mu,sigma):
        p1 = (x-mu)**2/(2*sigma**2)
        return a*np.exp(-p1)
    
    @staticmethod
    def Gauss1nobackground1(x,params):
        a,mu,sigma = params
        p1 = (x-mu)**2/(2*sigma**2)
        return a*np.exp(-p1)
    
    @staticmethod
    def Gauss2linearbackground(x,a1,a2,mu1,mu2,sigma1,sigma2,m,c):
        p1 = (x-mu1)**2/(2*sigma1**2)
        p2 = (x-mu2)**2/(2*sigma2**2)

        return a1*np.exp(-p1)+a2*np.exp(-p2)+m*x+c
    
    @staticmethod
    def Gauss2linearbackground1(x,params):
        a1,a2,mu1,mu2,sigma1,sigma2,m,c = params
        p1 = (x-mu1)**2/(2*sigma1**2)
        p2 = (x-mu2)**2/(2*sigma2**2)

        return a1*np.exp(-p1)+a2*np.exp(-p2)+m*x+c
    
    @staticmethod
    def Gauss1ConstBackground(x,a,mu,sigma,c):
        p1 = (x-mu)**2/(2*sigma**2)
        return a*np.exp(-p1)+c
    
    @staticmethod
    def Gauss1ConstBackground1(x,params):
        a,mu,sigma,c = params
        p1 = (x-mu)**2/(2*sigma**2)
        return a*np.exp(-p1)+c


    @staticmethod
    def Gauss1ZeroBackground(x,a,mu,sigma):
        p1 = (x-mu)**2/(2*sigma**2)
        return a*np.exp(-p1)
    
    def Gaussian7linearbackground(x,a1,a2,a3,a4,a5,a6,a7,mu1,mu2,mu3,mu4,mu5,mu6,mu7,sigma1,sigma2,sigma3,sigma4,sigma5,sigma6,sigma7,m,c):
        """
        params = a1,a2,a3,a4,a5,a6,a7,mu1,mu2,mu3,mu4,mu5,mu6,mu7,sigma1,sigma2,sigma3,sigma4,sigma5,sigma6,sigma7,m,c
        """
        # a1,a2,a3,a4,a5,a6,a7,mu1,mu2,mu3,mu4,mu5,mu6,mu7,sigma1,sigma2,sigma3,sigma4,sigma5,sigma6,sigma7,m,c = params
        p1 =  (x-mu1)**2/(2*sigma1**2)
        p2 =  (x-mu2)**2/(2*sigma2**2)
        p3 =  (x-mu3)**2/(2*sigma3**2)
        p4 =  (x-mu4)**2/(2*sigma4**2)
        p5 =  (x-mu5)**2/(2*sigma5**2)
        p6 =  (x-mu6)**2/(2*sigma6**2)
        p7 =  (x-mu7)**2/(2*sigma7**2)
        
        g1 = a1*np.exp(-p1)
        g2 = a2*np.exp(-p2)
        g3 = a3*np.exp(-p3)
        g4 = a4*np.exp(-p4)
        g5 = a5*np.exp(-p5)
        g6 = a6*np.exp(-p6)
        g7 = a7*np.exp(-p7)
        
        return g1+g2+g3+g4+g5+g6+g7+m*x+c
    

    def Gaussian7linearbackground1(x,params):
        """
        params = a1,a2,a3,a4,a5,a6,a7,mu1,mu2,mu3,mu4,mu5,mu6,mu7,sigma1,sigma2,sigma3,sigma4,sigma5,sigma6,sigma7,m,c
        """
        a1,a2,a3,a4,a5,a6,a7,mu1,mu2,mu3,mu4,mu5,mu6,mu7,sigma1,sigma2,sigma3,sigma4,sigma5,sigma6,sigma7,m,c = params
        p1 =  (x-mu1)**2/(2*sigma1**2)
        p2 =  (x-mu2)**2/(2*sigma2**2)
        p3 =  (x-mu3)**2/(2*sigma3**2)
        p4 =  (x-mu4)**2/(2*sigma4**2)
        p5 =  (x-mu5)**2/(2*sigma5**2)
        p6 =  (x-mu6)**2/(2*sigma6**2)
        p7 =  (x-mu7)**2/(2*sigma7**2)
        
        g1 = a1*np.exp(-p1)
        g2 = a2*np.exp(-p2)
        g3 = a3*np.exp(-p3)
        g4 = a4*np.exp(-p4)
        g5 = a5*np.exp(-p5)
        g6 = a6*np.exp(-p6)
        g7 = a7*np.exp(-p7)
        
        return g1+g2+g3+g4+g5+g6+g7+m*x+c

    def Gaussian7constbackground(x,a1,a2,a3,a4,a5,a6,a7,mu1,mu2,mu3,mu4,mu5,mu6,mu7,sigma1,sigma2,sigma3,sigma4,sigma5,sigma6,sigma7,c):
        p1 =  (x-mu1)**2/(2*sigma1**2)
        p2 =  (x-mu2)**2/(2*sigma2**2)
        p3 =  (x-mu3)**2/(2*sigma3**2)
        p4 =  (x-mu4)**2/(2*sigma4**2)
        p5 =  (x-mu5)**2/(2*sigma5**2)
        p6 =  (x-mu6)**2/(2*sigma6**2)
        p7 =  (x-mu7)**2/(2*sigma7**2)
        
        g1 = a1*np.exp(-p1)
        g2 = a2*np.exp(-p2)
        g3 = a3*np.exp(-p3)
        g4 = a4*np.exp(-p4)
        g5 = a5*np.exp(-p5)
        g6 = a6*np.exp(-p6)
        g7 = a7*np.exp(-p7)
        
        return g1+g2+g3+g4+g5+g6+g7+c
    
    
    def Gaussian7constbackground1(x,params):
        a1,a2,a3,a4,a5,a6,a7,mu1,mu2,mu3,mu4,mu5,mu6,mu7,sigma1,sigma2,sigma3,sigma4,sigma5,sigma6,sigma7,c = params
        p1 =  (x-mu1)**2/(2*sigma1**2)
        p2 =  (x-mu2)**2/(2*sigma2**2)
        p3 =  (x-mu3)**2/(2*sigma3**2)
        p4 =  (x-mu4)**2/(2*sigma4**2)
        p5 =  (x-mu5)**2/(2*sigma5**2)
        p6 =  (x-mu6)**2/(2*sigma6**2)
        p7 =  (x-mu7)**2/(2*sigma7**2)
        
        g1 = a1*np.exp(-p1)
        g2 = a2*np.exp(-p2)
        g3 = a3*np.exp(-p3)
        g4 = a4*np.exp(-p4)
        g5 = a5*np.exp(-p5)
        g6 = a6*np.exp(-p6)
        g7 = a7*np.exp(-p7)
        
        return g1+g2+g3+g4+g5+g6+g7+c


    def Gaussian7nobackground(x,a1,a2,a3,a4,a5,a6,a7,mu1,mu2,mu3,mu4,mu5,mu6,mu7,sigma1,sigma2,sigma3,sigma4,sigma5,sigma6,sigma7):
        p1 =  (x-mu1)**2/(2*sigma1**2)
        p2 =  (x-mu2)**2/(2*sigma2**2)
        p3 =  (x-mu3)**2/(2*sigma3**2)
        p4 =  (x-mu4)**2/(2*sigma4**2)
        p5 =  (x-mu5)**2/(2*sigma5**2)
        p6 =  (x-mu6)**2/(2*sigma6**2)
        p7 =  (x-mu7)**2/(2*sigma7**2)
        
        g1 = a1*np.exp(-p1)
        g2 = a2*np.exp(-p2)
        g3 = a3*np.exp(-p3)
        g4 = a4*np.exp(-p4)
        g5 = a5*np.exp(-p5)
        g6 = a6*np.exp(-p6)
        g7 = a7*np.exp(-p7)
        
        return g1+g2+g3+g4+g5+g6+g7
    
    
    def Gaussian7nobackground1(x,params):
        a1,a2,a3,a4,a5,a6,a7,mu1,mu2,mu3,mu4,mu5,mu6,mu7,sigma1,sigma2,sigma3,sigma4,sigma5,sigma6,sigma7 = params
        p1 =  (x-mu1)**2/(2*sigma1**2)
        p2 =  (x-mu2)**2/(2*sigma2**2)
        p3 =  (x-mu3)**2/(2*sigma3**2)
        p4 =  (x-mu4)**2/(2*sigma4**2)
        p5 =  (x-mu5)**2/(2*sigma5**2)
        p6 =  (x-mu6)**2/(2*sigma6**2)
        p7 =  (x-mu7)**2/(2*sigma7**2)
        
        g1 = a1*np.exp(-p1)
        g2 = a2*np.exp(-p2)
        g3 = a3*np.exp(-p3)
        g4 = a4*np.exp(-p4)
        g5 = a5*np.exp(-p5)
        g6 = a6*np.exp(-p6)
        g7 = a7*np.exp(-p7)
        
        return g1+g2+g3+g4+g5+g6+g7

    @staticmethod
    def Linear(x,m,c):
        return m*x+c
