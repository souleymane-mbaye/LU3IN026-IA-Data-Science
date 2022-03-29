# -*- coding: utf-8 -*-

"""
Package: iads
File: Classifiers.py
Année: LU3IN026 - semestre 2 - 2021-2022, Sorbonne Université
"""

# Classfieurs implémentés en LU3IN026
# Version de départ : Février 2022

# Import de packages externes
import numpy as np
import pandas as pd
import copy
import graphviz as gv
import sys 
import math

# ---------------------------
# ------------------------ A COMPLETER :
class Classifier:
  """ Classe (abstraite) pour représenter un classifieur
      Attention: cette classe est ne doit pas être instanciée.
  """
  
  def __init__(self, input_dimension):
    """ Constructeur de Classifier
      Argument:
          - intput_dimension (int) : dimension de la description des exemples
      Hypothèse : input_dimension > 0
    """
    raise NotImplementedError("Please Implement this method")
      
  def train(self, desc_set, label_set):
      """ Permet d'entrainer le modele sur l'ensemble donné
          desc_set: ndarray avec des descriptions
          label_set: ndarray avec les labels correspondants
          Hypothèse: desc_set et label_set ont le même nombre de lignes
      """        
      raise NotImplementedError("Please Implement this method")
  
  def score(self,x):
      """ rend le score de prédiction sur x (valeur réelle)
          x: une description
      """
      raise NotImplementedError("Please Implement this method")
  
  def predict(self, x):
      """ rend la prediction sur x (soit -1 ou soit +1)
          x: une description
      """
      raise NotImplementedError("Please Implement this method")

  def accuracy(self, desc_set, label_set):
    """ Permet de calculer la qualité du système sur un dataset donné
        desc_set: ndarray avec des descriptions
        label_set: ndarray avec les labels correspondants
        Hypothèse: desc_set et label_set ont le même nombre de lignes
    """
    # ------------------------------
    # COMPLETER CETTE FONCTION ICI : 
    yhat = np.array([self.predict(x) for x in desc_set])
    return np.where(label_set == yhat, 1., 0.).mean()
# ---------------------------

# ------------------------ A COMPLETER :
class ClassifierKNN(Classifier):
    """ Classe pour représenter un classifieur par K plus proches voisins.
        Cette classe hérite de la classe Classifier
    """

    # ATTENTION : il faut compléter cette classe avant de l'utiliser !
    
    def __init__(self, input_dimension, k):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension d'entrée des exemples
                - k (int) : nombre de voisins à considérer
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.k = k
        self.desc_set = None
        self.label_set = None
        
    def score(self,x):
        """ rend la proportion de +1 parmi les k ppv de x (valeur réelle)
            x: une description : un ndarray
        """
        taille = len(self.desc_set)
        tab_dist_x = np.zeros(taille)
        for i in range(taille) :
            tab_dist_x[i] = np.dot(x-self.desc_set[i], x-self.desc_set[i])
        voisins_ord = np.argsort(tab_dist_x)
        k_1 = 0
        for i in range(self.k):
            if self.label_set[voisins_ord[i]] == 1 :
                k_1 += 1
        p = float(k_1) / float(self.k)
        return 2*(p - 0.5)
    def predict(self, x):
        """ rend la prediction sur x (-1 ou +1)
            x: une description : un ndarray
        """
        return 1 if (self.score(x) >= 0) else -1
        

    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        self.desc_set = desc_set
        self.label_set = label_set
# ---------------------------

# ------------------------ A COMPLETER :
class ClassifierLineaireRandom(Classifier):
    """ Classe pour représenter un classifieur linéaire aléatoire
        Cette classe hérite de la classe Classifier
    """
    
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        # self.desc_set = None
        # self.label_set = None
        v = np.random.uniform(-1,1,input_dimension)
        # v = np.array([np.random.randn() for i in range(input_dimension)])
        n = np.linalg.norm(v)
        nor = np.array([n for i in range(self.dimension)])
        self.w = v / nor
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        # self.desc_set = desc_set
        # self.label_set = label_set
        print("Pas d'apprentissage pour ce classifieur")
    def getW(self):
        return self.w
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if( self.score(x)>= 0) else -1
# ---------------------------

# ------------------------ A COMPLETER : DEFINITION DU CLASSIFIEUR PERCEPTRON
class ClassifierPerceptron(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.dimension = input_dimension
        self.learning_rate = learning_rate

        self.w = np.zeros(self.dimension)
        if init != 0:
          petit_reel = 0.001
          self.w = (2*(np.random.rand(self.dimension)) - 1) * petit_reel
    
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        taille = len(desc_set)  # nombre de variables Xi contenues dans X(desc_set)
        ind_val_alea = np.array([i for i in range(taille)])
        np.random.shuffle(ind_val_alea) # ordre des Xi aleatoire
        
        for i in ind_val_alea:
            sc = self.score(desc_set[i])    # prediction avec w
            # Si bien prédit alors le score et Yi sont de même signe
            if sc*label_set[i] <= 0:
                epYi = np.array([self.learning_rate*label_set[i]]*self.dimension)
                self.w += epYi * desc_set[i]
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        
        l_diff = []
        for i in range(niter_max):
            w_pred = np.copy(self.w)
            self.train_step(desc_set,label_set)
            w_pred -= self.w
            w_pred = np.sqrt(w_pred ** 2)
            
            diff = np.sum(w_pred) - seuil
            l_diff.append(diff)
            if diff <= 0:
                return l_diff
        
        # print("Nombre d'itérations max atteint !")
        return l_diff
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) >= 0 else -1
# ---------------------------

# CLasse (abstraite) pour représenter des noyaux
class Kernel():
    """ Classe pour représenter des fonctions noyau
    """
    def __init__(self, dim_in, dim_out):
        """ Constructeur de Kernel
            Argument:
                - dim_in : dimension de l'espace de départ (entrée du noyau)
                - dim_out: dimension de l'espace de d'arrivée (sortie du noyau)
        """
        self.input_dim = dim_in
        self.output_dim = dim_out
        
    def get_input_dim(self):
        """ rend la dimension de l'espace de départ
        """
        return self.input_dim

    def get_output_dim(self):
        """ rend la dimension de l'espace d'arrivée
        """
        return self.output_dim
    
    def transform(self, V):
        """ ndarray -> ndarray
            fonction pour transformer V dans le nouvel espace de représentation
        """        
        raise NotImplementedError("Please Implement this method")
# ---------------------------
class KernelBias(Kernel):
    """ Classe pour un noyau simple 2D -> 3D
    """
    def transform(self, V):
        """ ndarray de dim 2 -> ndarray de dim 3            
            rajoute une 3e dimension au vecteur donné
        """
        V_proj = np.append(V,np.ones((len(V),1)),axis=1)
        return V_proj
# ------------------------ A COMPLETER :
class ClassifierPerceptronKernel(Classifier):
    """ Perceptron de Rosenblatt kernelisé
    """
    def __init__(self, input_dimension, learning_rate, noyau, init=0):
        """int x float x Kernel x int -> ClassifierPerceptronKernel
            Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (espace originel)
                - learning_rate : epsilon
                - noyau : Kernel à utiliser
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.dimension = input_dimension
        self.learning_rate = learning_rate
        self.kernel = noyau

        self.w = np.zeros(self.kernel.get_output_dim())
        if init != 0:
          petit_reel = 0.001
          self.w = (2*(np.random.rand(self.kernel.get_output_dim())) - 1) * petit_reel
        
    def train_step(self, desc_set, label_set):
        """ 
            Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        if np.shape(desc_set)[1] == self.kernel.get_input_dim():
            desc_set = self.kernel.transform(desc_set)  # Kernalisation
        taille,d = np.shape(desc_set)  # nombre de variables Xi contenues dans X(desc_set)
        ind_val_alea = np.array([i for i in range(taille)])
        np.random.shuffle(ind_val_alea) # ordre des Xi aleatoire
        
        for i in ind_val_alea:
            sc = self.score(desc_set[i])    # prediction avec w
            # Si bien prédit alors le score et Yi sont de même signe
            if sc*label_set[i] <= 0:
                epYi = np.array([self.learning_rate*label_set[i]]*d)
                self.w += epYi * desc_set[i]
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments: (dans l'espace originel)
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.01) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        
        if np.shape(desc_set)[1] == self.kernel.get_input_dim():
            desc_set = self.kernel.transform(desc_set)  # Kernalisation
        l_diff = []
        for i in range(niter_max):
            w_pred = np.copy(self.w)
            self.train_step(desc_set,label_set)
            w_pred -= self.w
            w_pred = np.sqrt(w_pred ** 2)
            
            diff = np.sum(w_pred) - seuil
            l_diff.append(diff)
            if diff <= 0:
                return l_diff
        
        # print("Nombre d'itérations max atteint !")
        return l_diff
    
    def score(self,x):
        """ rend le score de prédiction sur x 
            x: une description (dans l'espace originel)
        """
        if np.shape(x)[0] == self.kernel.get_input_dim():
            x = self.kernel.transform(x.reshape(1,self.kernel.get_input_dim()))  # Kernalisation

        return np.dot(x,self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description (dans l'espace originel)
        """
        return 1 if self.score(x) >= 0 else -1
# ---------------------------

# ---------------------------
class ClassifierPerceptronBiais(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, init=0):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples (>0)
                - learning_rate : epsilon
                - init est le mode d'initialisation de w: 
                    - si 0 (par défaut): initialisation à 0 de w,
                    - si 1 : initialisation par tirage aléatoire de valeurs petites
        """
        self.dimension = input_dimension
        self.learning_rate = learning_rate

        self.w = np.zeros(self.dimension)
        if init == 1:
          petit_reel = 0.001
          for i in range(self.dimension):
            v = (2*(np.random.rand()) - 1) * petit_reel
            self.w[i] = v
        self.allw = [self.w.copy()]
    
    def get_allw(self):
        return self.allw
    
    def train_step(self, desc_set, label_set):
        """ Réalise une unique itération sur tous les exemples du dataset
            donné en prenant les exemples aléatoirement.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
        """        
        taille = len(desc_set)  # nombre de variables Xi contenues dans X(desc_set)
        ind_val_alea = np.array([i for i in range(taille)])
        np.random.shuffle(ind_val_alea) # ordre des Xi aleatoire
        
        for i in ind_val_alea:
            fxi = self.score(desc_set[i])    # prediction avec w
            # Si bien prédit alors le score et Yi sont de même signe
            if fxi*label_set[i] <= 1:
                # epYi_fxi = np.array([self.learning_rate*(label_set[i]-fxi)]*self.dimension)
                self.w += self.learning_rate*label_set[i] * desc_set[i]
                self.allw.append(self.w.copy())
     
    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
        """ Apprentissage itératif du perceptron sur le dataset donné.
            Arguments:
                - desc_set: ndarray avec des descriptions
                - label_set: ndarray avec les labels correspondants
                - niter_max (par défaut: 100) : nombre d'itérations maximale
                - seuil (par défaut: 0.001) : seuil de convergence
            Retour: la fonction rend une liste
                - liste des valeurs de norme de différences
        """        
        # self.allw = [self.w.copy()]
        l_diff = []
        for i in range(niter_max):
            w_pred = np.copy(self.w)
            self.train_step(desc_set,label_set)
            w_pred -= self.w
            w_pred = np.sqrt(w_pred ** 2)
            
            diff = np.sum(w_pred) - seuil
            l_diff.append(diff)
            # self.allw.append(self.w.copy())
            if diff <= 0:
                return l_diff
        
        # print("Nombre d'itérations max atteint !")
        return l_diff
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) >= 0 else -1

# ---------------------------
class Perceptron_MC(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self, input_dimension, learning_rate, nb_classes=2, init=0):
      """ Constructeur de Classifier
        Argument:
          - input_dimension (int) : dimension de la description des exemples (>0)
          - learning_rate : epsilon
          - init est le mode d'initialisation de w: 
            - si 0 (par défaut): initialisation à 0 de w,
            - si 1 : initialisation par tirage aléatoire de valeurs petites
      """
      self.input_dimension = input_dimension
      self.learning_rate = learning_rate
      self.nb_classes = nb_classes
      self.init = init
      self.classes = []
    
    def get_l_allw(self):
      return np.array([c.get_allw() for c in self.classes])    
    
    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
      """ Apprentissage itératif du perceptron sur le dataset donné.
        Arguments:
          - desc_set: ndarray avec des descriptions
          - label_set: ndarray avec les labels correspondants
          - niter_max (par défaut: 100) : nombre d'itérations maximale
          - seuil (par défaut: 0.001) : seuil de convergence
        Retour: la fonction rend une liste
          - liste des valeurs de norme de différences
      """        
      self.classes = []
      l_diff = []
      for i in range(0,self.nb_classes):
        c = ClassifierPerceptronBiais(self.input_dimension,self.learning_rate,self.init)
        Ytmp = np.where(label_set==i, 1, -1)
        d = c.train(desc_set,Ytmp,niter_max, seuil)
        self.classes.append(c)
        l_diff.append(d)
      
      # print("Nombre d'itérations max atteint !")
      return l_diff
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.array([c.score(x) for c in self.classes])
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.argmax(self.score(x))

class ClassifierMultiOAA(Classifier):
    """ Perceptron de Rosenblatt
    """
    def __init__(self,classif_bin):
      """ Constructeur de Classifier
        Argument:
          - input_dimension (int) : dimension de la description des exemples (>0)
          - learning_rate : epsilon
          - init est le mode d'initialisation de w: 
            - si 0 (par défaut): initialisation à 0 de w,
            - si 1 : initialisation par tirage aléatoire de valeurs petites
      """
      self.classif_bin = classif_bin
      self.classes = []
    
    def get_l_allw(self):
      return np.array([c.get_allw() for c in self.classes])    
    
    def train(self, desc_set, label_set, niter_max=100, seuil=0.001):
      """ Apprentissage itératif du perceptron sur le dataset donné.
        Arguments:
          - desc_set: ndarray avec des descriptions
          - label_set: ndarray avec les labels correspondants
          - niter_max (par défaut: 100) : nombre d'itérations maximale
          - seuil (par défaut: 0.001) : seuil de convergence
        Retour: la fonction rend une liste
          - liste des valeurs de norme de différences
      """        
      self.classes = []
      l_diff = []
      nb_classes = np.shape(np.unique(label_set))[0]
      for i in range(0,nb_classes):
        c = copy.deepcopy(self.classif_bin)
        Ytmp = np.where(label_set==i, 1, -1)
        d = c.train(desc_set,Ytmp,niter_max, seuil)
        self.classes.append(c)
        l_diff.append(d)
      
      # print("Nombre d'itérations max atteint !")
      return l_diff
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.array([c.score(x) for c in self.classes])
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return np.argmax(self.score(x))

# code de la classe pour le classifieur ADALINE


# ATTENTION: contrairement à la classe ClassifierPerceptron, on n'utilise pas de méthode train_step()
# dans ce classifier, tout se fera dans train()


#TODO: Classe à Compléter

class ClassifierADALINE(Classifier):
    """ Perceptron de ADALINE
    """
    def __init__(self, input_dimension, learning_rate, history=False, niter_max=1000, seuil=0.001):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.learning_rate = learning_rate
        self.history = history
        self.niter_max = niter_max
        self.seuil = seuil
        self.allw=[]

        self.w = np.zeros(self.dimension)
        petit_reel = 0.001
        self.w = (2*(np.random.rand(self.dimension)) - 1) * petit_reel
    
    def get_allw(self):
        return self.allw
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        taille = len(desc_set)  # nombre de variables Xi contenues dans X(desc_set)
        ind_val_alea = np.array([i for i in range(taille)])
        np.random.shuffle(ind_val_alea) # ordre des Xi aleatoire
            
        l_diff = []
        self.allw = [self.w.copy()]
        for i in range(0,self.niter_max):
            w_pred = np.copy(self.w)
            
            # Tirage d'un i 
            # i = np.random.randint(0,len(desc_set))
            # Mise à jour pour une epoch
            for i in ind_val_alea:
                d = np.dot(desc_set[i],self.w) 
                d -= label_set[i]
                delta_C = desc_set[i].T * d
                self.w -= self.learning_rate * delta_C
            
                if self.history:
                    self.allw.append(self.w.copy())
            
            
            w_pred -= self.w
            w_pred = np.sqrt(w_pred ** 2)
            
            diff = np.sum(w_pred) - self.seuil
            l_diff.append(diff)
            if diff <= 0:
                return l_diff
            
        return l_diff
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) >= 0 else -1
# ---------------------------

class ClassifierADALINE2(Classifier):
    def __init__(self, input_dimension):
        """ Constructeur de Classifier
            Argument:
                - input_dimension (int) : dimension de la description des exemples
                - learning_rate : epsilon
                - history : stockage des poids w en cours d'apprentissage
                - niter_max : borne sur les iterations
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension

        self.w = np.zeros(self.dimension)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            réalise une itération sur l'ensemble des données prises aléatoirement
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        X, Y = desc_set, label_set
        XT = X.T
        self.w = np.linalg.solve(XT@X, XT@Y)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        return np.dot(x, self.w)
    
    def predict(self, x):
        """ rend la prediction sur x (soit -1 ou soit +1)
            x: une description
        """
        return 1 if self.score(x) >= 0 else -1
# ---------------------------

def classe_majoritaire(Y):
    """ Y : (array) : array de labels
        rend la classe majoritaire ()
    """
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    return valeurs[np.argmax(nb_fois)]
    #### A compléter pour répondre à la question posée
# ---------------------------

def shannon(P):
    """ list[Number] -> float
        Hypothèse: la somme des nombres de P vaut 1
        P correspond à une distribution de probabilité
        rend la valeur de l'entropie de Shannon correspondante
    """
    k = len(P)
    if k <2 :
        return 0
    tab_p = np.array([(pi*math.log(pi) if pi!=0 else 0) for pi in P])
    
    return -1 * np.sum(tab_p)

def entropie(Y):
    """ Y : (array) : ensemble de labels de classe
        rend l'entropie de l'ensemble Y
    """
    valeurs, nb_fois = np.unique(Y,return_counts=True)
    return shannon(np.array(nb_fois)/len(Y))
# ---------------------------

# La librairie suivante est nécessaire pour l'affichage graphique de l'arbre:
# import graphviz as gv

# Pour plus de détails : https://graphviz.readthedocs.io/en/stable/manual.html

# Eventuellement, il peut être nécessaire d'installer graphviz sur votre compte:
# pip install --user --install-option="--prefix=" -U graphviz

class NoeudCategoriel:
    """ Classe pour représenter des noeuds d'un arbre de décision
    """
    def __init__(self, num_att=-1, nom=''):
        """ Constructeur: il prend en argument
            - num_att (int) : le numéro de l'attribut auquel il se rapporte: de 0 à ...
              si le noeud se rapporte à la classe, le numéro est -1, on n'a pas besoin
              de le préciser
            - nom (str) : une chaîne de caractères donnant le nom de l'attribut si
              il est connu (sinon, on ne met rien et le nom sera donné de façon 
              générique: "att_Numéro")
        """
        self.attribut = num_att    # numéro de l'attribut
        if (nom == ''):            # son nom si connu
            self.nom_attribut = 'att_'+str(num_att)
        else:
            self.nom_attribut = nom 
        self.Les_fils = None       # aucun fils à la création, ils seront ajoutés
        self.classe   = None       # valeur de la classe si c'est une feuille
        
    def est_feuille(self):
        """ rend True si l'arbre est une feuille 
            c'est une feuille s'il n'a aucun fils
        """
        return self.Les_fils == None
    
    def ajoute_fils(self, valeur, Fils):
        """ valeur : valeur de l'attribut de ce noeud qui doit être associée à Fils
                     le type de cette valeur dépend de la base
            Fils (NoeudCategoriel) : un nouveau fils pour ce noeud
            Les fils sont stockés sous la forme d'un dictionnaire:
            Dictionnaire {valeur_attribut : NoeudCategoriel}
        """
        if self.Les_fils == None:
            self.Les_fils = dict()
        self.Les_fils[valeur] = Fils
        # Rem: attention, on ne fait aucun contrôle, la nouvelle association peut
        # écraser une association existante.
    
    def ajoute_feuille(self,classe):
        """ classe: valeur de la classe
            Ce noeud devient un noeud feuille
        """
        self.classe    = classe
        self.Les_fils  = None   # normalement, pas obligatoire ici, c'est pour être sûr
        
    def classifie(self, exemple):
        """ exemple : numpy.array
            rend la classe de l'exemple (pour nous, soit +1, soit -1 en général)
            on rend la valeur 0 si l'exemple ne peut pas être classé (cf. les questions
            posées en fin de ce notebook)
        """
        if self.est_feuille():
            return self.classe
        if exemple[self.attribut] in self.Les_fils:
            # descente récursive dans le noeud associé à la valeur de l'attribut
            # pour cet exemple:
            return self.Les_fils[exemple[self.attribut]].classifie(exemple)
        else:
            # Cas particulier : on ne trouve pas la valeur de l'exemple dans la liste des
            # fils du noeud... Voir la fin de ce notebook pour essayer de résoudre ce mystère...
            print('\t*** Warning: attribut ',self.nom_attribut,' -> Valeur inconnue: ',exemple[self.attribut])
            return 0
    
    def to_graph(self, g, prefixe='A'):
        """ construit une représentation de l'arbre pour pouvoir l'afficher graphiquement
            Cette fonction ne nous intéressera pas plus que ça, elle ne sera donc pas expliquée            
        """
        if self.est_feuille():
            g.node(prefixe,str(self.classe),shape='box')
        else:
            g.node(prefixe, self.nom_attribut)
            i =0
            for (valeur, sous_arbre) in self.Les_fils.items():
                sous_arbre.to_graph(g,prefixe+str(i))
                g.edge(prefixe,prefixe+str(i), valeur)
                i = i+1        
        return g
# ---------------------------
# import sys 

def construit_AD(X,Y,epsilon,LNoms = []):
    """ X,Y : dataset
        epsilon : seuil d'entropie pour le critère d'arrêt 
        LNoms : liste des noms de features (colonnes) de description 
    """
    
    # dimensions de X:
    (nb_lig, nb_col) = X.shape
    
    entropie_classe = entropie(Y)
    
    if (entropie_classe <= epsilon) or  (nb_lig <=1):
        # ARRET : on crée une feuille
        noeud = NoeudCategoriel(-1,"Label")
        noeud.ajoute_feuille(classe_majoritaire(Y))
    else:
        gain_max = sys.float_info.max  # meilleur gain trouvé (initalisé à -infinie)
        i_best = -1         # numéro du meilleur attribut
        Xbest_valeurs = None
        
        #############
        
        # COMPLETER CETTE PARTIE : ELLE DOIT PERMETTRE D'OBTENIR DANS
        # i_best : le numéro de l'attribut qui maximise le gain d'information.  En cas d'égalité,
        #          le premier rencontré est choisi.
        # gain_max : la plus grande valeur de gain d'information trouvée.
        # Xbest_valeurs : la liste des valeurs que peut prendre l'attribut i_best
        #
        # Il est donc nécessaire ici de parcourir tous les attributs et de calculer
        # la valeur du gain d'information pour chaque attribut.
        
        ##################
        ## COMPLETER ICI !
        ##################
        for j in range(0,nb_col):
            Xj = X[:,j]
            vj, n_vj = np.unique(Xj, return_counts=True)
            entropie_vj = 0
            
            for vjl,n_vjl in zip(vj,n_vj):
                Xvjl, Yvjl = Xj[Xj==vjl], Y[Xj==vjl] # pas besoin de Xvjl
                entropie_vj += entropie(Yvjl) * n_vjl/len(Xj)
            
            if(entropie_vj < gain_max):
                gain_max = entropie_vj
                i_best = j
                Xbest_valeurs = vj
            
        #############
        
        if len(LNoms)>0:  # si on a des noms de features
            noeud = NoeudCategoriel(i_best,LNoms[i_best])    
        else:
            noeud = NoeudCategoriel(i_best)
        for v in Xbest_valeurs:
            noeud.ajoute_fils(v,construit_AD(X[X[:,i_best]==v], Y[X[:,i_best]==v],epsilon,LNoms))
    return noeud
# ---------------------------
class ClassifierArbreDecision(Classifier):
    """ Classe pour représenter un classifieur par arbre de décision
    """
    
    def __init__(self, input_dimension, epsilon, LNoms=[]):
        """ Constructeur
            Argument:
                - intput_dimension (int) : dimension de la description des exemples
                - epsilon (float) : paramètre de l'algorithme (cf. explications précédentes)
                - LNoms : Liste des noms de dimensions (si connues)
            Hypothèse : input_dimension > 0
        """
        self.dimension = input_dimension
        self.epsilon = epsilon
        self.LNoms = LNoms
        # l'arbre est manipulé par sa racine qui sera un Noeud
        self.racine = None
        
    def toString(self):
        """  -> str
            rend le nom du classifieur avec ses paramètres
        """
        return 'ClassifierArbreDecision ['+str(self.dimension) + '] eps='+str(self.epsilon)
        
    def train(self, desc_set, label_set):
        """ Permet d'entrainer le modele sur l'ensemble donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """        
        ##################
        ## COMPLETER ICI !
        ##################
        self.racine = construit_AD(desc_set,label_set,self.epsilon,self.LNoms)
    
    def score(self,x):
        """ rend le score de prédiction sur x (valeur réelle)
            x: une description
        """
        # cette méthode ne fait rien dans notre implémentation :
        pass
    
    def predict(self, x):
        """ x (array): une description d'exemple
            rend la prediction sur x             
        """
        ##################
        ## COMPLETER ICI !
        ##################
        return self.racine.classifie(x)
    def accuracy(self, desc_set, label_set):  # Version propre à aux arbres
        """ Permet de calculer la qualité du système sur un dataset donné
            desc_set: ndarray avec des descriptions
            label_set: ndarray avec les labels correspondants
            Hypothèse: desc_set et label_set ont le même nombre de lignes
        """
        nb_ok=0
        for i in range(desc_set.shape[0]):
            if self.predict(desc_set[i,:]) == label_set[i]:
                nb_ok=nb_ok+1
        acc=nb_ok/(desc_set.shape[0] * 1.0)
        return acc

    def affiche(self,GTree):
        """ affichage de l'arbre sous forme graphique
            Cette fonction modifie GTree par effet de bord
        """
        self.racine.to_graph(GTree)
# ---------------------------
