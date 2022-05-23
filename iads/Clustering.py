import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import graphviz as gv
import sys 
import math

import scipy.cluster.hierarchy

def normalisation(data_df):
  nbLig,nbCol = data_df.shape
  new_data_df = pd.DataFrame({})
  
  for c in data_df.columns:
    minCol,maxCol = data_df[c].min(), data_df[c].max()
    diff = maxCol - minCol
    
    col = np.zeros(nbLig)
    for l,ind in zip(range(0,nbLig),data_df.index):
      col[l] = (data_df[c][ind] - minCol) / diff

    new_data_df[c] = col
  return new_data_df

def dist_euclidienne(x1,x2):
  return np.sqrt(np.sum((x1-x2) ** 2))

def dist_manhattan(x1,x2):
  return np.sum(np.abs((x1-x2)))

def dist_vect(dist_t,x1,x2):
  if dist_t == "euclidienne":
    return dist_euclidienne(x1,x2)
  if dist_t == "manhattan":
    return dist_manhattan(x1,x2)
  
  print("Type de distance non reconnu")
  return None

def centroid_linkage(dist_t,data_1,data_2):
  return dist_vect(dist_t,centroide(data_1),centroide(data_2))

def complete_linkage(dist_t,data_1,data_2):
  data_1,data_2 = np.asarray(data_1),np.asarray(data_2)
  d_max = 0
  
  for ind_1 in range(0,len(data_1)):
    for ind_2 in range(0,len(data_2)):
      d = dist_vect(dist_t,data_1[ind_1],data_2[ind_2])
      if d>d_max:
        d_max = d
  
  return d_max

def simple_linkage(dist_t,data_1,data_2):
  dist_min = sys.float_info.max
  data_1,data_2 = np.asarray(data_1),np.asarray(data_2)

  for ind_1 in range(0,len(data_1)):
    for ind_2 in range(0,len(data_2)):
      d = dist_vect(dist_t,data_1[ind_1],data_2[ind_2])
      if d<dist_min:
        dist_min = d
  
  return dist_min

def average_linkage(dist_t,data_1,data_2):
  data_1,data_2 = np.asarray(data_1),np.asarray(data_2)
  sum_d = 0
  
  for ind_1 in range(0,len(data_1)):
    for ind_2 in range(0,len(data_2)):
      sum_d += dist_vect(dist_t,data_1[ind_1],data_2[ind_2])
  
  n = len(data_1)*len(data_2)
  if n > 0:
    return sum_d/n
  else:
    return 0

def centroide(data):
  return data.mean(axis=0)

def dist_centroides(data_1,data_2,methode="centroid linkage"):
  if methode == "centroid linkage":
    return centroid_linkage("euclidienne",data_1,data_2)
  
  if methode == "complete linkage":
    return complete_linkage("euclidienne",data_1,data_2)
  
  if methode == "simple linkage":
    return simple_linkage("euclidienne",data_1,data_2)
  
  if methode == "average linkage":
    return average_linkage("euclidienne",data_1,data_2)
  
  print("Methode non reconnue (reconnues 'centroide linkage' ou 'complete linkage' ou 'simple linkage' ou 'average linkage')")

def initialise(df):
  clust = {}
  for i in range(0,len(df)):
    clust[i] = [i]
  
  return clust

def fusionne(df,P0,methode="centroid linkage",verbose=False,centres=False):
  dist_min = sys.float_info.max
  ind_min_1, ind_min_2 = -1,-1
  
  for ind_1 in P0:
    for ind_2 in P0:
      if ind_1 != ind_2:
        if centres != False:
          d = dist_vect(centres[ind_1],centres[ind_2])
        else:
          d = dist_centroides(df.iloc[P0[ind_1]],df.iloc[P0[ind_2]],\
          methode=methode)
        if d < dist_min:
          dist_min = d
          ind_min_1, ind_min_2 = ind_1,ind_2
  
  ind_max = -1
  P1 = {}
  for ind in P0:
    if ind>ind_max:
      ind_max = ind
    
    if ind!=ind_min_1 and ind!=ind_min_2:
      P1[ind] = P0[ind]
  P1[ind_max+1] = P0[ind_min_1]+P0[ind_min_2]
  
  return P1,ind_min_1,ind_min_2,dist_min

def clustering_hierarchique(df,methode="centroid linkage",verbose=False,dendrogramme=False):
  P0 = initialise(df)
  l = []
  centres = False
  
  while(len(P0)>1):
    # pour eviter le recalcul de centre pour chaque ensemble
    if methode == "centroid linkage":
      centres = {}
      for c in P0:
        centres[c] = centroide(df.iloc[P0[c]])

    P1,ind_1,ind_2,d = fusionne(df,P0,methode=methode,centres=centres)
    l.append([ind_1,ind_2,d, len(P0[ind_1])+len(P0[ind_2])])
    P0 = P1
    if verbose:
      print("Distance minimale trouvée entre [",ind_1,",",ind_2,"] =",d)
  
  if dendrogramme:
    # Paramètre de la fenêtre d'affichage: 
    plt.figure(figsize=(30, 15)) # taille : largeur x hauteur
    plt.title('Dendrogramme', fontsize=25)    
    plt.xlabel("Indice d'exemple", fontsize=25)
    plt.ylabel('Distance', fontsize=25)

    # Construction du dendrogramme pour notre clustering :
    scipy.cluster.hierarchy.dendrogram(
        clustering_hierarchique(df), 
        leaf_font_size=24.,  # taille des caractères de l'axe des X
    )

    # Affichage du résultat obtenu:
    plt.show()
  return l

#------------------------------------------------------

def dist_vect(v1, v2):
  return  dist_euclidienne(v1, v2)
############# A COMPLETER 

def inertie_cluster(Ens):
  Ens_ar = np.asarray(Ens)
  g = centroide(Ens_ar)
  inert = 0
  for i in range(0,Ens_ar.shape[0]):
    inert += dist_vect(Ens_ar[i],g) ** 2
  
  return inert
############# A COMPLETER 

def init_kmeans(K,Ens):
  Ens_ar = np.asarray(Ens)
  ind = np.random.choice(len(Ens_ar),size=K,replace=False)

  return Ens_ar[ind]
############# A COMPLETER 

def plus_proche(Exe,Centres):
  ind_pp, d_pp = 0, dist_vect(Exe,Centres[0])
  for i in range(1,len(Centres)):
    d = dist_vect(Exe,Centres[i])
    if d<d_pp:
      ind_pp = i
      d_pp = d
      
  return ind_pp

def affecte_cluster(Base,Centres):
  Base_ar = np.asarray(Base)
  U = {}
  for i in range(0,len(Centres)):
    U[i] = []
  
  for i in range(0,len(Base_ar)):
    i_cent = plus_proche(Base_ar[i],Centres)
    U[i_cent].append(i)
  
  return U
############# A COMPLETER 

def nouveaux_centroides(Base,U):
  Base_ar = np.asarray(Base)
  Centres = []
  for i in U:
    Centre = centroide(Base_ar[U[i]])
    Centres.append(Centre)
  
  return np.asarray(Centres)

def inertie_globale(Base, U):
  Base_ar = np.asarray(Base)
  # G = clust.centroide(Base_ar)
  
  # inertie_inter = 0
  inertie_intra = 0
  for i in range(0,len(U)):
    Ens = Base_ar[U[i]]
    
    # Gk = clust.centroide(Ens)
    # inertie_inter += len(Ens)*(dist_vect(Gk,G) ** 2)
    
    inertie_intra += inertie_cluster(Ens)
  
  return inertie_intra
  # return inertie_inter+inertie_intra
  
def kmoyennes(K, Base, epsilon, iter_max,verbose=False):
  Centres_0 = init_kmeans(K,Base)
  U_0 = affecte_cluster(Base,Centres_0)
  inert_0 = inertie_globale(Base,U_0)
  
  Centres_1,U_1 = Centres_0,U_0 # si jamais on ne rentre pas dans la boucle
  for it in range(1,iter_max+1):
    Centres_1 = nouveaux_centroides(Base,U_0)
    U_1 = affecte_cluster(Base,Centres_1)
    inert_1 = inertie_globale(Base,U_1)
    
    d = inert_0-inert_1
    if verbose == True:
      print(f'Iteration {it} Inertie : {inert_1:1.4f} Difference : {d:1.4f}')
    
    if d<epsilon :
      break
    
    Centres_0,U_0,inert_0 = Centres_1, U_1, inert_1
  
  return Centres_1, U_1

def diametre(Ens):
  d_max = 0
  
  for i in range(0,len(Ens)):
    for j in range(0,len(Ens)):
      d = dist_vect(Ens[i],Ens[j])
      if d_max<d:
        d_max = d
  
  return d_max

def dist_min(Ens):
  d_min = dist_vect(Ens[0],Ens[1])
  
  for i in range(0,len(Ens)):
    for j in range(0,len(Ens)):
      if i != j:
        d = dist_vect(Ens[i],Ens[j])
        if d<d_min:
          d_min = d
  
  return d_min

def index_Dunn(Base,Centres,U,verbose=False):
  Base_ar = np.asarray(Base)
  Dk = []
  for i in range(0,len(U)):
    Ens = Base_ar[U[i]]
    Dk.append(diametre(Ens))
  
  if verbose:
    print("Centres :\n",Centres)
    print("Diamètre :",Dk)
  
  return sum(Dk) / inertie_globale(Base,U)

def index_Xie_Beni(Base,Centres,U):
  return dist_min(Centres) / inertie_globale(Base,U)

