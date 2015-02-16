#!/usr/bin/env python
# Author: Alex Ratner <ajratner@stanford.edu>
# Created: 2015-02-15
# Functions for getting sparse matrix of features from DeepDive output

import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
import psycopg2


# simple dictionary class for indexing
class Index:
  
  def __init__(self):
    self.table = {}
    
  def get_index(self, x):
    if self.table.has_key(x):
      return self.table[x]
    else:
      k = len(self.table)
      self.table[x] = k
      return k

  def __len__(self):
    return len(self.table)
    

# connect to the db
def get_db_conn():
  return psycopg2.connect("user=senwu host=raiders4 port=6432 dbname=genomics_recall")


# load all the features into a sparse matrix
SQL_TEMPLATE_1 = "SELECT * FROM %s_features"
def load_all(entity):
  
  # Get data & index by (e, feature)
  e_idx = Index()
  f_idx = Index()
  data_rows = []
  data_cols = []
  with get_db_conn() as conn:
    with conn.cursor() as cur:
      sql = SQL_TEMPLATE_1 % (entity,)
      cur.execute(sql)
      for row in cur:
        data_rows.append(e_idx.get_index(row[1]))
        data_cols.append(f_idx.get_index(row[2]))
      
  # Load into sparse matrix- COO format for conversion to CSC / CSR
  F = sparse.coo_matrix(([1]*len(data_rows), (data_rows, data_cols)))
  return F


# perform HAC on the feature matrix -> return the linkage matrix
def hac_linkage(F, metric='cosine', method='single'):
  
  # compute the distance matrix
  Y = pdist(F, metric=metric)

  # compute the linkage matrix
  Z = linkage(Y, method=method, metric=metric)
  return Z


# given a linkage matrix and threshold, return a set of flat clusters
def hac_cluster(Z, thresh):
  T = fcluster(Z, thresh, criterion='distance')
  return T
