import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Read in the data

ratings = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW4/hw4-data/ratings.csv', header = None).values
ratings_test = pd.read_csv('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW4/hw4-data/ratings_test.csv', header = None).values

movies = []
with open('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW4/hw4-data/movies.txt') as inputfile:
    for line in inputfile:
        movies.append(line.strip())
        
N1 = len(np.unique(ratings[:,0]))
N2 = len(movies)
var = 0.5
lmb = 1
d = 10

#%%

def get_UV():
    U = np.empty(shape=(N1, d))
    U[:] = np.nan
    V = np.random.multivariate_normal(np.repeat(0, d), np.identity(d)/lmb, N2).T
    return U, V

#%%
def get_matrix(ratings, N1, N2, train = False, test = False):
    mat = np.empty(shape = (N1, N2))
    mat[:] = np.nan
    for i in range(ratings[:, 0].size):
        mat[int(ratings[i, 0]) - 1, int(ratings[i, 1]) - 1] = ratings[i, 2]
    return mat

M = get_matrix(ratings, N1, N2)
M_test = get_matrix(ratings_test, N1, N2)

#%%
    
def get_objective(U, V, matrix, var = 0.5, lmb = 1):
    preds = U.dot(V)
    obs = ~np.isnan(matrix)
    error = matrix[obs] - preds[obs]
    return - (error.dot(error))/(2*var) - (U**2).sum()*lmb /2 - (V**2).sum()*lmb/2

#%%
    
def get_rmse(U, V, ratings_test):
    pred_ratings=[]
    for i in range(ratings_test.shape[0]):
        pred_ratings.append(np.dot(U[ratings_test[i][0].astype(int)-1,:].reshape(1,10), V[:,ratings_test[i][1].astype(int)-1]))
    pred_ratings=np.asarray(pred_ratings).flatten()
    val=np.dot((ratings_test[:,2]-pred_ratings),(ratings_test[:,2]-pred_ratings))
    rmse = np.sqrt(val/len(pred_ratings))
    return rmse

#%%
    
def coord_ascent(U, V, M, n1, n2, n_iter = 100, var = 0.5, lmb = 1, d = 10):
    objective_list = []
    for t in range(n_iter):
            
        for i in range(N1):
            obs_indexes = ~np.isnan(M[i,:])
            # print(sum(obs_indexes))
            Vi = V[:, obs_indexes]
            Mi = M[i, obs_indexes]
            # print(Ri.shape, Mi.shape)
            U[i,:] = np.linalg.inv(lmb * var * np.identity(d) + Vi.dot(Vi.T)).dot(Vi.dot(Mi.T))

        for j in range(N2):
            obs_indexes = ~np.isnan(M[:,j])
            # print(sum(obs_indexes))
            Uj = U[obs_indexes,:]
            Mj = M[obs_indexes,j]
            # print(Ri.shape, Mi.shape)
            V[:,j] = np.linalg.inv(lmb * var * np.identity(d) + Uj.T.dot(Uj)).dot(Uj.T.dot(Mj.T))

        if t > 0:
            obj = get_objective(U, V, M)
            objective_list.append(obj)
            print('L = {}'.format(obj))
            print('Iteration {} complete.'.format(t))
            
    return objective_list
            
#%%
            
runs = 10
n_iter = 100

objective_array = np.zeros(shape = (runs, n_iter-1))
rmse_list = np.zeros(shape = (runs, 1))
for r in range(runs):
    print('Run {}'.format(r+1))
    U, V = get_UV()
    objective_array[r, :] = coord_ascent(U, V, M, N1, N2, n_iter = n_iter)
    rmse_list[r] = get_rmse(U, V, ratings_test)
    
plt.figure()
plt.xticks([int(x) for x in np.linspace(2,100, 10)])
plt.xlabel('Iteration')
plt.ylabel('Log Joint Likelihood')
for i in range(10):
    plt.plot(objective_array[i, :], label='Run {}'.format(i + 1))
    plt.legend(loc='best')
plt.show()
plt.savefig('C:/Users/Nicholas/Dropbox/Spring 2018/ML/HW4/2a_graph.png')

Results = pd.DataFrame(index = range(runs), columns=['Run','Objective','RMSE'])
Results['Run'] = list(range(1, runs + 1))
Results['Objective'] = objective_array[:, -1]
Results['RMSE'] = rmse_list
Results = Results.sort_values(by='Objective',axis=0,ascending=False)
print(Results)


#%%

for i in range(len(movies)):
    if ('Star Wars' in movies[i]):
        StarWars_i = i
    elif 'My Fair Lady' in movies[i]:
        MyFairLady_i = i
    elif 'GoodFellas' in movies[i]:
        Goodfellas_i = i

StarWars = V[:, StarWars_i]
MyFairLady = V[:, MyFairLady_i]
Goodfellas = V[:, Goodfellas_i]

StarWars = StarWars.reshape(len(StarWars), 1)
MyFairLady = MyFairLady.reshape(len(MyFairLady), 1)
Goodfellas = Goodfellas.reshape(len(Goodfellas), 1)

StarWarsDistance = np.linalg.norm(V - StarWars, axis=0)
MyFairLadyDistance = np.linalg.norm(V - MyFairLady, axis=0)
GoodfellasDistance = np.linalg.norm(V - Goodfellas, axis=0)

StarWarsSimilar_i = StarWarsDistance.flatten().argsort()[1:11]
MyFairLadySimilar_i = MyFairLadyDistance.flatten().argsort()[1:11]
GoodfellasSimilar_i = GoodfellasDistance.flatten().argsort()[1:11]

StarWarsSimilar = [movies[i] for i in StarWarsSimilar_i.tolist()]
MyFairLadySimilar = [movies[i] for i in MyFairLadySimilar_i.tolist()]
GoodfellasSimilar = [movies[i] for i in GoodfellasSimilar_i.tolist()]

df_StarWars = pd.DataFrame({'Similar movies': StarWarsSimilar, 'Distance':np.sort(StarWarsDistance.flatten())[1:11]})
df_MyFairLady = pd.DataFrame({'Similar movies': MyFairLadySimilar, 'Distance':np.sort(MyFairLadyDistance.flatten())[1:11]})
df_Goodfellas = pd.DataFrame({'Similar movies': GoodfellasSimilar, 'Distance':np.sort(GoodfellasDistance.flatten())[1:11]})

#df_StarWars.to_csv('Star Wars.csv', sep=',')
#df_MyFairLady.to_csv('My Fair lady.csv', sep=',')
#df_GoodFellas.to_csv('Good Fellas.csv', sep=',')