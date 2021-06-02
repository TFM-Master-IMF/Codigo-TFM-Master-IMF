import numpy as np
import pandas as pd
import math

from os.path import dirname, abspath

from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from Hyperparameter_optimization import evaluate_hyperparameter
from utils import split_train_test, plot_roc_curve


class GeneticAlgorithm:
    def __init__(self, data, feature_list, target, n, max_iter):
        self.data = data
        self.feature_list = feature_list
        self.target= target
        self.n = n
        self.max_iter= max_iter
        self.population = self.init_population(len(self.feature_list))


    def init_population(self, c):
        return np.array([[math.ceil(e) for e in pop] for pop in (np.random.rand(self.n, c)-0.5)])

    def single_point_crossover(self):
        r,c, n = self.population.shape[0], self.population.shape[1], np.random.randint(1, self.population.shape[1])         
        for i in range(0,r,2):                
            self.population[i], self.population[i+1] = np.append(self.population[i][0:n], self.population[i+1][n:c]),np.append(self.population[i+1][0:n], self.population[i][n:c])        

    def flip_mutation(self):
        return self.population.max() - self.population

    def random_selection(self):
        r = self.population.shape[0]
        new_population = self.population.copy()    
        for i in range(r):        
            new_population[i] = self.population[np.random.randint(0,r)]
        return new_population

    def get_fitness(self):    
        fitness = []
        for i in range(self.population.shape[0]):        
            columns = [self.feature_list[j] for j in range(self.population.shape[1]) if self.population[i,j]==1]                    
            fitness.append(self.predictive_model(self.data[columns], self.data[self.target]))                
        return fitness

    def predictive_model(self, X,y):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, shuffle=False)
        lr = LogisticRegression(solver='liblinear', max_iter=100)
        lr.fit(X_train,y_train)    
        return accuracy_score(y_test, lr.predict(X_test))

    def ga(self):
        
        fitness = self.get_fitness()    
        
        optimal_value = max(fitness)
        optimal_solution = self.population[np.where(fitness==optimal_value)][0]    
        
        for i in range(self.max_iter):                
            self.population = self.random_selection()
            self.single_point_crossover()

            if np.random.rand() < 0.3:
                self.population = self.flip_mutation()   
                            
            fitness = self.get_fitness()
                    
            if max(fitness) > optimal_value:
                optimal_value = max(fitness)
                optimal_solution = self.population[np.where(fitness==optimal_value)][0]                               
            
        return optimal_solution, optimal_value

def main():
    
    dataset = pd.read_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/Datos.csv', sep=';', decimal=',')
    dataset.set_index('Date', inplace=True)
    dataset.drop('Bitcoin Stock Price (USD)', axis=1, inplace=True)

    # Preparation of the data to feed the Genetic Algorithm
    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(dataset)
    X = X_train.append(X_val)
    y = y_train.append(y_val)
    df = pd.concat([X,y], axis=1)
    target, feature_list = 'Bitcoin sign change', [i for i in df.columns if i != 'Bitcoin sign change']

    # Executing Genetic Algorithm to find the most relevant features
    Genetic_Algorithm = GeneticAlgorithm(df, feature_list, target, 30, 1000)
    feature_set, acc_score = Genetic_Algorithm.ga() 
    # Filter Selected Features
    feature_set = [feature_list[i] for i in range(len(feature_list)) if feature_set[i]==1]

    # Print List of Features
    print('Optimal Feature Set\n',feature_set,'\nOptimal Accuracy =', acc_score*100, '%', 'Number of features considered = ', len(feature_set))

if __name__ == "__main__":
    main()