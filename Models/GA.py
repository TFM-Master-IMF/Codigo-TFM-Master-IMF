import numpy as np
import pandas as pd
import random

from os.path import dirname, abspath
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from Hyperparameter_optimization import evaluate_hyperparameter
from utils import split_train_test, plot_roc_curve


def initilization_of_population(size,n_feat):
    population = []
    for i in range(size):
        chromosome = np.ones(n_feat,dtype=np.bool)
        chromosome[:int(0.3*n_feat)]=False
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population

def fitness_score(population, model, X_train, X_test, y_train, y_test):
    scores = []
    for chromosome in population:
        model.fit(X_train.iloc[:,chromosome],y_train)
        predictions = model.predict(X_test.iloc[:,chromosome])
        scores.append(accuracy_score(y_test,predictions))
    scores, population = np.array(scores), np.array(population) 
    inds = np.argsort(scores)
    return list(scores[inds][::-1]), list(population[inds,:][::-1])

def selection(pop_after_fit,n_parents):
    population_nextgen = []
    for i in range(n_parents):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen

def crossover(pop_after_sel):
    population_nextgen=pop_after_sel
    for i in range(len(pop_after_sel)):
        child=pop_after_sel[i]
        child[3:7]=pop_after_sel[(i+1)%len(pop_after_sel)][3:7]
        population_nextgen.append(child)
    return population_nextgen

def mutation(pop_after_cross,mutation_rate):
    population_nextgen = []
    for i in range(0,len(pop_after_cross)):
        chromosome = pop_after_cross[i]
        for j in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[j]= not chromosome[j]
        population_nextgen.append(chromosome)
    return population_nextgen

def generations(size,n_feat,n_parents,mutation_rate,n_gen, model, X_train, X_test, y_train, y_test):
    best_chromo= []
    best_score= []
    population_nextgen=initilization_of_population(size,n_feat)
    for i in range(n_gen):
        scores, pop_after_fit = fitness_score(population_nextgen, model, X_train, X_test, y_train, y_test)
        pop_after_sel = selection(pop_after_fit,n_parents)
        pop_after_cross = crossover(pop_after_sel)
        population_nextgen = mutation(pop_after_cross,mutation_rate)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
    return best_chromo,best_score

def main():
    
    dataset = pd.read_csv(dirname(dirname(abspath(__file__))) + '/Ficheros Outputs/Datos.csv', sep=';', decimal=',')
    dataset.set_index('Date', inplace=True)

    X_train, X_val, X_test, y_train, y_val, y_test = split_train_test(dataset)
    num_features = X_train.shape[1]

    LDA = LinearDiscriminantAnalysis()

    chromo,score=generations(size=200,n_feat=num_features,n_parents=20,mutation_rate=0.05,
                     n_gen=20, model=LDA, X_train=X_train,X_test=X_val,y_train=y_train,y_test=y_val)
    best_chromo = chromo[score.index(max(score))]
    
    X_train =  X_train.iloc[:,best_chromo]
    X_val = X_val.iloc[:,best_chromo]
    X_test = X_test.iloc[:,best_chromo]

    params = evaluate_hyperparameter(LDA, X_train, X_val, y_train, y_val)

    model = LinearDiscriminantAnalysis(**params)
    model.fit(np.row_stack([X_train, X_val]), np.concatenate([y_train, y_val]))
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print('Accuracy achieved with the test set: ', accuracy_score(y_test, y_pred))
    plot_roc_curve(y_test, y_pred_proba)


    #print(X_train.iloc[:,chromo[-1]])
    #LDA.fit(np.row_stack([X_train.iloc[:,chromo[-1]], X_val.iloc[:,chromo[-1]]]), np.concatenate([y_train, y_val])) 
    #predictions = LDA.predict(X_test.iloc[:,chromo[-1]])
    #print("Accuracy score after genetic algorithm is= "+str(accuracy_score(y_test,predictions)))"""


if __name__ == "__main__":
    main()