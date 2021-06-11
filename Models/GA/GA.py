import numpy as np
import pandas as pd
import warnings

from os.path import dirname, abspath
import sys
sys.path.insert(0, dirname(dirname(abspath(__file__))))

from random import randint

from sklearn.linear_model import LogisticRegression 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import split_train_test, plot_roc_curve


class GeneticAlgorithm:
    def __init__(self, data, feature_list, target, n_pop, max_iter, model):
        self.data = data
        self.feature_list = feature_list
        self.target= target
        self.n_pop = n_pop
        self.max_iter= max_iter
        self.model = model
        self.population = self.init_population()

    def init_population(self):
        # initial population of random bitstring
        return np.array([list(np.random.randint(low = 0,high=2,size=len(self.feature_list))) for _ in range(self.n_pop)])

    def get_fitness(self, individual): 
        columns = [feature for counter, feature in enumerate(self.feature_list) if individual[counter]==1]           
        X_train, X_test, y_train, y_test = train_test_split(self.data[columns], self.data[self.target], test_size=0.2, shuffle=False)
        self.model.fit(X_train,y_train) 
        y_pred = self.model.predict(X_test)  
        accuracy = accuracy_score(y_test, y_pred)
        return accuracy               

    def selection(self, scores, k=3):
        # first random selection
        selection_ix = np.random.randint(0, len(self.population), 1)[0]
        for ix in np.random.randint(0, len(self.population), k-1):
            # check if better (e.g. perform a tournament)
            if scores[ix] < scores[selection_ix]:
                selection_ix = ix
        return self.population[selection_ix]

    def crossover(self, p1, p2, r_cross=0.1):
        # children are copies of parents by default
        c1, c2 = p1.copy(), p2.copy()
        # check for recombination
        if np.random.rand() < r_cross:
            # select crossover point that is not on the end of the string
            pt = randint(1, len(p1)-2)
            # perform crossover
            c1 = np.append(p1[:pt], p2[pt:])
            c2 = np.append(p2[:pt], p1[pt:])
        return [c1, c2]
    
    def mutation(self, bitstring, r_mut=0.1):
        for i in range(len(bitstring)):
            # check for a mutation
            if np.random.rand() < r_mut:
                # flip the bit
                bitstring[i] = 1 - bitstring[i]

    def ga(self):
        # keep track of best solution
        best, best_eval = 0, self.get_fitness(self.population[0])
        # enumerate generations
        for gen in range(self.max_iter):
            # evaluate all candidates in the population
            scores = [self.get_fitness(c) for c in self.population]
            # check for new best solution
            for i in range(self.n_pop):
                if scores[i] > best_eval:
                    best, best_eval = self.population[i], scores[i]
                    #print(f'New best population {self.population[i]}')
                    #print(f'Accuracy achieved with the test set = {scores[i]*100} %\n')
            # select parents
            selected = [self.selection(scores) for _ in range(self.n_pop)]
            # create the next generation
            children = list()
            for i in range(0, self.n_pop, 2):
                # get selected parents in pairs
                p1, p2 = selected[i], selected[i+1]
                # crossover and mutation
                for c in self.crossover(p1, p2, 0.2):
                    # mutation
                    self.mutation(c, 0.2)
                    # store for next generation
                    children.append(c)
            # replace population
            self.population = children
        return [best, best_eval]

def eval_feature_selection(model, X_train_GA, X_test_GA, y_train_GA, y_test_GA):
    previous_acc = 0
    for i in range(200):  
        #model.fit(np.row_stack([X_train_GA, X_val_GA]), np.concatenate([y_train_GA, y_val_GA]))
        model.fit(X_train_GA, y_train_GA)
        y_pred = model.predict(X_test_GA)
        #y_pred_proba = model.predict_proba(X_test_GA)[:, 1]
        current_acc = accuracy_score(y_test_GA, y_pred)
        if current_acc > previous_acc:
            print(f'Accuracy achieved with the test set = {accuracy_score(y_test_GA, y_pred)}',)
            print(f'Precision achieved with the test set = {precision_score(y_test_GA, y_pred)}')
            print(f'Recall achieved with the test set = {recall_score(y_test_GA, y_pred)}')
            print(f'F1 Score achieved with the test set= {f1_score(y_test_GA, y_pred)}')
            previous_acc = current_acc
        #plot_roc_curve(y_test_GA, y_pred_proba)

def main():
    np.random.seed(0)

    # Ignore desired warning
    warnings.filterwarnings("ignore", message="Variables are collinear")
    
    dataset = pd.read_csv(dirname(dirname(dirname(abspath(__file__)))) + '/Ficheros Outputs/Datos.csv', sep=';', decimal=',')
    dataset.set_index('Date', inplace=True)
    dataset.drop('Bitcoin Stock Price (USD)', axis=1, inplace=True)

    # Preparation of the data to feed the Genetic Algorithm
    X_train, X_test, y_train, y_test = split_train_test(dataset)
    X = X_train.append(X_test) # .append(X_test)
    y = y_train.append(y_test) #.append(y_test)
    df = pd.concat([X,y], axis=1)
    target, feature_list = 'Bitcoin sign change', [i for i in df.columns if i != 'Bitcoin sign change']

    models_for_GA = [LogisticRegression(solver='liblinear'), LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(), RandomForestClassifier(), SVC(), XGBClassifier(use_label_encoder=False, eval_metric='logloss')] #, LinearDiscriminantAnalysis(), QuadraticDiscriminantAnalysis(reg_param=0.0), RandomForestClassifier(), XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    for i, model in enumerate(models_for_GA):
        # Executing Genetic Algorithm to find the most relevant features
        Genetic_Algorithm = GeneticAlgorithm(df, feature_list, target, 10, 100, model)
        feature_set, acc_score = Genetic_Algorithm.ga() 
        # Filter Selected Features
        feature_set = [feature_list[j] for j in range(len(feature_list)) if feature_set[j]==1]
        # Print List of Features
        print(f'------------------------------------------------------------------------------------------{type(model).__name__}-----------------------------------------------------------------------------------------------')
        print(f'Feature selection for the algorithm {type(model).__name__}')
        print('Optimal Feature Set\n',feature_set,'\nNumber of features considered = ', len(feature_set)) #'\nAccuracy achieved with the validation set (GA) =', acc_score)
        print(f'\nEvaluation of the future selection process for the algorithm {type(model).__name__}')

        eval_feature_selection(model, X_train.loc[:,feature_set], X_test.loc[:,feature_set], y_train, y_test)

if __name__ == "__main__":
    main()



