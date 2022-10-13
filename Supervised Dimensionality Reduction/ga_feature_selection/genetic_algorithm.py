import numpy as np
import rich
from rich.table import Table
import argparse
import multiprocessing as mp
import numbers
import time

from sklearn.base import is_classifier
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler

from Utils.metrics import return_regression_result, return_classification_result



class GA_FeatureSelector():
    def __init__(self, model, args: argparse, seed):
        self.args = args
        self.seed = seed
        
        # A supervised learning estimator with a 'fit' method from scikit-learn
        self.model = model
        
        # Scaling - Normalization
        self.normalization = args.normalization
        
        # Evaluation Metirc
        if is_classifier(self.model):
            self.criterion = args.c_metric
        else:
            self.criterion = args.r_metric
        
        # Number of generations
        if args.n_generation > 0:
            self.n_generation = args.n_generation
        else:
            raise ValueError('The number of generations must be greater than 1')

        # Size of population (number of chromosomes)
        self.n_population = args.n_population
        
        # Crossover and mutations likelihood
        if args.crossover_rate <= 0.0 or args.mutation_rate <= 0.0 or args.crossover_rate > 1.0 or args.mutation_rate > 1.0:
            raise ValueError('Mutation and crossover rate must be a value in the range (0.0, 1.0]')
        else:
            self.crossover_rate = args.crossover_rate
            self.mutation_rate = args.mutation_rate
        
        # Tournament size in selection process
        self.tournament_k = args.tournament_k
        
        # Number of threads
        if args.n_jobs < 0 and args.n_jobs == -1:
            self.n_jobs = mp.cpu_count()
        elif args.n_jobs > 0 and args.n_jobs <= mp.cpu_count():
            self.n_jobs = args.n_jobs
        else:
            raise ValueError(
                f'n_jobs == {args.n_jobs} is invalid! You have a maximum of'
                f' {mp.cpu_count()} cores.')
        
        # for randomizing initial population   
        self.initial_best_chromosome = args.initial_best_chromosome
        
        # set random seed
        if isinstance(self.seed, numbers.Integral):
            np.random.seed(self.seed)
            
        # Verbose
        if args.verbose < 0:
            self.verbose = 0
        else:
            self.verbose = args.verbose
        
        # Best chromosome (np.ndarray, float, int) (chromosome, score, generation index)
        self.best_chromosome = (None, None, None)
        
        # Population convergence variables
        self.convergence = False
        self.n_times_convergence = 0
        self.threshold = 1e-6


    def data_prepare(self, X: np.ndarray, y: np.ndarray):
        # Train and Test data split
        train_index, test_index = train_test_split(np.arange(len(X)), test_size=.2, random_state=self.seed)
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index].reshape(-1,1), y[test_index].reshape(-1,1)
        
        # Scaling both Train and Testing data (Both X and Y)
        if self.normalization:
            scaler = MinMaxScaler()
            full_train_scaled = scaler.fit_transform(np.concatenate([X_train, y_train],axis=1))
            full_test_scaled = scaler.transform(np.concatenate([X_test, y_test],axis=1))
            
            self.y_min = scaler.data_min_[-y_test.shape[1]:]
            self.y_max = scaler.data_max_[-y_test.shape[1]:]
            
            X_train , y_train, X_test, y_test = full_train_scaled[:,:-y_test.shape[1]] , full_train_scaled[:,-y_test.shape[1]:] , full_test_scaled[:,:-y_test.shape[1]] , full_test_scaled[:,-y_test.shape[1]:]
            
        # Return training and testing data.
        return X_train, X_test, y_train, y_test
    
        
    def run(self, 
            X_train,
            X_test,
            y_train,
            y_test):
        
        # Initialize output variables
        self.train_scores = []
        self.val_scores = []
        self.chromosomes_history = []
        
        # Time when training begins
        training_start = time.time()
        
        # Data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        # Initialize population
        rich.print(f'[bold magenta]Creating initial population with {self.n_population} chromosomes', ":dna:")
        
        ## Create n_population chromosomes
        self.population = np.random.randint(2, size=(self.n_population, self.X_train.shape[1]))
        
        # Insert initial best chromosome if is defined
        if self.initial_best_chromosome is None:
            
            # Evaluate initial population and update best_chromosome
            rich.print(' :heavy_check_mark:', 'Evaluating initial population...')
            pool = mp.Pool(self.n_jobs)
            population_scores = np.array(pool.map(self.evaluate_fitness, self.population))
            best_chromosome_index = np.argmax(population_scores[:, 0])
            
        elif type(self.initial_best_chromosome) == np.ndarray and len(self.initial_best_chromosome) == self.X_train.shape[1] and len(np.where((self.initial_best_chromosome <= 1) & (self.initial_best_chromosome >= 0))[0]):
            # Introduce the best in the population
            index_insertion = np.random.randint(self.n_population)
            self.population[index_insertion] = self.initial_best_chromosome
            
            # Evaluate initial population and update best_chromosome
            print(' :heavy_check_mark:', 'Evaluating initial population...')
            pool = mp.Pool(self.n_jobs)
            population_scores = np.array(pool.map(self.evaluate_fitness, self.population))
            best_chromosome_index = index_insertion
            
        else:
            raise ValueError('Initial best chromosome must be a 1 '
                             'dimensional binary array with a length of '
                             'X.shape[1]')
        
        # Update best chromosome found
        self.update_best_chromosome(
            self.population, population_scores[:, 0],
            best_chromosome_index, i_gen=0
        )
        
        # Save result
        self.save_results(val_score=population_scores[best_chromosome_index, 0],
                          best_current_chromosome=self.population[best_chromosome_index],
                          train_score=population_scores[best_chromosome_index, 1])
        
        # Loop until evaluation converge
        i = 0
        while i < self.n_generation and not self.convergence:
            # Time when generation begins
            generation_start = time.time()
            rich.print(f'[bold magenta]Creating generation {i + 1}...')
            
            # Step 1: Selection
            new_population = self.selection(self.population, population_scores[:, 0], best_chromosome_index)
            # print(f'\n Selection {i+1} done')
            
            # Step 2: Crossover
            new_population = self.crossover(new_population)
            # print(f'\n Crossover {i+1} done')
            
            # Step 3: Mutation
            new_population = self.mutation(new_population)
            # print(f'\n Mutation {i+1} done')
            
            # Step 4: Replace previous population with new_population
            self.population = new_population.copy()
            
            # Step 5: Evaluate new population and update best_chromosome
            rich.print(' :heavy_check_mark:', f'Evaluating population of new generation {i+1}...')
            pool = mp.Pool(self.n_jobs)
            population_scores = np.array(pool.map(self.evaluate_fitness, self.population))
            best_chromosome_index = np.argmax(population_scores[:, 0])
            self.update_best_chromosome(
                self.population, population_scores[:, 0],
                best_chromosome_index, i+1)
            
            # Save results
            self.save_results(
                val_score=population_scores[best_chromosome_index, 0],
                best_current_chromosome=self.population
                [best_chromosome_index],
                train_score=population_scores[best_chromosome_index, 1]
            )
            
            # Step 6: Next generation
            i = i+1
            
            # Time when generation ends
            generation_end = time.time() - generation_start
            rich.print('    [bold]Elapsed generation time: ', f'%.2f [bold]seconds' % generation_end)
        
        
        # Time when training ends
        self.training_end = time.time() - training_start
        rich.print('[bold]Training time[/bold]: ', f'%.2f [bold]seconds' % self.training_end)
            
            
    
    def evaluate_fitness(self, chromosome):
        # Select those features with ones in chromosome
        x = [x.astype(bool) for x in chromosome]
        X_train_ = self.X_train[:, x]
        X_test_ = self.X_test[:, x]

        # fit
        self.model.fit(X_train_, self.y_train.ravel())
        
        if self.normalization:  # Undo the testing-data scaling of Y(target) according to fitted training-data scaling about Y(target)
            y_train_pred = self.model.predict(X_train_) * (self.y_max-self.y_min) + self.y_min
            y_test_pred = self.model.predict(X_test_) * (self.y_max-self.y_min) + self.y_min
            
        else:
            y_train_pred = self.model.predict(X_train_)
            y_test_pred = self.model.predict(X_test_)
            
        
        if is_classifier(self.model):
            y_train_proba = self.model.predict_proba(X_train_)
            y_test_proba = self.model.predict_proba(X_test_)
            train_result = return_classification_result(self.y_train, y_train_pred, y_train_proba)
            test_result = return_classification_result(self.y_test, y_test_pred, y_test_proba)
            return (train_result[self.criterion], test_result[self.criterion])
            
        else:
            train_result = return_regression_result(self.y_train, y_train_pred)
            test_result = return_regression_result(self.y_test, y_test_pred)
            return (train_result[self.criterion], test_result[self.criterion])
    
        
        
    def update_best_chromosome(self, population, population_scores, best_chromosome_index, i_gen: int):
        # Initialize best_chromosome
        if self.best_chromosome[0] is None and self.best_chromosome[1] is None:
            self.best_chromosome = (
                population[best_chromosome_index],
                population_scores[best_chromosome_index],
                i_gen
            )
            self.threshold_times_convergence = 5
            
        # Update if new generation is better
        elif population_scores[best_chromosome_index] > self.best_chromosome[1]:
            if i_gen >= round(0.5 * self.n_generation):
                self.threshold_times_convergence = int(np.ceil(0.3 * i_gen))
            self.best_chromosome = (
                population[best_chromosome_index],
                population_scores[best_chromosome_index],
                i_gen
            )
            self.n_times_convergence = 0
            rich.print(' :heavy_check_mark:', f'[bold green](Better)[/bold green] A better chromosome than the current one has been found {self.best_chromosome[1]}')
        
        # If is smaller than self.threshold count it until convergence
        elif abs(population_scores[best_chromosome_index] - self.best_chromosome[1]) <= self.threshold:
            self.n_times_convergence = self.n_times_convergence + 1
            rich.print(' :heavy_check_mark:', f'[magenta]Same scoring value found {self.n_times_convergence} / {self.threshold_times_convergence} times.')
            if self.n_times_convergence == self.threshold_times_convergence:
                self.convergence = True
        
        else:
            self.n_times_convergence = 0
            rich.print(' :heavy_check_mark:', f'[bold red](WORSE)[/bold red] No better chromosome than the current one has been found {population_scores[best_chromosome_index]}')
        
        rich.print(" :heavy_check_mark:", f'Current best chromosome:', f'{self.best_chromosome[0]},', f'Score: {self.best_chromosome[1]}')
        
        
        
    def save_results(self, val_score, best_current_chromosome, train_score: float = None):
        self.val_scores.append(val_score)
        if train_score is not None:
            self.train_scores.append(train_score)
        self.chromosomes_history.append(best_current_chromosome)
        
    
    
    def selection(self, population, population_scores, best_chromosome_index):
        # Create new population
        new_population = [population[best_chromosome_index]]
        
        # Tournament_k chromosome tournament until fill the numpy array
        while len(new_population) != self.n_population:
            # Generate tournament_k positions randomly
            k_chromosomes = np.random.randint(len(population), size = self.tournament_k)
            # Get the best onr of these tournament_k chromosomes
            best_of_tournament_index = np.argmax(population_scores[k_chromosomes])
            # Append it to the new population
            new_population.append(population[k_chromosomes[best_of_tournament_index]])
            
        return np.array(new_population)
    
    
    def crossover(self, population):
        # Define the number of crosses
        n_crosses = int(self.crossover_rate * int(self.n_population / 2))
        
        # Make a copy from current population
        crossover_population = population.copy()
        
        # Make n_crosses crosses
        for i in range(0, n_crosses*2, 2):
            cut_index = np.random.randint(1, self.X_train.shape[1])
            tmp = crossover_population[i, cut_index:].copy()
            crossover_population[i, cut_index:], crossover_population[i+1, cut_index:] = crossover_population[i+1, cut_index:], tmp
            # Avoid null chromosomes
            if not all(crossover_population[i]):
                crossover_population[i] = population[i]
            if not all(crossover_population[i+1]):
                crossover_population[i+1] = population[i+1]
                
        return crossover_population
    
    
    def mutation(self, population):
        # Define number of mutations to do
        n_mutations = int(
            self.mutation_rate * self.n_population * self.X_train.shape[1])
        
        # Mutating n_mutations genes
        for _ in range(n_mutations):
            chromosome_index = np.random.randint(0, self.n_population)
            gene_index = np.random.randint(0, self.X_train.shape[1])
            population[chromosome_index, gene_index] = 0 if \
                population[chromosome_index, gene_index] == 1 else 1
                
        return population
        
        
    def summary_table(self):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Best Chromosome", justify="center")
        table.add_column("Selected Features ID", justify="center")
        table.add_column("Best Test Score", justify="center")
        table.add_column("Best Generation", justify="center")
        table.add_column("Best Train Score", justify="center")
        table.add_column("Training Time (sec)", justify="center")
        table.add_row(str(self.best_chromosome[0]), 
                      str(np.where(self.best_chromosome[0])[0]),
                      str(self.best_chromosome[1]),
                      str(self.best_chromosome[2]),
                      str(np.max(self.train_scores)),
                      str(round(self.training_end, 2))
                      )
        
        table_history = Table(show_header=True, header_style="bold green")
        table_history.add_column("Chromosomes History", justify="center")
        for history in self.chromosomes_history:
            table_history.add_row(str(history))
            
        parameter_table = Table(show_header=True, header_style="bold blue")
        parameter_table.add_column("Number of Generation", justify="center")
        parameter_table.add_column("Number of Population", justify="center")
        parameter_table.add_column("Crossover Rate", justify="center")
        parameter_table.add_column("Mutation Rate", justify="center")
        parameter_table.add_column("Metric", justify="center")
        parameter_table.add_row(
            str(self.n_generation),
            str(self.n_population),
            str(self.crossover_rate),
            str(self.mutation_rate),
            str(self.criterion)
        )
        
        return table, parameter_table