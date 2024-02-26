import random
from src.dataset.dataset_factory import DatasetFactory
from src.evaluation.evaluation_metric_factory import EvaluationMetricFactory
from src.evaluation.evaluator_base import Evaluator
from src.explainer.explainer_factory import ExplainerFactory
from src.oracle.embedder_factory import EmbedderFactory
from src.oracle.oracle_factory import OracleFactory
from src.utils.context import Context


class EvaluatorManager:

    def __init__(self, context: Context) -> None:
        self.context = context
        self._output_store_path = self.context.output_store_path
        self._datasets = [] 
        self._evaluators = []
        
        #NOTE: Move the Factories creation outside?
        self.context.factories['datasets'] = DatasetFactory(context)
        self.context.factories['embedders'] = EmbedderFactory(context)
        self.context.factories['oracles'] = OracleFactory(context)
        self.context.factories['explainers'] = ExplainerFactory(context)
        self.context.factories['metrics'] = EvaluationMetricFactory(context.conf)

        self._create_evaluators()

    def get_dataset_instances(self):
        # Assumi che self.dataset sia l'attributo che contiene il dataset generato
        return self.dataset.instances
    
    @property
    def evaluators(self):
        return self._evaluators
    
        
    def get_dataset(self):
        # Restituisce l'elenco dei datasets
        return self._datasets

    def _create_evaluators(self):
        # Get the lists of main componets from the main configuration file.
        '''datasets_list = self.context.conf['datasets']
        oracles_list = self.context.conf['oracles']'''

        do_pairs_list = self.context.conf['do-pairs']
        metrics_list = self.context.conf['evaluation_metrics']
        explainers_list = self.context.conf['explainers']
        evaluation_metrics = []

        # Shuffling dataset_oracles pairs and explainers will enabling by chance
        # parallel distributed cration and training.
        random.shuffle(do_pairs_list)
        random.shuffle(explainers_list) 

        # Instantiate the evaluation metrics that will be used for the evaluation;
        for metric_dict in metrics_list:
            evaluation_metrics.append(self.context.factories['metrics'].get_evaluation_metric_by_name(metric_dict))

        for explainer_snippet in explainers_list:
            for do_pair_snippet in do_pairs_list:
                # The get_dataset method return an already builded/loaded/generated dataset with all its features already in place;
                dataset = self.context.factories['datasets'].get_dataset(do_pair_snippet['dataset'])
                self._datasets.append(dataset)  # Memorizza il dataset per l'accesso esterno
          
                
    def evaluate(self):
        """Evaluates each combination of dataset-oracle-explainer using the chosen evaluation metrics
         
        -------------
        INPUT: None

        -------------
        OUTPUT: None
        """
        for evaluator in self._evaluators:
            evaluator.evaluate()

        
    def evaluate_multiple_runs(self, n_runs):
        """Evaluates each combination of dataset-oracle-explainer using the chosen evaluation metrics.
        Each evaluator is run "n_runs" times
         
        -------------
        INPUT: n_runs the number of times the evaluate method of each evaluator is going to be called

        -------------
        OUTPUT: None
        """
        for evaluator in self._evaluators:
            for i in range(0, n_runs):
                evaluator.evaluate()
