import os # for checking whether files exist
import errno # for returning error messages
from ragas.metrics import ContextRelevancy, Faithfulness, AnswerRelevancy # metrics for assessment of RAG LLM
import pandas as pd # for working with dataframes
from datasets import Dataset # required format for the ragas metrics
from timeit import default_timer as timer # for timing execution
import openai # for passing responses to GPT-3.5-Turbo-16k for evaluation
openai.api_key = os.environ["OPENAI_API_KEY"]

class Evaluator:
    '''
    Contains the data and functions needed to evaluate an LLM
    '''
        
    def __init__(self, config, retriever, generator):        

        self.config = config

        self.MODEL = config['MODEL']
        self.TEMPERATURE = config['TEMPERATURE']
        self.TOP_P = config['TOP_P']
        self.NUM_CHUNKS = config['NUM_CHUNKS']
        self.CHUNK_SIZE = config['CHUNK_SIZE']
        self.test_data_df = None
        self.TEST_QUESTIONS_CSV = 'eval/questions.csv'
        self.TEST_DATA_PKL = f'eval/test_data-temp={str(self.TEMPERATURE)}-top_p={str(self.TOP_P)}-chunks={str(self.CHUNK_SIZE)}x{str(self.NUM_CHUNKS)}.pkl'
        self.TEST_DATA_CSV = f'eval/test_data-temp={str(self.TEMPERATURE)}-top_p={str(self.TOP_P)}-chunks={str(self.CHUNK_SIZE)}x{str(self.NUM_CHUNKS)}.csv'
        self.EVAL_RESULTS_PKL = f'eval/results-temp={str(self.TEMPERATURE)}-top_p={str(self.TOP_P)}-chunks={str(self.CHUNK_SIZE)}x{str(self.NUM_CHUNKS)}.pkl'
        self.EVAL_RESULTS_CSV = f'eval/results-temp={str(self.TEMPERATURE)}-top_p={str(self.TOP_P)}-chunks={str(self.CHUNK_SIZE)}x{str(self.NUM_CHUNKS)}.csv'
        
        self.retriever = retriever
        self.generator = generator
        
        self.get_test_data()
    
    
    def get_test_data(self) -> None:
        '''
        Retrieves test data from file if available in TEST_DATA_PKL.
        Test data is question-answer-context triples generated by the model.
        '''
        
        if os.path.isfile(self.TEST_DATA_PKL):
            
            self.test_data_df = pd.read_pickle(self.TEST_DATA_PKL)
            
        elif os.path.isfile(self.TEST_QUESTIONS_CSV):
            
            self.set_test_data()
            
        else:
            
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.TEST_DATA_PKL)
        
            
    def set_test_data(self) -> None:
        '''
        Generates question-answer-contexts triples from questions in data/questions.csv and saves to file
        '''
        
        test_data = []
        questions = pd.read_csv(self.TEST_QUESTIONS_CSV)
        questions = questions['question'].dropna()
        
        print(f'Building test data for {len(questions)} question(s) from {self.TEST_QUESTIONS_CSV}\n')
        start = timer()
        for i, question in enumerate(questions):
            try:
                print(f'Question {i + 1} of {len(questions)}: {question}\n')
                contexts, references = self.retriever.get_nearest_neighbours(question)
                context_list = list(contexts.split('\n'))
                answer = self.generator.get_llm_response(contexts, question)[1:-1]
                print(answer,'\n\n')
                test_data.append([question, answer, context_list])
                checkpoint = timer()
                elapsed_time = (checkpoint - start) / 60
                remaining_questions = len(questions) - (i+1)
                average_time = elapsed_time / (i+1)
                estimated_time_remaining = average_time * remaining_questions
                print(f'\nAnswered {i+1} of {len(questions)} in {elapsed_time:.2f} minutes. Estimated time left: {estimated_time_remaining:.2f} minutes.\n')
            except:
                print("Error. Max context length exceeded. Question not answered.")
                test_data.append([question, "Exceeded max context", ["NA"]])
                continue
        
        self.test_data_df = pd.DataFrame(test_data, columns=['question','answer','contexts'])
        self.test_data_df.to_pickle(self.TEST_DATA_PKL)
        self.test_data_df.to_csv(self.TEST_DATA_CSV)
        
                
    def evaluate_with_ragas(self) -> None:
        '''
        Evaluates the model using RAGAs for faithfulness, answer relevancy, and context relevancy.
        Sends each question-answer-context triple sequentially to GPT-3.5-Turbo-16k following the process
        described in the RAGAs paper. Saves results to EVAL_RESULTS_PATH as
        question-answer-contexts-answer_relevancy-faithfulness-context_relevancy. API timeouts are caught and
        'NA' is entered as the result.
        '''
        
        faithfulness = Faithfulness()
        answer_relevancy = AnswerRelevancy()
        context_relevancy = ContextRelevancy()
        
        ragas_score = pd.DataFrame(columns=['question','answer','contexts','answer_relevancy','faithfulness','context_relevancy'])
        num_questions = self.test_data_df.shape[0]
                
        for i in range(num_questions):

            print(f'Evaluating question-contexts-answer triple {i} of {num_questions}.')
            
            question_contexts_answer_df = self.test_data_df.iloc[[i]]
            question_contexts_answer = Dataset.from_pandas(question_contexts_answer_df)
                        
            answer_relevancy_result  = answer_relevancy.score(question_contexts_answer).to_pandas()
            answer_relevancy_score = answer_relevancy_result['answer_relevancy'].iloc[0]
            
            try:
                faithfulness_result = faithfulness.score(question_contexts_answer).to_pandas()
            except:
                faithfulness_result = question_contexts_answer_df.assign(faithfulness=['NA'])
            faithfulness_score = faithfulness_result['faithfulness'].iloc[0]

            try:
                context_relevancy_result = context_relevancy.score(question_contexts_answer).to_pandas()
            except:
                context_relevancy_result = question_contexts_answer_df.assign(context_relevancy=['NA'])
            context_relevancy_score = context_relevancy_result['context_relevancy'].iloc[0]

            print(f'Result {i}:\nFaithfulness:      {faithfulness_score}\nAnswer_relevancy:  {answer_relevancy_score}\
                \nContext relevancy: {context_relevancy_score}\n')
            
            question_score = pd.concat([answer_relevancy_result, faithfulness_result['faithfulness'], context_relevancy_result['context_relevancy']], axis=1, ignore_index=True)
            ragas_score = pd.concat([ragas_score, question_score], axis=0)
            ragas_score.to_pickle(self.EVAL_RESULTS_PKL)
            ragas_score.to_csv(self.EVAL_RESULTS_CSV)
            
        print('Evaluation completed and results saved.')
        
        # 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