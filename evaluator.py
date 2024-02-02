import os # for checking whether files exist
import errno # for returning error messages
from ragas.metrics import ContextRelevancy, Faithfulness, AnswerRelevancy # metrics for assessment of RAG LLM
import pandas as pd # for working with dataframes
from datasets import Dataset # required format for the ragas metrics
from timeit import default_timer as timer # for timing execution
import openai # for passing responses to GPT-3.5-Turbo-16k for evaluation
openai.api_key = os.environ["OPENAI_API_KEY"]
# openai.api_key='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'

class Evaluator:
    '''
    Contains the data and functions needed to evaluate an LLM
    '''
        
    def __init__(self, config, retriever, generator):        

        self.config = config
        self.TEST_QUESTIONS_PATH = config['TEST_QUESTIONS_PATH']
        self.TEST_DATA_PATH = config['TEST_DATA_PATH']
        self.EVAL_RESULTS_PATH = config['EVAL_RESULTS_PATH']
        self.test_data_df = None
        
        self.retriever = retriever
        self.generator = generator
        
        self.get_test_data()
    
    
    def get_test_data(self) -> None:
        '''
        Retrieves test data from file if available in TEST_DATA_PATH.
        Test data is question-answer-context triples generated by the model.
        '''
        
        if os.path.isfile(self.TEST_DATA_PATH):
            
            self.test_data_df = pd.read_pickle(self.TEST_DATA_PATH)
            
        elif os.path.isfile(self.TEST_QUESTIONS_PATH):
            
            self.set_test_data()
            
        else:
            
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), self.TEST_DATA_PATH)
        
            
    def set_test_data(self) -> None:
        '''
        Generates question-answer-contexts triples from questions in data/questions.csv and saves to file
        '''
        
        test_data = []
        questions = pd.read_csv(self.TEST_QUESTIONS_PATH)
        questions = questions['question'].dropna()
        
        print(f'Building test data for {len(questions)} question(s) from {self.TEST_QUESTIONS_PATH}\n')
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
        self.test_data_df.to_pickle(self.TEST_DATA_PATH)
        self.test_data_df.to_csv('eval/testdata.csv')
        
                
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
                
        for i in range(151, num_questions):

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
            ragas_score.to_pickle('eval/ragas_score.pkl')
            ragas_score.to_csv('eval/ragas_score.csv')
            
        print('Evaluation completed and results saved.')
        