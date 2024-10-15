import yaml
import pandas as pd

class TitanicDataBaseManager:
    _GENDER_SUBMISSION = 'gender_submission.csv'
    _TRAIN = 'train.csv'
    _TEST = 'test.csv'

    def __init__(self):
        """Construtor que carrega os dados a partir do arquivo de configuração."""
        self._carregar_config()
        self._train_df = None
        self._test_df = None
        self._gender_submission_df = None
        self._carregar_dados()

    def _carregar_config(self):
        """Carrega as configurações do arquivo config.yaml."""
        with open('config.yaml', 'r') as file:
            self._config = yaml.safe_load(file)

    def _carregar_dados(self):
        """Carrega os dados dos arquivos CSV com base nas configurações."""
        path = self._config['processamento']['titanic']['path']
        self._train_df = pd.read_csv(path + self._TRAIN)
        self._test_df = pd.read_csv(path + self._TEST)
        self._gender_submission_df = pd.read_csv(path + self._GENDER_SUBMISSION)

    def get_train(self):
        """Retorna o DataFrame do conjunto de treino."""
        return self._train_df

    def get_test(self):
        """Retorna o DataFrame do conjunto de teste."""
        return self._test_df

    def get_gender_submission(self):
        """Retorna o DataFrame do arquivo de submissão de gênero."""
        return self._gender_submission_df
    
    def exibirDados(self, csvData):
        print("Intervalo de linhas do train.csv:")
        print("----------------------------------------------------------------------------------")
        print(csvData.head())
        print("----------------------------------------------------------------------------------")

        print("\nInformações do train.csv:")
        print("----------------------------------------------------------------------------------")
        print(csvData.info())
        print("----------------------------------------------------------------------------------")

        print("\nEstatísticas descritivas do train.csv:")
        print("----------------------------------------------------------------------------------")
        print(csvData.describe())
        print("----------------------------------------------------------------------------------")