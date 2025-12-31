pipeline {
    agent any

    environment {
        VENV_NAME = "automl-venv"
        PYTHON_VERSION = "python3"
    }

    stages {

        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python Virtual Environment') {
            steps {
                sh '''
                set -e

                if [ ! -d "$VENV_NAME" ]; then
                    echo "Creating virtual environment"
                    $PYTHON_VERSION -m venv $VENV_NAME
                fi

                .   $VENV_NAME/bin/activate
                pip install --upgrade pip

                echo "Installing project requirements"
                pip install -r requirements.txt

                echo "Installing ML tools"
                pip install dvc[all] prefect mlflow
                '''
            }
        }

        stage('DVC Pull') {
            steps {
                sh '''
                set -e
                .   $VENV_NAME/bin/activate
                dvc pull
                '''
            }
        }

        stage('Run DVC Pipeline') {
            steps {
                sh '''
                set -e
                .   $VENV_NAME/bin/activate
                dvc repro
                '''
            }
        }
    }

    post {
        success {
            echo 'Pipeline executed successfully'
        }
        failure {
            echo 'Pipeline failed'
        }
        always {
            echo 'CI/CD run completed'
        }
    }
}
