pipeline {
    agent any

    environment {
        VENV_NAME = "automl-venv"
        PYTHON_VERSION = "python3"
        IMAGE_NAME = "harshitrai20/automl"
        IMAGE_TAG = "${BUILD_NUMBER}"
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
                    $PYTHON_VERSION -m venv $VENV_NAME
                fi
                . $VENV_NAME/bin/activate
                pip install --upgrade pip
                pip install -r requirements.txt
                pip install dvc[all] prefect mlflow
                '''
            }
        }

        stage('DVC Pull') {
            steps {
                sh '''
                set -e
                . $VENV_NAME/bin/activate
                dvc pull
                '''
            }
        }

        stage('Run DVC Pipeline') {
            steps {
                sh '''
                set -e
                . $VENV_NAME/bin/activate
                dvc repro
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                docker build -t $IMAGE_NAME:$IMAGE_TAG .
                '''
            }
        }

        stage('Push Docker Image') {
            steps {
                sh '''
                docker tag $IMAGE_NAME:$IMAGE_TAG $IMAGE_NAME:latest
                docker push $IMAGE_NAME:$IMAGE_TAG
                docker push $IMAGE_NAME:latest
                '''
            }
        }
    }

    post {
        success {
            echo 'CI pipeline executed successfully'
        }
        failure {
            echo 'CI pipeline failed'
        }
        always {
            echo 'CI run completed'
        }
    }
}
