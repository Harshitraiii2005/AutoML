pipeline{
    agent any

    enviroment {
        VENV_NAME ="automl-venv"
        PYTHON_VERSION = "python3"
    }

    stages {

        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Setup Python Virtual Enviroment'){
            steps {
                sh '''
                set -e

                if [ ! -d "$automl-venv" ]; then
                    echo "Creating virtual enviroment"
                    $PYTHON_VERSION -m venve $automl-venv
                fi

                source $automl-venv/bin/activate
                pip install --upgrade pip

                echo "Installing project requirements"
                pip install -r requirements.txt

                echo "Installing Jenkins, DVC, Prefect, mlflow"
                echo"First Installing Jenkins(ubuntu)"
                sudo apt update
                sudo apt install fontconfig openjdk-21-jre
                java -version
                sudo wget -O /etc/apt/keyrings/jenkins-keyring.asc \
                https://pkg.jenkins.io/debian-stable/jenkins.io-2023.key
                echo "deb [signed-by=/etc/apt/keyrings/jenkins-keyring.asc]" \
                    https://pkg.jenkins.io/debian-stable binary/ | sudo tee \
                    /etc/apt/sources.list.d/jenkins.list > /dev/null
                sudo apt update
                sudo apt install jenkins
                pip install dvc[all] prefect mlflow
                echo "all requirements installed successfully"
                '''


            }
        }

        stage('DVC PULL'){
            steps {
                sh '''
                set -e
                source automl-venv/bin/activate
                dvc pull -f dvc-pipeline.yaml
                '''
            }
        }

        stage('Run DVC Pipeline'){
            steps{
                sh '''
                set -e
                source automl-venv/bin/activate
                dvc repro -f dvc-pipeline.yaml
                '''
            }
        }
    }

    post {
        success {
            echo 'pipeline executed successfully'
        }
        failure {
            echo 'pipeline failed'
        }
        always{
            echo 'CI/CD run completed'
        }
    }


}