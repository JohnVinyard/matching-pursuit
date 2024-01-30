from subprocess import check_output
import json


PROJECT_NAME = "matching-pursuit"
REGION = "us-central1"
REPOSITORY_NAME = "matching-pursuit-app-repo"
IMAGE_NAME = "matching-pursuit_api"
VM_NAME = "matching-pursuit-app-vm"

def execute(command):
    op = check_output(command, shell=True)
    return op

def json_command(command):
    resp = execute(command)
    data = json.loads(resp)
    return data

def list_repositories():
    command = f'''gcloud artifacts repositories list \
        --project={PROJECT_NAME} \
        --location={REGION} \
        --format=json
    '''
    data = json_command(command)
    return data

def create_repository():
    command = f'''gcloud artifacts repositories create {REPOSITORY_NAME} \
        --project={PROJECT_NAME} \
        --repository-format=docker \
        --location={REGION} \
        --description="Matching Pursuit Repository" \
        --format=json
    '''
    data = json_command(command)
    return data


def repository_name(path: str):
    segments = path.split('/')
    return segments[-1]

def repository_exists(name):
    repos = list_repositories()
    return any([name == repository_name(x['name']) for x in repos])

def ensure_repository_exists():
    print('Ensuring Repository Exists')
    
    exists = repository_exists(REPOSITORY_NAME)
    if not exists:
        print('Creating repository')
        create_repository()
    else:
        print('Repository already exists')


def build_local_image():
    print('Building Docker Image')
    tag = f'{REGION}-docker.pkg.dev/{PROJECT_NAME}/{REPOSITORY_NAME}/{IMAGE_NAME}:latest'
    command = f'sudo docker build . --tag {tag}'
    execute(command)

def push_image():
    print('Pushing Docker Image')
    # authenticate
    command = f'gcloud auth print-access-token'
    token = execute(command)
    print(f'Authenticating with token {token}')
    
    # the last character of the token is a newline
    cmd = f'sudo docker login -u oauth2accesstoken -p {token[:-1].decode()} https://us-central1-docker.pkg.dev'
    execute(cmd)
    
    # push
    command = f'sudo docker push {REGION}-docker.pkg.dev/{PROJECT_NAME}/{REPOSITORY_NAME}/{IMAGE_NAME}:latest'
    execute(command)


def list_compute_instances():
    command = f'''gcloud compute instances list \
        --project={PROJECT_NAME} \
        --format=json
    '''
    data = json_command(command)
    return data

def compute_instance_exists():
    data = list_compute_instances()
    return any([x['name'] == VM_NAME for x in data])

def create_compute_instance():
    print('Creating Compute Instance')
    command = f'''gcloud compute instances create-with-container {VM_NAME} \
        --container-image={REGION}-docker.pkg.dev/{PROJECT_NAME}/{REPOSITORY_NAME}/{IMAGE_NAME}:latest \
        --zone=us-central1-a \
        --format=json \
        --machine-type=c2=standard-4 \
        --boot-disk-size=50GB \
        --project={PROJECT_NAME}
    '''
    data = json_command(command)
    return data

def update_compte_instance():
    print('Updating Compute instance')
    command = f'''gcloud compute instances update-container {VM_NAME} \
        --container-image={REGION}-docker.pkg.dev/{PROJECT_NAME}/{REPOSITORY_NAME}/{IMAGE_NAME}:latest \
        --zone=us-central1-a \
        --format=json \
        --project={PROJECT_NAME}
    '''
    data = json_command(command)
    return data

def create_or_update_vm():
    compute_exists = compute_instance_exists()
    if not compute_exists:
        create_compute_instance()
    else:
        update_compte_instance()

def build_frontend():
    print(f'WARNING: This should buld the nextjs static app')
    
def assign_static_ip():
    print('WARNING: No static IP, follow instructions here https://cloud.google.com/sdk/gcloud/reference/compute/instances/update')

def deploy():
    build_frontend()
    ensure_repository_exists()
    build_local_image()
    push_image()
    create_or_update_vm()
    assign_static_ip()
    

if __name__ == '__main__':
    deploy()