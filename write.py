import argparse


def make_command(post_name):
    cmd = f'''python ../explainer/explainer.py --markdown ~/workspace/JohnVinyard.github.io/_metaposts/{post_name}.md --output ~/workspace/JohnVinyard.github.io/_posts --s3 zounds-blog-media --storage blog --watch'''
    return cmd

import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-name', type=str, required=True)
    args = parser.parse_args()
    
    # Open the VS Code instance where you'll be writing
    subprocess.Popen('code ~/workspace/JohnVinyard.github.io', shell=True)
    
    # Start up the explainer.py process that will transform the markdown
    # and embedded code
    subprocess.Popen(make_command(args.post_name), shell=True)
    
    # Prompt to start up the jekyll process
    input('Press any key once jekyll is running')
    
    # start up chrome so progress can be monitored    
    subprocess.Popen('google-chrome http://localhost:4000', shell=True)
    
    input('Get writing...')