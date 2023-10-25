import os
from datetime import datetime

import requests
from dotenv import load_dotenv

if __name__ == '__main__':

    # directory management
    wdir = os.getcwd()
    ddir = os.path.join(wdir,'data')

    # read constants from environment variables
    load_dotenv()
    BACKEND_USER = os.environ['BACKEND_USER']
    BACKEND_PASSWORD = os.environ['BACKEND_PASSWORD']
    BACKEND_URL = os.environ['BACKEND_URL']
    EXPERIMENT_TYPE=os.environ['EXPERIMENT_TYPE']
    FINISHED=os.environ['FINISHED']

    url = f'https://{BACKEND_URL}/results'
    headers = {'Accept': 'application/json'}
    auth = (BACKEND_USER, BACKEND_PASSWORD)
    current_datatime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # ---subjects----
    # subjects = requests.get(f'{url}/subjects', headers=headers, auth=auth)
    # with open(f'subjects_{current_datatime}_TEST.json', 'wb') as out_file:
    #    out_file.write(subjects.content)

    # ---sessions----
    # sessions = requests.get(f'{url}/sessions?experiment_type={EXPERIMENT_TYPE}&finished={FINISHED}', headers=headers,
    # auth=auth)

    # test, finished is false here (normally true)
    sessions = requests.get(f'{url}/sessions?experiment_type={EXPERIMENT_TYPE}&finished={FINISHED}', headers=headers,
                            auth=auth)

    # create rawdata project folder if not existent and download data in it
    if not os.path.exists(os.path.join(ddir,'raw',EXPERIMENT_TYPE)):
            os.makedirs(os.path.join(ddir,'raw',EXPERIMENT_TYPE))
    with open(os.path.join(ddir,'raw',EXPERIMENT_TYPE,f'sessions_{EXPERIMENT_TYPE}_{current_datatime}.json'), 'wb') as out_file:
        out_file.write(sessions.content)
