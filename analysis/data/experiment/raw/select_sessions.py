import json


# read the json file
sessions = json.load(open('sessions_completed.json'))


# remove some fields from the trials field in the data
data = sessions[-1:]
for d in data:
    for t in d['trials']:
        t.pop("finished", None)
        t.pop("started_at", None)
        t.pop("finished_at", None)
        t.pop("solution", None)
        t.pop("social_learning_block_idx", None)
        t.pop("block_network_idx", None)





json_string=json.dumps(sessions[-1:]).replace("null", '""').replace("[]", '""')
with open('last_session.json', 'w') as f:
    f.write(json_string)
    f.close()
