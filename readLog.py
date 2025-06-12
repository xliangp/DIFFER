import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import shutil
from tensorboard.backend.event_processing import event_accumulator

def read_tensorboard_data(logdir):
    # Initialize an event accumulator
    ea = event_accumulator.EventAccumulator(logdir,
        size_guidance={  # see the tensorboard documentation for more available options
            event_accumulator.SCALARS: 0,  # 0 means load all scalar events
        }
    )
    
    # Load all events from the logdir
    ea.Reload()
    
    # Get all scalar tags (for example)
    scalar_tags = ea.Tags()['scalars']
    scalar_tags=list(set.intersection(set(scalar_tags),set(['rank1','mAP'])))

    # Access scalar data by tag
    
    data = {}
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        data[tag] = [(e.wall_time, e.step, e.value) for e in events]

    return data

# Example usage: Replace 'path_to_tensorboard_logdir' with the path to your TensorBoard log directory
from pathlib import Path
script_path = str(Path(__file__).absolute())
#script_path='/home/xi860799/code/MADE/readLog.py'

homeDir=script_path[:script_path.find('code')+4]
print(homeDir)

import csv
results={}
dataset=None
runnames=[]

logdirs = [homeDir+'/DIFFER/logs']
outfile=os.path.join(logdirs[0],'results.csv')
if os.path.isfile(outfile):
    results_time=os.path.getmtime(outfile)
    print(outfile)
    with open(outfile, mode='r') as file:
        csvFile = csv.reader(file)
        for line in csvFile:
            if len(line)>0:
                if len(line)==1 or line[1]=='':
                    dataset=line[0]
                    results[dataset]=[]
                elif dataset is not None:
                    results[dataset].append(line)
                    runnames.append(line[0])
else:
    results_time=0



#print(results)

#results={'prcc':[],'ltcc':[]}
for logdir in logdirs:
    #os.path.abspath(logdir)
    for root,dirs,files in os.walk(logdir):
        eventFiles=[x for x in files if x.startswith('events')]
        if len(eventFiles):
            #print(dd)
            file=eventFiles[0]
            runName=root.split('/')[-2]
            models=[x for x in os.listdir(root[:root.rfind('/')]) if x.endswith('pth')]
            if len(models)==0 and runName.find('[')>=0:
                #print(dd)
                #print('removing: ',root[:root.rfind('/')])
                #shutil.rmtree(root[:root.rfind('/')])
                continue
            
            if results_time<os.path.getmtime(os.path.join(root,file)) or runName not in runnames:
                print('reading: ',os.path.join(root,file))
                tensorboard_data = read_tensorboard_data(root)
                if 'rank1' in tensorboard_data:
                    if runName in runnames:
                        continue
                    else:
                        runnames.append(runName)
                    datasetName=root.split('/')[-3]
                    if datasetName not in results:
                        results[datasetName]=[]
                   
                    testResults=tensorboard_data['rank1']
                    maxIndex=np.argmax(np.asarray(testResults)[:,2])
                    if 'mAP' in tensorboard_data:
                        results[datasetName].append([runName,maxIndex,testResults[maxIndex][2],tensorboard_data['mAP'][maxIndex][2]])
                    else:
                        results[datasetName].append([runName,maxIndex,testResults[maxIndex][2],-1])
                    print(results[datasetName][-1])
        #print(tensorboard_data.keys())


import csv
def sort_key(item):
    if item[0].find('[')>0:
        return int(item[0].split('[')[0])
    elif item[0].find('_')>0:
        return int(item[0].split('_')[0])
    else:
        return 0
 
fieldnames = ['runname', 'epoch', 'rank1','mAP']
# Open a file and write the data

with open(outfile, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(fieldnames)
    for dataset in results:
        writer.writerow('')
        writer.writerow([dataset])
        data=results[dataset]
        data.sort(key=sort_key)
        writer.writerows(data)
   