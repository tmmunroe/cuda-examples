import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from collections import namedtuple
from typing import List

experiments = (
    'arrayadd',
    'arrayaddUnifiedMemory',
    'matmult00',
    'matmult01',
    'vecadd00',
    'vecadd01'
)

def parse_results(results_file) -> List[str]:
    results = []
    currentExperiment = []
    for line in results_file.readlines():
        if line[0] == '#':
            continue

        items = line.split()
        if not items:
            if currentExperiment:
                results.append('\n'.join(currentExperiment))
                currentExperiment = []
            continue

        if not currentExperiment:
            if items[0] not in experiments:
                continue

        currentExperiment.append(line)
            
    return results


def generate_arrayadd_graphs():
    with open('results.txt', 'r') as fin:
        results = parse_results(fin)
        
    results = [result for result in results if 'arrayadd' in result]
    print(f'Found {len(results)} arrayadd results')


    def extract_time(metric_info):
        metrics = metric_info.split(',')
        time_output = metrics[0].replace('Time: ', '').replace('(sec)', '').strip()
        return float(time_output)

    def experiment_results(experiment_name):
        parsed_results = []
        for result in results:
            items = [res for res in result.split('\n') if res]
            meta_info = items[0].split()
            metrics_line = items[1]

            name, k, compute, *_ = meta_info
            if name != experiment_name:
                continue
            threading_arg = meta_info[3] if len(meta_info) > 3 else ''
            elapsed_time = extract_time(metrics_line)

            if threading_arg:
                compute += ' ' + threading_arg

            info = name, compute, int(k), elapsed_time
            parsed_results.append(info)

        return pd.DataFrame(parsed_results, columns=["name", "compute", "k", "time"])
    
    arrayadd_results = experiment_results('arrayadd')
    fig, ax = plt.subplots()
    res = sns.lineplot(data=arrayadd_results, x="k", y="time", hue="compute", style="compute", ax=ax)
    res.set(yscale='log')
    res.set(xticks=list(arrayadd_results['k'].unique()))
    res.set(xticklabels=list(arrayadd_results['k'].unique()))
    res.set(xlabel ="K (millions)", ylabel = "log Time (seconds)", title ='ArrayAdd')
    fig.savefig('arrayadd')

    arrayadd_um_results = experiment_results('arrayaddUnifiedMemory')
    fig, ax = plt.subplots()
    res = sns.lineplot(data=arrayadd_um_results, x="k", y="time", hue="compute", style="compute", ax=ax)
    res.set(yscale='log')
    res.set(xticks=list(arrayadd_um_results['k'].unique()))
    res.set(xticklabels=list(arrayadd_um_results['k'].unique()))
    res.set(xlabel ="K (millions)", ylabel = "log Time (seconds)", title ='ArrayAdd Unified Memory')
    fig.savefig('arrayadd-unified-memory')
    

if __name__ == '__main__':
    generate_arrayadd_graphs()
    
    