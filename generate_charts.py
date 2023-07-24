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


def generate_matmult_graphs():
    with open('results.txt', 'r') as fin:
        results = parse_results(fin)
        
    results = [result for result in results if 'matmult' in result]
    print(f'Found {len(results)} matmult results')


    def extract_time(metric_info):
        metrics = metric_info.split(',')
        time_output = metrics[0].replace('Time: ', '').replace('(sec)', '').strip()
        return float(time_output)

    def extract_gflops(metric_info):
        metrics = metric_info.split(',')
        gflops_output = metrics[2].replace('GFlopsS: ', '').strip()
        return float(gflops_output)

    def experiment_results():
        parsed_results = []
        for result in results:
            items = [res for res in result.split('\n') if res]
            meta_info = items[0].split()
            metrics_line = items[5]

            name, k = meta_info
            k = int(k)

            footprint = 16 if '00' in name else 32
            dim = k*footprint
            
            elapsed_time = extract_time(metrics_line)
            gflops = extract_gflops(metrics_line)

            info = name, footprint, dim, elapsed_time, gflops, f"{name} Footprint={footprint}"
            parsed_results.append(info)

        return pd.DataFrame(parsed_results, columns=["name", "footprint", "dim", "time", "gflops", "footprintLabel"])
    
    def add_speedups(pdTable):
        pdTable['speedup'] = 0.0
        for i, row in pdTable.iterrows():
            pdTable.at[i, 'speedup'] = matmult_results.loc[(matmult_results['name'] == 'matmult00') & (matmult_results['dim'] == row['dim'])]['time']/row['time']


    matmult_results = experiment_results()
    add_speedups(matmult_results)

    matmult_results.to_csv('matmult-result-table.csv')

    fig, ax = plt.subplots()
    res = sns.lineplot(data=matmult_results, x="dim", y="time", hue="footprintLabel", style="footprintLabel", ax=ax)
    res.set(xticks=list(matmult_results['dim'].unique()))
    res.set(xticklabels=list(matmult_results['dim'].unique()))
    res.set(xlabel ="Dim", ylabel = "Time (seconds)", title ='matmult')
    fig.savefig('matmult')

    fig, ax = plt.subplots()
    res = sns.lineplot(data=matmult_results, x="dim", y="gflops", hue="footprintLabel", style="footprintLabel", ax=ax)
    res.set(xticks=list(matmult_results['dim'].unique()))
    res.set(xticklabels=list(matmult_results['dim'].unique()))
    res.set(xlabel ="Dim", ylabel = "GFlopsS", title ='matmult')
    fig.savefig('matmult-gflops')
    

    fig, ax = plt.subplots()
    res = sns.lineplot(data=matmult_results, x="dim", y="speedup", hue="footprintLabel", style="footprintLabel", ax=ax)
    res.set(xticks=list(matmult_results['dim'].unique()))
    res.set(xticklabels=list(matmult_results['dim'].unique()))
    res.set(xlabel ="Dim", ylabel = "Speedup", title ='matmult Speedup')
    fig.savefig('matmult-speedup')


def generate_vecadd_graphs():
    with open('results.txt', 'r') as fin:
        results = parse_results(fin)
        
    results = [result for result in results if 'vecadd' in result]
    print(f'Found {len(results)} vecadd results')


    def extract_time(metric_info):
        metrics = metric_info.split(',')
        time_output = metrics[0].replace('Time: ', '').replace('(sec)', '').strip()
        return float(time_output)

    def experiment_results():
        parsed_results = []
        for result in results:
            items = [res for res in result.split('\n') if res]
            meta_info = items[0].split()
            metrics_line = items[2]

            name, k = meta_info
            elapsed_time = extract_time(metrics_line)

            info = name, int(k), elapsed_time
            parsed_results.append(info)

        return pd.DataFrame(parsed_results, columns=["name", "k", "time"])
    
    def add_speedups(pdTable):
        pdTable['speedup'] = 0.0
        for i, row in pdTable.iterrows():
            pdTable.at[i, 'speedup'] = vecadd_results.loc[(vecadd_results['name'] == 'vecadd00') & (vecadd_results['k'] == row['k'])]['time']/row['time']


    vecadd_results = experiment_results()
    add_speedups(vecadd_results)
    vecadd_results.to_csv('vecadd-result-table.csv')
    fig, ax = plt.subplots()
    res = sns.lineplot(data=vecadd_results, x="k", y="time", hue="name", style="name", ax=ax)
    res.set(xticks=list(vecadd_results['k'].unique()))
    res.set(xticklabels=list(vecadd_results['k'].unique()))
    res.set(xlabel ="K", ylabel = "Time (seconds)", title ='VecAdd')
    fig.savefig('vecadd')
    
    fig, ax = plt.subplots()
    res = sns.lineplot(data=vecadd_results, x="k", y="speedup", hue="name", style="name", ax=ax)
    res.set(xticks=list(vecadd_results['k'].unique()))
    res.set(xticklabels=list(vecadd_results['k'].unique()))
    res.set(xlabel ="K", ylabel = "Speedup", title ='VecAdd Speedup')
    fig.savefig('vecadd-speedup')


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
    arrayadd_results.to_csv('array-add-result-table.csv')
    fig, ax = plt.subplots()
    res = sns.lineplot(data=arrayadd_results, x="k", y="time", hue="compute", style="compute", ax=ax)
    res.set(yscale='log')
    res.set(xticks=list(arrayadd_results['k'].unique()))
    res.set(xticklabels=list(arrayadd_results['k'].unique()))
    res.set(xlabel ="K (millions)", ylabel = "log Time (seconds)", title ='ArrayAdd')
    fig.savefig('arrayadd')

    arrayadd_um_results = experiment_results('arrayaddUnifiedMemory')
    arrayadd_um_results.to_csv('arrayadd-unified-memory-result-table.csv')
    fig, ax = plt.subplots()
    res = sns.lineplot(data=arrayadd_um_results, x="k", y="time", hue="compute", style="compute", ax=ax)
    res.set(yscale='log')
    res.set(xticks=list(arrayadd_um_results['k'].unique()))
    res.set(xticklabels=list(arrayadd_um_results['k'].unique()))
    res.set(xlabel ="K (millions)", ylabel = "log Time (seconds)", title ='ArrayAdd Unified Memory')
    fig.savefig('arrayadd-unified-memory')

    

if __name__ == '__main__':
    generate_arrayadd_graphs()
    generate_vecadd_graphs()
    generate_matmult_graphs()
    