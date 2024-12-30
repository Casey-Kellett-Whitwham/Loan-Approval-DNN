### Project Averages

import matplotlib.pyplot as plt
import pandas as pd
def Project_Averages(results):
    df_avgs = results.groupby('Project')['Accuracy'].mean()
    project_1_avg = df_avgs[1]
    project_2_avg = df_avgs[2]
    diff = project_2_avg - project_1_avg
    plt.style.use('dark_background')

    plt.figure(figsize=(4, 5))

    bars = plt.bar([1, 2], [project_1_avg, project_2_avg], color=['green', 'green'], alpha=0.7, label='Average Accuracy')

    plt.bar(1, diff, bottom=project_1_avg, color='red', alpha=0.7, label='Difference')

    plt.xlabel('Project')
    plt.ylabel('Accuracy')
    plt.title('Project Average Accuracy Comparison')
    plt.xticks([1, 2], ['Project 1', 'Project 2'], rotation=45, ha='right')

    for bar in bars:
        yval = bar.get_height()
        ybottom = bar.get_y()
        plt.text(bar.get_x() + bar.get_width() / 2, (yval + ybottom) / 2, 
                f'{yval:.2f}', ha='center', va='center', color='white', fontsize=8)

    plt.text(1, project_1_avg + diff / 2, f'Diff: {diff:.2f}', ha='center', va='center', color='black', fontsize=10)
    plt.legend()
    plt.tight_layout()


#### Compare project 2 models with DT from project 1

def dtcomp(results):
    plt.style.use('dark_background')
    project_2_comp = results[results['Project'] == 2].sort_values(by=['Accuracy'], ascending=False)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(project_2_comp['Model'], project_2_comp['Accuracy'], color='green')
    plt.axhline(y=91.16, color='red', linestyle='--', label='Accuracy = 91.16')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Project 2 Accs with line at Project 1 Decision Tree Acc')
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        yval = bar.get_height()
        ybottom = bar.get_y()
        plt.text(bar.get_x() + bar.get_width() / 2, (yval + ybottom) / 2, 
                f'{yval:.2f}', ha='center', va='center', color='white', fontsize=10)

    plt.legend()
    plt.tight_layout()
    plt.show()


### Group based on modle type (avgs)

def groups(results):

    BasicModels = ['DecisionTree', 'LogisticRegression', 'LinearSVC']
    DNN = ['Casey DNN', 'Evan DNN']
    Ensemble = ['GradientBoostingClassifier', 'Stacking', 'HistGradientBoostingClassifier', 'RandomForest', 'BaggingClassifier']

    results['Group'] = results['Model'].apply(
        lambda x: 'BasicModels' if x in BasicModels else ('DNN' if x in DNN else 'Ensemble')
    )

    grouped_averages = results.groupby('Group')['Accuracy'].mean().reset_index()

    return grouped_averages


