# 首先设置matplotlib配置（必须在所有导入之前）
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
matplotlib.rcParams['text.usetex'] = False  # 禁用LaTeX
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 设置字体


import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'uav'
"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))
"""ostrack"""
'''trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300', dataset_name=dataset_name,
                            run_ids=None, display_name='OSTrack256'))
trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_384_mae_ce_32x4_ep300', dataset_name=dataset_name,
                            run_ids=None, display_name='OSTrack384'))'''
#trackers.extend(trackerlist(name='ostrack', parameter_name='deitt_256_32x4_got10k_ep100', dataset_name=dataset_name,
                           # run_ids=None, display_name='OSTrack256b')) 
trackers.extend(trackerlist(name='ostrack', parameter_name='selat_256_32x4_got10k_ep100_6_11', dataset_name=dataset_name,
                            run_ids=None, display_name='OSTrack256b')) 


##dataset = get_dataset(dataset_name)
dataset = get_dataset('uav')
#dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
plot_results(trackers, dataset, 'uav', merge_results=True, plot_types=('success', 'norm_prec'),
              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, 'uav', merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))
