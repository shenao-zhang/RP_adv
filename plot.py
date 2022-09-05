import matplotlib.pyplot as plt
plt.style.use('bmh')
from svg.analysis import plot_exp, plot_ablation, sweep_summary
#d = '/Users/shenao/Documents/mbrl_code/my_svg/RPgrad/exp/local/2022.07.10/0535_test'
d = '/Users/shenao/Documents/mbrl_code/my_svg/RPgrad/exp/local/2022.07.17/2049_test'
all_summary, groups, configs = sweep_summary(d)

#env_raws = ['mbpo_hopper']
env_raws = ['mbpo_walker2d']
#env_raws = ['mbpo_ant', 'mbpo_cheetah', 'mbpo_hopper', 'mbpo_walker2d', 'mbpo_humanoid']
for env in env_raws:
    I = (all_summary['env_name'] == env)
    t = all_summary[I]

    # Can append more directories for other ablations
    groups = []
    groups.append({'roots': t.d.values})

    env_pretty = env.split('_')[1].title()
    plot_ablation(
        groups=groups,
        save=f'./data/mbpo_{env}.pdf',
        title=env_pretty,
        legend=False,
    )
