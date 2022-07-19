import numpy as np
from interaction_calc import PairInteraction

d14_res = np.array(list(range(265)))
d3_res = np.array(list(range(795, 965)))

pair_calc = PairInteraction(pdb="AtD14_pair_template.pdb", proteinA_res=d14_res, proteinB_res=d3_res)

#pair_calc.namd_setup("AtD14_OsD3_apo_bound_combined.dcd", "AtD14_OsD3_template.conf")

#pair_calc.run_calc()

#pair_calc.plot_results()
top_pairs, top_interactions = pair_calc.interaction_sorter()
print(top_pairs)
print(top_interactions)

np.save("top_pairs_rep.npy",top_pairs)
np.save("top_interactions_rep.npy",top_interactions)

pair_calc.plot_bars()
