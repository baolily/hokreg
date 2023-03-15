#%%
import torch
from mermaid.data_wrapper import AdaptVal
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.util as sku
import os
import matplotlib

import mermaid.example_generation as eg
import mermaid.module_parameters as pars
import mermaid.multiscale_optimizer as MO
import mermaid.registration_networks as RN
import mermaid.forward_models as FM
import mermaid.utils as us

#%%###########################
use_map = True
model_name = 'lddmm_shooting'
map_low_res_factor = 1

if use_map:
    model_name = model_name + '_map'
else:
    model_name = model_name + '_image'


#%%###########################
optimizer_name = 'sgd'
nr_of_iterations = 140
visualize = True
visualize_step = 10

#%%########################
params = pars.ParameterDict()

#%%############################
# I0 = skio.imread('../data/Toy_Template_gra.png')
I0 = skio.imread('../data/wheel_T7.png')
I0 = sku.img_as_float32(I0[np.newaxis, np.newaxis, ...])
# I1 = skio.imread('../data/Toy_Reference_gra.png')
I1 = skio.imread('../data/wheel_R7.png')
I1 = sku.img_as_float32(I1[np.newaxis, np.newaxis, ...])

sz = np.array(I0.shape)
spacing = 1. / (sz[2::] - 1)  # the first two dimensions are batch size and number of image channels

#%%#############################
# Moving from numpy to pytorch
ISource = AdaptVal(torch.from_numpy(I0.copy()))
ITarget = AdaptVal(torch.from_numpy(I1))

#%%############################
# Instantiating the optimizer
from imp import reload
reload(RN)
reload(FM)

so = MO.SingleScaleRegistrationOptimizer(sz,spacing,use_map,map_low_res_factor,params)
model_name = 'lddmm_shooting_map'
model_name += 'multik'
so.add_model(model_name, RN.LDDMMShootingVectorMomentumMapMultiKNet, RN.LDDMMShootingVectorMomentumMapMultiKLoss, use_map=True)
so.set_model(model_name)
so.set_optimizer_by_name( optimizer_name )
so.set_visualization( visualize )
so.set_visualize_step( visualize_step )

so.set_number_of_iterations(nr_of_iterations)

so.set_source_image(ISource)
so.set_target_image(ITarget)

so.set_recording_step(1)
# and now do the optimization
so.optimize()

#%%###########################
# Plotting some results
h = so.get_history()

plt.clf()
e_p, = plt.plot(h['energy'], label='energy')
s_p, = plt.plot(h['similarity_energy'], label='similarity_energy')
r_p, = plt.plot(h['regularization_energy'], label='regularization_energy')
plt.legend(handles=[e_p, s_p, r_p])
plt.show()

# %%
