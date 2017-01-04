require 'torch'

local t = loadfile "build_model_for_experiments.lua"
t('hfilp', 'adam', 'Y')
t('vflip', 'adam', 'N')
t('rotate', 'adam', 'N')
t('hfilp', 'sgd', 'N')
t('vflip', 'sgd', 'N')
t('rotate', 'sgd', 'N')
