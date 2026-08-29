python runs/leduc/leduc_tom_ablation.py --condition cfr --archetype Nit --games 30 --save-prompts;
python runs/leduc/leduc_tom_ablation.py --condition tom_cfr --archetype Nit --games 30 --decay 0.6 --save-prompts;

python runs/leduc/leduc_tom_ablation.py --condition cfr --archetype TAG --games 30 --save-prompts;
python runs/leduc/leduc_tom_ablation.py --condition tom_cfr --archetype TAG --games 30 --decay 0.6 --save-prompts;

python runs/leduc/leduc_tom_ablation.py --condition cfr --archetype LAG --games 30 --save-prompts;
python runs/leduc/leduc_tom_ablation.py --condition tom_cfr --archetype LAG --games 30 --decay 0.6 --save-prompts;

python runs/leduc/leduc_tom_ablation.py --condition cfr --archetype GTO --games 30 --save-prompts;
python runs/leduc/leduc_tom_ablation.py --condition tom_cfr --archetype GTO --games 30 --decay 0.6 --save-prompts;