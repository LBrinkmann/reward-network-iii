# Qualitative Analysis, Visualization, and Agent-Based Model

## Computation of Bonus Payments

To compute the bonus payments for the participants, run the following notebook:[compute_compensation.ipynb](compute_compensation.ipynb)

## Preprocessing of Data

To download the data from the database run the following notebook: [save_data.ipynb](save_data.ipynb)


To process the data, run the following notebook: [process.ipynb](process.ipynb)

To add the alignment between human and machine actions, run the following notebook: [algorithm/compute_alignment.ipynb](algorithm/compute_alignment.ipynb)

## Visualizations of Experimental Data

To create the visualizations of the experimental data, run the following notebook: [visualize.ipynb](visualize.ipynb)

This will generate the following figures:
* Figure 3: Four prototypical populations [figure 3](plots/experiment/network_compressed_player_score.pdf)
* Figure 4: Evolution of Task Performance, Machine Alignment, and Strategy Descrip-
tion [figure 4](plots/experiment/metrics_overview.pdf)
* Supplementary Figure 8 [sup fig](plots/experiment/network_player_score.pdf)
* Supplementary Figure 9 [sup fig](plots/experiment/network_machine_alignment.pdf)
* Supplementary Figure 10 [sup fig](plots/experiment/network_loss_strategy_int.pdf)

## Agent-Based Model

To run the agent-based model, run the following notebook: [abm.ipynbb](abm.ipynb)

This notebook will run the agent-based model and generate the following figures:
* Figure 5: Agent-Based Model - Boundaries and Uplift [figure 5](plots/abm/heatmap.pdf)
* Supplementary Figure 7 [sup fig](plots/abm/heatmap_details.png)

## Algorithmic Learning Curve

To compute the learning curve of the algorithm (figure 6), run the following notebook: [plot_learning_curve.ipynb](plot_learning_curve.ipynb)

This will generate the following figure:
* Figure 6: Learning Curve of the Algorithm [figure 6](plots/algorithm/algorithm.pdf)
